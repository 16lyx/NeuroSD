import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from openstl.api import BaseExperiment
from openstl.utils import (
    create_parser,
    default_parser,
    load_config,
    setup_multi_processes,
)
from openstl.utils.experiment_log import (
    profile_model,
    read_json,
    save_experiment_metadata,
    sha256_file,
    write_json,
)


SUPPORTED_METHODS = {
    'neurosd',
    'convlstm',
    'predrnn',
    'predrnnpp',
    'predrnnv2',
    'e3dlstm',
    'mim',
    'mau',
    'phydnet',
    'prednet',
    'simvp',
    'tau',
    'resnet',
    'unet',
    'vit',
    'cno',
    'fno',
    'mmvp',
    'earthformer',
    'predformer',
}


def parse_args():
    parser = create_parser()

    for action in parser._actions:
        if action.dest == 'method' and action.choices is not None:
            choices = list(action.choices)

            if 'neurosd' not in choices:
                choices.append('neurosd')

            action.choices = choices

    args = parser.parse_args()

    config = vars(args).copy()
    config.update(load_config(args.config_file))

    explicit = set()

    for token in sys.argv[1:]:
        option = token.split('=', 1)[0]
        action = parser._option_string_actions.get(option)

        if action is not None:
            explicit.add(action.dest)

    for key in explicit:
        config[key] = getattr(args, key)

    for key, value in default_parser().items():
        if config.get(key) is None:
            config[key] = value

    extra_defaults = {
        'selection_metric': 'mse_pixel',
        'log_every_updates': 50,
        'profile_warmup': 10,
        'profile_repeats': 30,
        'research_notes': {},
    }

    for key, value in extra_defaults.items():
        if config.get(key) is None:
            config[key] = value

    config['method'] = config['method'].lower()
    config['use_gpu'] = str(config['device']).startswith('cuda')
    config['no_display_method_info'] = True

    if config['method'] not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method: {config['method']}"
        )

    if config['dist'] or config['launcher'] != 'none':
        raise ValueError(
            'This entry point supports single-device experiments only.'
        )

    if config['fp16']:
        raise ValueError('This configuration requires FP32.')

    if config['auto_resume']:
        raise ValueError(
            'Use --resume_from to explicitly specify '
            'the original experiment checkpoint.'
        )

    if config['log_step'] != 1:
        raise ValueError(
            'Set log_step=1 for evaluation after every epoch.'
        )

    if config['selection_metric'] != 'mse_pixel':
        raise ValueError(
            'Set selection_metric="mse_pixel" for '
            'MSE-based checkpoint selection.'
        )

    if not config.get('metric_clip_prediction', True):
        raise ValueError(
            'SSIM and PSNR require predictions clipped to [0,1].'
        )

    if config['log_every_updates'] < 1:
        raise ValueError('log_every_updates must be at least 1.')

    if config['profile_warmup'] < 0:
        raise ValueError('profile_warmup must be nonnegative.')

    if config['profile_repeats'] < 2:
        raise ValueError('profile_repeats must be at least 2.')

    if config['use_gpu'] and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was requested, but no CUDA device is available.'
        )

    config['total_length'] = (
        config['pre_seq_length'] + config['aft_seq_length']
    )

    return type(args)(**config)


def check_test_config(args, output):
    saved_config_path = output / 'config_resolved.json'

    if not saved_config_path.is_file():
        raise FileNotFoundError(
            f'The resolved training configuration was not found: '
            f'{saved_config_path}'
        )

    saved = read_json(saved_config_path)

    keys = (
        'method',
        'dataname',
        'pre_seq_length',
        'aft_seq_length',
        'hid_channels',
        'modes1',
        'modes2',
        'in_shape',
        'model_type',
        'hid_S',
        'hid_T',
        'N_S',
        'N_T',
        'spatio_kernel_enc',
        'spatio_kernel_dec',
        'ode_method',
        'ode_time_end',
        'use_fno',
        'use_ode',
        'use_e3d',
        'use_ca',
        'use_rnn',
    )

    for key in keys:
        current = getattr(args, key, None)
        previous = saved.get(key)

        if isinstance(current, tuple):
            current = list(current)

        if isinstance(previous, tuple):
            previous = list(previous)

        if previous != current:
            raise ValueError(
                f'Test configuration differs from training: '
                f'{key}, saved={previous!r}, current={current!r}'
            )


def load_best_checkpoint(exp, output):
    checkpoint = output / 'checkpoint.pth'

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f'Best checkpoint not found: {checkpoint}'
        )

    state = torch.load(checkpoint, map_location=exp.device)
    exp._load_from_state_dict(state)

    print(f'Loaded checkpoint: {checkpoint}', flush=True)

    return checkpoint


def main():
    args = parse_args()

    output = (
        Path(args.res_dir).expanduser() / args.ex_name
    ).resolve()

    resume_from = getattr(args, 'resume_from', None)

    if resume_from is not None:
        resume_checkpoint = (
            Path(resume_from).expanduser().resolve()
        )

        if not resume_checkpoint.is_file():
            raise FileNotFoundError(
                f'Resume checkpoint not found: {resume_checkpoint}'
            )

        if resume_checkpoint.parent != output / 'checkpoints':
            raise ValueError(
                'The resume checkpoint must be inside the '
                'checkpoints directory of the original experiment. '
                f'Experiment: {output}; '
                f'checkpoint: {resume_checkpoint}'
            )

    if (
        not args.test
        and (output / 'config_resolved.json').exists()
        and resume_from is None
    ):
        raise FileExistsError(
            f'Experiment records already exist in {output}. '
            'Use a new ex_name to avoid mixing different runs.'
        )

    if args.test:
        check_test_config(args, output)

    args.tb_dir = str(output / 'tensorboard')

    setup_multi_processes(vars(args))

    exp = BaseExperiment(args)

    from openstl.methods.baseline_logging import attach_baseline_logs

    attach_baseline_logs(exp)

    try:
        if not args.test:
            save_experiment_metadata(exp, ROOT)

            start = time.perf_counter()

            try:
                exp.train()
            except Exception as exc:
                write_json(
                    output / 'run_status.json',
                    {
                        'status': 'failed',
                        'stage': 'training',
                        'exception': repr(exc),
                        'elapsed_seconds': (
                            time.perf_counter() - start
                        ),
                    },
                )
                raise

            write_json(
                output / 'training_time.json',
                {
                    'completed_epochs': exp._epoch + 1,
                    'planned_epochs': args.epoch,
                    'train_wall_seconds_including_evaluation_and_io': (
                        time.perf_counter() - start
                    ),
                    'note': (
                        'sum(train_epoch.train_seconds) gives '
                        'the logged training-loop time '
                        'excluding validation'
                    ),
                },
            )

        checkpoint = load_best_checkpoint(exp, output)

        final_metrics = exp.method.save_final_test(exp)
        efficiency = profile_model(exp)

        best_record_path = output / 'best_record.json'

        report = {
            'status': 'completed',
            'seed': args.seed,
            'best_record': (
                read_json(best_record_path)
                if best_record_path.exists() else None
            ),
            'final_evaluation': final_metrics,
            'efficiency': efficiency,
            'best_checkpoint_sha256': sha256_file(checkpoint),
        }

        write_json(output / 'run_summary.json', report)
        write_json(
            output / 'run_status.json',
            {'status': 'completed'},
        )

        print(f'Experiment results saved to: {output}')

    finally:
        if exp.writer:
            exp.writer.close()


if __name__ == '__main__':
    main()