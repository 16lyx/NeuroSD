import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from pytorch_msssim import ssim
from skimage.metrics import structural_similarity
from timm.utils import AverageMeter

from .base_method import Base_method
from openstl.models.neurosd_model import ST_E3D_ODE_FNO
from openstl.utils import print_log
from openstl.utils.experiment_log import (
    append_csv,
    memory_now,
    replace_csv,
    reset_gpu_peak,
    sync_device,
    write_json,
)


def terminal_progress(iterable, desc):
    terminal = None

    try:
        try:
            terminal = open(
                '/dev/tty',
                'w',
                buffering=1,
                encoding='utf-8',
            )
        except OSError:
            terminal = None

        with tqdm(
            iterable,
            desc=desc,
            total=len(iterable),
            file=terminal,
            disable=terminal is None,
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
        ) as progress:
            yield from progress
    finally:
        if terminal is not None:
            terminal.close()


class ForecastLoss(nn.Module):
    def __init__(self, lambda_ssim):
        super().__init__()

        if lambda_ssim < 0:
            raise ValueError('lambda_ssim must be nonnegative.')

        self.lambda_ssim = lambda_ssim

    def components(self, pred, true):
        mse = F.mse_loss(pred, true)

        if self.lambda_ssim > 0:
            b, t, c, h, w = pred.shape

            score = ssim(
                pred.reshape(b * t, c, h, w),
                true.reshape(b * t, c, h, w),
                data_range=1.0,
                size_average=True,
            )

            total = mse + self.lambda_ssim * (1 - score)
        else:
            score = mse.new_tensor(float('nan'))
            total = mse

        return total, mse, score

    def forward(self, pred, true):
        return self.components(pred, true)[0]


class NeuroSD(Base_method):
    def __init__(self, args, device, steps_per_epoch):
        super().__init__(args, device, steps_per_epoch)

        if args.dist:
            raise NotImplementedError(
                'This implementation supports single-device experiments only.'
            )

        if args.fp16:
            raise ValueError(
                'This configuration requires FP32 training.'
            )

        if not getattr(args, 'metric_clip_prediction', True):
            raise ValueError(
                'SSIM and PSNR require predictions clipped to [0,1].'
            )

        if args.selection_metric != 'mse_pixel':
            raise ValueError(
                'Set selection_metric="mse_pixel" for MSE-based '
                'checkpoint selection.'
            )

        self.model = ST_E3D_ODE_FNO(
            in_channels=args.in_shape[1],
            hid_channels=args.hid_channels,
            out_frames=args.aft_seq_length,
            modes1=args.modes1,
            modes2=args.modes2,
            ode_method=args.ode_method,
            ode_time_end=args.ode_time_end,
            use_fno=getattr(args, 'use_fno', True),
            use_ode=getattr(args, 'use_ode', True),
            use_e3d=getattr(args, 'use_e3d', True),
            use_ca=getattr(args, 'use_ca', True),
            use_rnn=getattr(args, 'use_rnn', True),
        ).to(device)

        self.scheduler_step_at_epoch_end = (
            args.sched.lower() == 'torch_cosine'
        )

        if self.scheduler_step_at_epoch_end:
            self.model_optim = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.model_optim,
                T_max=args.epoch,
                eta_min=args.min_lr,
            )

            self.by_epoch = True
        else:
            self.model_optim, self.scheduler, self.by_epoch = (
                self._init_optimizer(steps_per_epoch)
            )

        self.criterion = ForecastLoss(args.lambda_ssim)
        self.best_score = float('inf')
        self.last_train = {}

        if getattr(args, 'ns_long_term', False):
            if not (
                hasattr(self.model, 'out_frames')
                and hasattr(self.model, 'ode_time_end')
            ):
                raise TypeError(
                    'The model must expose out_frames and ode_time_end.'
                )

            self.model.out_frames = int(args.aft_seq_length)
            self.model.ode_time_end = float(args.ode_time_end)

            print(
                f'[Long-term NeuroSD] '
                f'input={args.pre_seq_length}, '
                f'output={self.model.out_frames}, '
                f'ode_time_end={self.model.ode_time_end}',
                flush=True,
            )

    def _predict(self, batch_x, batch_y=None, **kwargs):
        return self.model(batch_x)

    def train_one_epoch(
        self,
        runner,
        train_loader,
        epoch,
        num_updates,
        eta=None,
        **kwargs,
    ):
        self.model.train()

        if self.by_epoch and not self.scheduler_step_at_epoch_end:
            self.scheduler.step(epoch)

        meters = {
            key: AverageMeter()
            for key in ('loss', 'mse', 'ssim', 'nfe')
        }

        lr_start = self.current_lr()[0]
        used_lrs = []

        sync_device(self.device)
        reset_gpu_peak(self.device)
        start = time.perf_counter()

        for batch_index, (batch_x, batch_y) in enumerate(
            terminal_progress(
                train_loader,
                desc=f'Train {epoch + 1}/{self.args.epoch}',
            )
        ):
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            runner.call_hook('before_train_iter')
            self.model_optim.zero_grad(set_to_none=True)

            used_lr = self.current_lr()[0]
            used_lrs.append(used_lr)

            pred = self._predict(batch_x)

            if pred.shape != batch_y.shape:
                raise RuntimeError(
                    f'Prediction shape mismatch: '
                    f'{pred.shape} vs {batch_y.shape}'
                )

            loss, mse, score = self.criterion.components(pred, batch_y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f'Non-finite loss at epoch={epoch + 1}, '
                    f'batch={batch_index + 1}.'
                )

            loss.backward()

            if self.clip_mode == 'norm' and self.clip_value is not None:
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_value,
                )
                last_grad_norm = float(norm)
            else:
                self.clip_grads(self.model.parameters())
                last_grad_norm = None

            self.model_optim.step()

            if not self.by_epoch:
                self.scheduler.step()

            n = batch_x.size(0)
            meters['loss'].update(loss.item(), n)
            meters['mse'].update(mse.item(), n)
            meters['nfe'].update(self.model.nfe, n)

            if torch.isfinite(score):
                meters['ssim'].update(score.item(), n)

            num_updates += 1
            runner._iter += 1
            runner.call_hook('after_train_iter')

            if (
                batch_index % self.args.log_every_updates == 0
                or batch_index + 1 == len(train_loader)
            ):
                append_csv(
                    os.path.join(runner.path, 'train_updates.csv'),
                    {
                        'epoch': epoch + 1,
                        'batch': batch_index + 1,
                        'global_update': num_updates,
                        'lr_used': used_lr,
                        'loss': loss.item(),
                        'mse_pixel': mse.item(),
                        'ssim_loss_impl': (
                            score.item()
                            if torch.isfinite(score) else None
                        ),
                        'grad_norm_before_clip': last_grad_norm,
                        'forward_nfe': self.model.nfe,
                    },
                )

        if not used_lrs:
            raise RuntimeError('The training loader is empty.')

        if self.scheduler_step_at_epoch_end:
            self.scheduler.step()

        sync_device(self.device)
        seconds = time.perf_counter() - start

        row = {
            'epoch': epoch + 1,
            'seed': self.args.seed,
            'global_update': num_updates,
            'train_samples_seen': int(meters['loss'].count),
            'lr_start': lr_start,
            'lr_used_min': min(used_lrs),
            'lr_used_max': max(used_lrs),
            'lr_next': self.current_lr()[0],
            'train_loss': meters['loss'].avg,
            'train_mse_pixel': meters['mse'].avg,
            'train_ssim_loss_impl': (
                meters['ssim'].avg
                if meters['ssim'].count else None
            ),
            'forward_nfe_mean': meters['nfe'].avg,
            'train_seconds': seconds,
            'train_sequences_per_second': (
                meters['loss'].count / seconds
            ),
            **memory_now(self.device),
        }

        self.last_train = row

        append_csv(
            os.path.join(runner.path, 'train_epoch.csv'),
            row,
        )

        print_log(
            f'[Train] epoch={epoch + 1}/{self.args.epoch} '
            f"loss={row['train_loss']:.6f} "
            f"MSE_pixel={row['train_mse_pixel']:.6f} "
            f"LR_used=[{row['lr_used_min']:.3e},"
            f"{row['lr_used_max']:.3e}] "
            f'time={seconds:.2f}s '
            f"NFE={row['forward_nfe_mean']:.1f}"
        )

        return num_updates, meters['loss'], eta

    @torch.no_grad()
    def evaluate_stream(self, loader, save_dir=None):
        self.model.eval()

        if loader.drop_last:
            raise ValueError(
                'Evaluation requires drop_last=False.'
            )

        horizon = self.args.aft_seq_length

        sums = {
            key: np.zeros(horizon, dtype=np.float64)
            for key in ('mse_pixel', 'mae_pixel', 'ssim', 'psnr')
        }

        normalization = getattr(loader.dataset, 'normalization', None)

        if normalization is not None:
            sums['relative_l2_raw'] = np.zeros(
                horizon,
                dtype=np.float64,
            )

        total_samples = 0
        loss_sum = 0.0
        nfe_sum = 0.0
        npy_arrays = None
        pixel_count = None

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        sync_device(self.device)
        reset_gpu_peak(self.device)
        start = time.perf_counter()

        for batch_x, batch_y in terminal_progress(
            loader,
            desc='Evaluating',
        ):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            pred = self._predict(batch_x)

            if pred.shape != batch_y.shape:
                raise RuntimeError(
                    f'Evaluation shape mismatch: '
                    f'{pred.shape} vs {batch_y.shape}'
                )

            if pred.ndim != 5 or pred.shape[1] != horizon:
                raise RuntimeError(
                    'Expected predictions with shape [B,T,C,H,W] '
                    'matching the configured prediction horizon.'
                )

            if (
                not torch.isfinite(pred).all()
                or not torch.isfinite(batch_y).all()
            ):
                raise RuntimeError(
                    'Predictions or targets contain non-finite values.'
                )

            current_pixel_count = int(np.prod(batch_y.shape[2:]))

            if pixel_count is None:
                pixel_count = current_pixel_count
            elif pixel_count != current_pixel_count:
                raise RuntimeError(
                    'The spatial or channel dimensions changed '
                    'during evaluation.'
                )

            n = batch_x.size(0)
            loss_sum += self.criterion(pred, batch_y).item() * n
            nfe_sum += self.model.nfe * n

            p = pred.cpu().numpy().astype(np.float64)
            y = batch_y.cpu().numpy().astype(np.float64)

            if save_dir is not None:
                batch_arrays = {
                    'inputs': batch_x.detach().cpu().numpy(),
                    'preds': p,
                    'trues': y,
                }

                if npy_arrays is None:
                    npy_arrays = {
                        name: np.lib.format.open_memmap(
                            os.path.join(save_dir, f'{name}.npy'),
                            mode='w+',
                            dtype=np.float32,
                            shape=(
                                len(loader.dataset),
                                *values.shape[1:],
                            ),
                        )
                        for name, values in batch_arrays.items()
                    }

                end_index = total_samples + n

                for name, values in batch_arrays.items():
                    npy_arrays[name][total_samples:end_index] = values

            diff = p - y

            sums['mse_pixel'] += (
                np.square(diff).mean(axis=(2, 3, 4)).sum(axis=0)
            )

            sums['mae_pixel'] += (
                np.abs(diff).mean(axis=(2, 3, 4)).sum(axis=0)
            )

            if normalization is not None:
                scale = normalization['scale']
                offset = normalization['min']

                pred_raw = p * scale + offset
                true_raw = y * scale + offset

                error_norm = np.sqrt(
                    np.square(pred_raw - true_raw).sum(
                        axis=(2, 3, 4)
                    )
                )

                target_norm = np.sqrt(
                    np.square(true_raw).sum(axis=(2, 3, 4))
                )

                relative_l2 = error_norm / np.maximum(
                    target_norm,
                    1e-12,
                )

                sums['relative_l2_raw'] += relative_l2.sum(axis=0)

            p_image = np.clip(p, 0.0, 1.0)

            image_mse = np.square(
                p_image - y
            ).mean(axis=(2, 3, 4))

            with np.errstate(divide='ignore'):
                sums['psnr'] += (
                    -10.0 * np.log10(image_mse)
                ).sum(axis=0)

            for b in range(n):
                for t in range(horizon):
                    channel_scores = [
                        structural_similarity(
                            y[b, t, c],
                            p_image[b, t, c],
                            data_range=1.0,
                            win_size=7,
                        )
                        for c in range(y.shape[2])
                    ]

                    sums['ssim'][t] += np.mean(channel_scores)

            total_samples += n

        if not total_samples:
            raise RuntimeError('The evaluation loader is empty.')

        if total_samples != len(loader.dataset):
            raise RuntimeError(
                f'Evaluation sample count mismatch: '
                f'{total_samples}/{len(loader.dataset)}'
            )

        if npy_arrays is not None:
            for array in npy_arrays.values():
                array.flush()

            write_json(
                os.path.join(save_dir, 'npy_info.json'),
                {
                    'num_samples': total_samples,
                    'dtype': 'float32',
                    'axis_order': (
                        '[sample,time,channel,height,width]'
                    ),
                    'shapes': {
                        name: list(array.shape)
                        for name, array in npy_arrays.items()
                    },
                    'prediction_clipped': False,
                    'value_scale': 'dataset loader output scale',
                    'normalization': normalization,
                },
            )

            print_log(f'NPY files saved to: {save_dir}')

        sync_device(self.device)
        seconds = time.perf_counter() - start

        steps = []

        for t in range(horizon):
            row = {
                'step': t + 1,
                **{
                    key: float(values[t] / total_samples)
                    for key, values in sums.items()
                },
            }

            row['mse'] = row['mse_pixel'] * pixel_count
            row['mae'] = row['mae_pixel'] * pixel_count
            steps.append(row)

        summary = {
            key: float(np.mean([row[key] for row in steps]))
            for key in (
                'mse_pixel',
                'mae_pixel',
                'mse',
                'mae',
                'ssim',
                'psnr',
            )
        }

        summary.update({
            'loss': loss_sum / total_samples,
            'num_samples': total_samples,
            'forward_nfe_mean': nfe_sum / total_samples,
            'evaluation_seconds_including_metrics': seconds,
            **memory_now(self.device),
        })

        if normalization is not None:
            scale = normalization['scale']

            summary['relative_l2_raw'] = float(
                np.mean([row['relative_l2_raw'] for row in steps])
            )

            summary['mse_pixel_raw'] = (
                summary['mse_pixel'] * scale**2
            )
            summary['mae_pixel_raw'] = (
                summary['mae_pixel'] * abs(scale)
            )

            for row in steps:
                row['mse_pixel_raw'] = row['mse_pixel'] * scale**2
                row['mae_pixel_raw'] = row['mae_pixel'] * abs(scale)

        return summary, steps

    def vali_one_epoch(self, runner, vali_loader, **kwargs):
        summary, steps = self.evaluate_stream(vali_loader)

        epoch = runner._epoch + 1
        selection = self.args.selection_metric

        if selection != 'mse_pixel':
            raise ValueError(
                'Set selection_metric="mse_pixel" for MSE-based '
                'checkpoint selection.'
            )

        score = summary[selection]

        if not np.isfinite(score):
            raise RuntimeError(
                'The checkpoint selection metric is non-finite.'
            )

        is_best = score <= self.best_score

        if is_best:
            self.best_score = score

            write_json(
                os.path.join(runner.path, 'best_record.json'),
                {
                    'best_epoch': epoch,
                    'selection_metric': selection,
                    'selection_mode': 'min',
                    'best_score': score,
                    'seed': self.args.seed,
                    'metrics': summary,
                    'checkpoint_path': os.path.join(
                        runner.path,
                        'checkpoint.pth',
                    ),
                },
            )

            replace_csv(
                os.path.join(runner.path, 'best_eval_per_step.csv'),
                steps,
            )

        row = {
            'epoch': epoch,
            'selection_metric': selection,
            'selection_score': score,
            'is_best': is_best,
            **summary,
        }

        append_csv(
            os.path.join(runner.path, 'eval_epoch.csv'),
            row,
        )

        append_csv(
            os.path.join(runner.path, 'epoch_summary.csv'),
            {
                **self.last_train,
                **{
                    f'eval_{key}': value
                    for key, value in row.items()
                    if key != 'epoch'
                },
            },
        )

        log = (
            f'epoch={epoch} '
            f"MSE={summary['mse']:.6f} "
            f"MAE={summary['mae']:.6f} "
            f"SSIM={summary['ssim']:.6f} "
            f"PSNR={summary['psnr']:.4f} "
            f'selection={selection}:{score:.6f} '
            f'best={is_best}'
        )

        if runner.writer:
            for key in ('mse', 'mae', 'ssim', 'psnr', 'loss'):
                runner.writer.add_scalar(
                    f'evaluation/{key}',
                    summary[key],
                    epoch,
                )

            if 'train_seconds' in self.last_train:
                runner.writer.add_scalar(
                    'runtime/train_seconds',
                    self.last_train['train_seconds'],
                    epoch,
                )

        results = {
            'mse': np.array([summary['mse']]),
            'loss': np.array([score]),
        }

        return results, log

    def save_final_test(self, runner):
        save_dir = os.path.join(runner.path, 'saved')

        summary, steps = self.evaluate_stream(
            runner.test_loader,
            save_dir=save_dir,
        )

        shared = (
            runner.vali_loader.dataset
            is runner.test_loader.dataset
        )

        write_json(
            os.path.join(runner.path, 'test_metrics.json'),
            {
                'seed': self.args.seed,
                'checkpoint': 'checkpoint.pth',
                'validation_test_same_dataset_object': shared,
                'independent_final_test': False if shared else None,
                **summary,
            },
        )

        replace_csv(
            os.path.join(runner.path, 'test_per_step.csv'),
            steps,
        )

        print_log(f'[Final evaluation] {summary}')
        return summary