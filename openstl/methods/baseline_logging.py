import ast
import inspect
import textwrap
import time
from types import MethodType
from pathlib import Path

import torch
from torch import nn

from .neurosd import NeuroSD, terminal_progress
from openstl.utils import print_log
from openstl.utils.experiment_log import (
    append_csv,
    memory_now,
    sync_device,
    reset_gpu_peak,
)


SUPPORTED = {
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
    'predformer',
    'PredFormer',
}


class ForecastView(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.network = method.model
        object.__setattr__(self, 'method', method)
        self.train(self.network.training)

    def forward(self, x):
        return self.method._predict(x)


class _Progress:
    def __init__(self, loader, desc):
        self.loader = loader
        self.desc = desc

    def __iter__(self):
        yield from terminal_progress(
            self.loader,
            desc=self.desc,
        )

    def set_description(self, *args, **kwargs):
        pass


def _predict(self, x, y=None, **kwargs):
    shape = (
        x.shape[0],
        self.args.aft_seq_length,
        *x.shape[2:],
    )

    dummy = x.new_zeros(shape)
    result = self._bl_predict(x, dummy, **kwargs)

    if tuple(result.shape) != shape:
        raise RuntimeError(
            f'Unexpected prediction shape: '
            f'{tuple(result.shape)} != {shape}'
        )

    return result


def _batch(self, runner, epoch, updates, loss, x, y, values):
    if not torch.isfinite(loss.detach()).all():
        raise RuntimeError(
            f'Non-finite loss: epoch={epoch + 1}, '
            f'update={updates + 1}'
        )

    state = self._bl_state
    state['batches'] += 1

    batch_size = x.shape[0]
    state['samples'] += batch_size

    lr = self.current_lr()[0]
    state['lrs'].append(lr)

    mse = None
    prediction = values.get('pred_y')

    if prediction is None:
        prediction = values.get('img_gen')

        if isinstance(prediction, torch.Tensor) and prediction.ndim == 5:
            if tuple(prediction.shape[2:]) != tuple(y.shape[2:]):
                from openstl.utils.predrnn_utils import (
                    reshape_patch_back,
                )

                patch_size = getattr(self.args, 'patch_size', 1)

                patch_shape = (
                    y.shape[-2] // patch_size,
                    y.shape[-1] // patch_size,
                    patch_size * patch_size * y.shape[2],
                )

                if tuple(prediction.shape[2:]) == patch_shape:
                    prediction = reshape_patch_back(
                        prediction.detach(),
                        patch_size,
                    ).permute(0, 1, 4, 2, 3)

            prediction = prediction[:, -y.shape[1]:]

    if isinstance(prediction, torch.Tensor) and prediction.shape == y.shape:
        mse = float(
            (
                prediction.detach().float()
                - y.detach().float()
            ).square().mean()
        )

        state['mse_sum'] += mse * batch_size
        state['mse_n'] += batch_size

    should_log = (
        (state['batches'] - 1) % self.args.log_every_updates == 0
        or state['batches'] == state['total']
    )

    if should_log:
        norm = None

        if self.args.clip_grad is not None:
            norms = [
                parameter.grad.detach().float().norm()
                for parameter in self.model.parameters()
                if parameter.grad is not None
            ]

            if norms:
                norm = float(torch.stack(norms).norm())

        append_csv(
            Path(runner.path) / 'train_updates.csv',
            {
                'epoch': epoch + 1,
                'batch': state['batches'],
                'global_update': updates + 1,
                'lr_used': lr,
                'loss': float(loss.detach()),
                'mse_pixel': mse,
                'ssim_loss_impl': None,
                'grad_norm_before_clip': norm,
                'forward_nfe': None,
            },
        )


def _instrument(method):
    original = type(method).train_one_epoch

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(original))
    )

    class Edit(ast.NodeTransformer):
        inserted = 0
        phy_model = 0
        phy_criterion = 0

        def visit_Assign(self, node):
            if (
                method.args.method == 'phydnet'
                and len(node.targets) == 1
            ):
                name = (
                    node.targets[0].id
                    if isinstance(node.targets[0], ast.Name)
                    else None
                )

                call = (
                    ast.unparse(node.value.func)
                    if isinstance(node.value, ast.Call)
                    else ''
                )

                if name == 'pred_y' and call == 'self.model':
                    node.targets[0].id = 'loss'
                    self.phy_model += 1

                elif name == 'loss' and call == 'self.criterion':
                    self.phy_criterion += 1
                    return None

            return self.generic_visit(node)

        def visit_Expr(self, node):
            if ast.unparse(node.value) == 'loss.backward()':
                self.inserted += 1

                callback = ast.parse(
                    'self._baseline_batch('
                    'runner, epoch, num_updates, loss, '
                    'batch_x, batch_y, locals())'
                ).body[0]

                return [node, callback]

            return self.generic_visit(node)

        def visit_Call(self, node):
            if ast.unparse(node.func) == 'torch.cuda.synchronize':
                replacement = ast.parse(
                    '_baseline_sync(self.device)'
                ).body[0].value

                return ast.copy_location(replacement, node)

            return self.generic_visit(node)

    edit = Edit()
    tree = edit.visit(tree)

    if edit.inserted != 1:
        raise RuntimeError(
            'The training function differs from the expected version: '
            + original.__qualname__
        )

    if (
        method.args.method == 'phydnet'
        and (edit.phy_model, edit.phy_criterion) != (1, 1)
    ):
        raise RuntimeError(
            'The PhyDNet source differs from the expected implementation.'
        )

    ast.fix_missing_locations(tree)

    scope = dict(original.__globals__)
    scope['_baseline_sync'] = sync_device
    scope['tqdm'] = lambda loader: _Progress(
        loader,
        f'Train {method._bl_epoch + 1}/{method.args.epoch}',
    )

    exec(
        compile(tree, inspect.getfile(original), 'exec'),
        scope,
    )

    return scope['train_one_epoch']


def _train(
    self,
    runner,
    loader,
    epoch,
    updates,
    eta=None,
    **kwargs,
):
    self._bl_epoch = epoch
    self._bl_state = {
        'batches': 0,
        'samples': 0,
        'lrs': [],
        'mse_sum': 0.0,
        'mse_n': 0,
        'total': len(loader),
    }

    lr_start = self.current_lr()[0]

    sync_device(self.device)
    reset_gpu_peak(self.device)
    start = time.perf_counter()

    result = self._bl_train(
        self,
        runner,
        loader,
        epoch,
        updates,
        eta,
        **kwargs,
    )

    sync_device(self.device)
    seconds = time.perf_counter() - start
    state = self._bl_state

    if (
        state['batches'] != len(loader)
        or result[0] - updates != state['batches']
    ):
        raise RuntimeError(
            'The training update count does not match the logged count.'
        )

    row = {
        'epoch': epoch + 1,
        'seed': self.args.seed,
        'global_update': result[0],
        'train_samples_seen': state['samples'],
        'lr_start': lr_start,
        'lr_used_min': min(state['lrs']),
        'lr_used_max': max(state['lrs']),
        'lr_next': self.current_lr()[0],
        'train_loss': float(result[1].avg),
        'train_mse_pixel': (
            state['mse_sum'] / state['mse_n']
            if state['mse_n'] else None
        ),
        'train_ssim_loss_impl': None,
        'forward_nfe_mean': None,
        'train_seconds': seconds,
        'train_sequences_per_second': state['samples'] / seconds,
        **memory_now(self.device),
    }

    self.last_train = row

    append_csv(
        Path(runner.path) / 'train_epoch.csv',
        row,
    )

    print_log(
        f'[Train] {self.args.method} '
        f'epoch={epoch + 1}/{self.args.epoch} '
        f"loss={row['train_loss']:.6f} "
        f'time={seconds:.2f}s'
    )

    return result


def _evaluation_function():
    original = inspect.unwrap(NeuroSD.evaluate_stream)

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(original))
    )

    tree.body[0].decorator_list = []

    class Counter(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if ast.unparse(node) == 'self.model.nfe':
                return ast.copy_location(
                    ast.Constant(0),
                    node,
                )

            return self.generic_visit(node)

    tree = Counter().visit(tree)
    ast.fix_missing_locations(tree)

    scope = dict(original.__globals__)

    exec(
        compile(tree, inspect.getfile(original), 'exec'),
        scope,
    )

    return torch.no_grad()(scope['evaluate_stream'])


def _evaluate(self, loader, save_dir=None):
    summary, steps = self._bl_eval(
        self,
        loader,
        save_dir=save_dir,
    )

    summary['forward_nfe_mean'] = None

    return summary, steps


def attach_baseline_logs(exp):
    if exp.args.method not in SUPPORTED:
        return

    method = exp.method

    if getattr(method, '_baseline_logging', False):
        return

    method.model.to(exp.device)

    method._bl_predict = method._predict
    method._bl_train = _instrument(method)
    method._bl_eval = _evaluation_function()

    method.best_score = float('inf')
    method.last_train = {}
    method._baseline_logging = True

    functions = {
        '_predict': _predict,
        '_baseline_batch': _batch,
        'train_one_epoch': _train,
        'evaluate_stream': _evaluate,
        'vali_one_epoch': NeuroSD.vali_one_epoch,
        'save_final_test': NeuroSD.save_final_test,
    }

    for name, function in functions.items():
        setattr(
            method,
            name,
            MethodType(function, method),
        )

    notes = dict(
        getattr(exp.args, 'research_notes', {})
    )

    notes.update({
        'training_loop': (
            'official model-specific loop with logging instrumentation'
        ),
        'evaluation_future_inputs': (
            'zeros; no ground-truth future is fed to the predictor'
        ),
        'non_ODE_nfe': (
            'not applicable; empty log fields'
        ),
        'training_mse': (
            'training-path forecast MSE when available; empty otherwise'
        ),
        'evaluation_loss': (
            'forecast MSE; model-specific full objective '
            'is logged as train_loss'
        ),
    })

    if exp.args.method == 'phydnet':
        notes['phydnet_fix'] = (
            'use scalar loss returned by the model, '
            'including moment constraints'
        )

    exp.args.research_notes = notes