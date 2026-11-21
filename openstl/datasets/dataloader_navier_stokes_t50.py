from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader


SPLIT_RANGES = {
    'train': (0, 1600),
    'val': (2000, 2200),
    'test': (1600, 2000),
}


def read_ns_data(file_path, axis_order=None):
    path = Path(file_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(f'Data file not found: {path}')

    if h5py.is_hdf5(str(path)):
        with h5py.File(path, 'r') as f:
            if 'u' not in f:
                raise KeyError(f'Variable "u" was not found in: {path}')

            data = np.array(f['u'], dtype=np.float32)

        default_order = 'THWN'
        source = 'hdf5_mat'
    else:
        values = sio.loadmat(path, variable_names=['u'])

        if 'u' not in values:
            raise KeyError(f'Variable "u" was not found in: {path}')

        data = np.asarray(values['u'], dtype=np.float32)
        default_order = None
        source = 'scipy_mat'

    raw_shape = tuple(data.shape)

    if data.ndim != 4:
        raise ValueError(
            f'Variable "u" must be four-dimensional, got {raw_shape}.'
        )

    order = default_order if axis_order is None else axis_order

    if order is None:
        raise ValueError(
            f'Please specify axis_order for raw shape {raw_shape}.'
        )

    order = str(order).upper()

    if len(order) != 4 or set(order) != set('NTHW'):
        raise ValueError(
            'axis_order must contain N, T, H, and W exactly once.'
        )

    axes = tuple(order.index(axis) for axis in 'NTHW')
    data = data.transpose(axes)

    if data.shape != (5000, 50, 64, 64):
        raise ValueError(
            'Expected converted T50 data with shape [5000,50,64,64], '
            f'got {data.shape}. Check the file and axis_order.'
        )

    print(
        f'[NS-T50 data] source={source}, '
        f'raw_shape={raw_shape}, axis_order={order}, '
        f'converted_shape={data.shape}',
        flush=True,
    )

    return data


def normalize_ns_data(data, total_length=20):
    if total_length != 20:
        raise ValueError(
            'T50 normalization must use the first 20 training frames '
            'for every prediction horizon.'
        )

    train_values = data[:1600, :20]

    min_val = float(train_values.min())
    max_val = float(train_values.max())

    if not np.isfinite(min_val) or not np.isfinite(max_val):
        raise ValueError(
            'The training normalization data contain NaN '
            'or infinite values.'
        )

    if max_val <= min_val:
        raise ValueError(
            'The training data must have a positive value range.'
        )

    scale = max_val - min_val + 1e-8

    for start in range(0, len(data), 64):
        block = data[start:start + 64]

        if not np.isfinite(block).all():
            raise ValueError(
                f'Non-finite values found near trajectory {start}.'
            )

        block -= min_val
        block /= scale

    normalization = {
        'method': 'train_minmax',
        'min': min_val,
        'max': max_val,
        'scale': float(scale),
        'epsilon': 1e-8,
        'fit_scope': 'training_trajectories_first_20_frames',
        'fit_trajectory_slice': [0, 1600],
        'fit_frame_slice': [0, 20],
        'fit_num_trajectories': 1600,
        'fit_num_frames': 20,
        'clipped': False,
    }

    print(
        f'[NS-T50 normalization] '
        f'fit=train[0:1600], frames=[0:20], '
        f'min={min_val}, max={max_val}, scale={scale}',
        flush=True,
    )

    return data, normalization


class NavierStokesDataset(Dataset):
    def __init__(
        self,
        data,
        normalization,
        mode='train',
        t_in=10,
        t_out=10,
    ):
        super().__init__()

        if mode not in SPLIT_RANGES:
            raise ValueError(
                f'Expected mode in {tuple(SPLIT_RANGES)}, got {mode}.'
            )

        if t_in != 10 or t_out not in (10, 20, 25, 30):
            raise ValueError(
                'T50 requires 10 input frames and '
                '10, 20, 25, or 30 prediction frames.'
            )

        if data.ndim != 4 or data.shape[2:] != (64, 64):
            raise ValueError(
                f'Expected data with shape [N,T,64,64], got {data.shape}.'
            )

        if data.shape[1] < t_in + t_out:
            raise ValueError(
                f'The task requires {t_in + t_out} frames, '
                f'but only {data.shape[1]} are available.'
            )

        start, end = SPLIT_RANGES[mode]

        if len(data) < end:
            raise ValueError(
                f'The {mode} split requires at least {end} trajectories, '
                f'got {len(data)}.'
            )

        self.data = data[start:end]
        self.trajectory_ids = np.arange(start, end, dtype=np.int64)
        self.mode = mode
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.mean = 0.0
        self.std = 1.0

        self.normalization = dict(normalization)
        self.min_val = float(normalization['min'])
        self.max_val = float(normalization['max'])
        self.scale = float(normalization['scale'])

        self.split_info = {
            'split': mode,
            'trajectory_slice': [start, end],
            'input_frame_slice': [0, self.t_in],
            'target_frame_slice': [
                self.t_in,
                self.t_in + self.t_out,
            ],
        }

        print(
            f'[NS-T50 {mode}] trajectories=[{start}:{end}], '
            f'samples={len(self)}, task={self.t_in}->{self.t_out}',
            flush=True,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames = self.data[idx, :self.t_in + self.t_out]

        sequence = torch.from_numpy(
            np.ascontiguousarray(frames)
        ).unsqueeze(1)

        inputs = sequence[:self.t_in]
        targets = sequence[self.t_in:self.t_in + self.t_out]

        return inputs, targets


def load_data(
    batch_size,
    val_batch_size,
    data_root,
    num_workers=4,
    data_file=None,
    axis_order=None,
    pre_seq_length=10,
    aft_seq_length=10,
    in_shape=None,
    distributed=False,
    use_augment=False,
    use_prefetcher=False,
    drop_last=False,
    **kwargs,
):
    if distributed:
        raise NotImplementedError(
            'This loader supports single-device training only.'
        )

    if use_augment or use_prefetcher:
        raise ValueError(
            'Please disable use_augment and use_prefetcher.'
        )

    if pre_seq_length != 10:
        raise ValueError('T50 requires 10 input frames.')

    if aft_seq_length not in (10, 20, 25, 30):
        raise ValueError(
            'The prediction length must be 10, 20, 25, or 30.'
        )

    total_length = pre_seq_length + aft_seq_length

    configured_total = kwargs.get('total_length')

    if configured_total is not None:
        if int(configured_total) != total_length:
            raise ValueError(
                f'Expected total_length={total_length}, '
                f'got {configured_total}.'
            )

    normalization_frames = kwargs.get('normalization_frames')

    if normalization_frames is not None:
        if int(normalization_frames) != 20:
            raise ValueError(
                'normalization_frames must remain 20 '
                'for every prediction horizon.'
            )

    if in_shape is not None:
        expected = [10, 1, 64, 64]

        if list(in_shape) != expected:
            raise ValueError(
                f'Expected in_shape {expected}, got {in_shape}.'
            )

    root = Path(data_root).expanduser()

    if data_file is None:
        path = root / 'Navier-StokesT50' / 'NavierStockT50.mat'
    else:
        path = Path(data_file).expanduser()

        if not path.is_absolute():
            path = root / path

    data = read_ns_data(path, axis_order=axis_order)
    data, normalization = normalize_ns_data(data, total_length=20)

    common = {
        'data': data,
        'normalization': normalization,
        't_in': pre_seq_length,
        't_out': aft_seq_length,
    }

    train_set = NavierStokesDataset(mode='train', **common)
    val_set = NavierStokesDataset(mode='val', **common)
    test_set = NavierStokesDataset(mode='test', **common)

    datasets = [train_set, val_set, test_set]

    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            overlap = np.intersect1d(
                datasets[i].trajectory_ids,
                datasets[j].trajectory_ids,
            )

            if overlap.size:
                raise RuntimeError(
                    f'The {datasets[i].mode} and '
                    f'{datasets[j].mode} splits overlap.'
                )

    loader_args = {
        'num_workers': num_workers,
        'pin_memory': True,
    }

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        **loader_args,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_args,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_args,
    )

    used_count = sum(len(dataset) for dataset in datasets)

    print(
        f'[NS-T50 split] train={len(train_set)}, '
        f'val={len(val_set)}, test={len(test_set)}, '
        f'unused_trajectories={len(data) - used_count}',
        flush=True,
    )

    return train_loader, val_loader, test_loader