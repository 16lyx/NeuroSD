from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


def read_ns_mat(file_path, axis_order=None):
    file_path = Path(file_path).expanduser()

    if not file_path.is_file():
        raise FileNotFoundError(f'Data file not found: {file_path}')

    if h5py.is_hdf5(str(file_path)):
        with h5py.File(file_path, 'r') as f:
            if 'u' not in f:
                raise KeyError(
                    f'Variable "u" was not found in: {file_path}'
                )
            raw = np.asarray(f['u'], dtype=np.float32)

        source = 'hdf5'
        order = axis_order or 'HWTN'
    else:
        contents = sio.loadmat(file_path)

        if 'u' not in contents:
            raise KeyError(
                f'Variable "u" was not found in: {file_path}'
            )

        raw = np.asarray(contents['u'], dtype=np.float32)
        source = 'scipy_mat'
        order = axis_order or 'NHWT'

    if raw.ndim != 4:
        raise ValueError(
            f'Variable "u" must be four-dimensional, got {raw.shape}.'
        )

    if not isinstance(order, str):
        raise ValueError('axis_order must be a string.')

    order = order.upper()

    if len(order) != 4 or set(order) != set('NTHW'):
        raise ValueError(
            'axis_order must contain N, T, H, and W exactly once.'
        )

    permutation = tuple(order.index(axis) for axis in 'NTHW')
    data = raw.transpose(permutation)

    print(
        f'[NS data] source={source}, '
        f'raw_shape={raw.shape}, axis_order={order}, '
        f'converted_shape={data.shape}',
        flush=True,
    )

    return data, order


class NavierStokesT20Dataset(Dataset):
    def __init__(
        self,
        data,
        trajectory_ids,
        split,
        pre_seq_length,
        aft_seq_length,
        normalization,
        file_path,
        axis_order,
    ):
        super().__init__()

        self.data = data
        self.trajectory_ids = list(trajectory_ids)
        self.split = split
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length

        self.normalization = dict(normalization)
        self.min_val = normalization['min']
        self.max_val = normalization['max']
        self.scale = normalization['scale']

        self.file_path = str(file_path)
        self.axis_order = axis_order
        self.use_augment = False
        self.mean = 0.0
        self.std = 1.0

        if len(self.trajectory_ids) != len(self.data):
            raise ValueError(
                'The number of trajectory IDs must match '
                'the number of samples.'
            )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        sequence = torch.from_numpy(
            self.data[index]
        ).unsqueeze(1)

        inputs = sequence[:self.pre_seq_length]
        targets = sequence[
            self.pre_seq_length:
            self.pre_seq_length + self.aft_seq_length
        ]

        return inputs, targets


def load_data(
    batch_size,
    val_batch_size,
    data_root,
    num_workers=4,
    pre_seq_length=10,
    aft_seq_length=10,
    in_shape=None,
    distributed=False,
    use_augment=False,
    use_prefetcher=False,
    drop_last=False,
    data_file=None,
    axis_order=None,
):
    if distributed:
        raise NotImplementedError(
            'This loader supports single-device training only.'
        )

    if use_augment:
        raise ValueError(
            'Please set use_augment=False for this configuration.'
        )

    if use_prefetcher:
        raise ValueError(
            'Please set use_prefetcher=False for this configuration.'
        )

    if (pre_seq_length, aft_seq_length) != (10, 10):
        raise ValueError(
            'The T20 configuration requires 10 input frames '
            'and 10 prediction frames.'
        )

    expected_in_shape = [10, 1, 64, 64]

    if in_shape is not None:
        if list(in_shape) != expected_in_shape:
            raise ValueError(
                f'Expected in_shape {expected_in_shape}, '
                f'got {in_shape}.'
            )

    root = Path(data_root).expanduser()

    if data_file is None:
        file_path = (
            root
            / 'navier_stokes_t20'
            / 'NavierStockT20.mat'
        )
    else:
        file_path = Path(data_file).expanduser()

        if not file_path.is_absolute():
            file_path = root / file_path

    file_path = file_path.resolve()

    data, actual_axis_order = read_ns_mat(
        file_path,
        axis_order=axis_order,
    )

    if data.shape != (1200, 20, 64, 64):
        raise ValueError(
            'Expected converted T20 data with shape [1200,20,64,64], '
            f'got {data.shape}. Check the file and axis_order.'
        )

    if not np.isfinite(data).all():
        raise ValueError('The data contain NaN or infinite values.')

    train_raw = data[:960]

    min_val = float(train_raw.min())
    max_val = float(train_raw.max())

    if max_val <= min_val:
        raise ValueError(
            'The training data must have a positive value range.'
        )

    scale = max_val - min_val + 1e-8

    normalization = {
        'method': 'train_minmax',
        'min': min_val,
        'max': max_val,
        'scale': scale,
        'fit_trajectory_range': [0, 960],
        'fit_time_range': [0, 20],
        'inverse': 'raw = normalized * scale + min',
        'clip': False,
    }

    normalized = np.ascontiguousarray(
        (data - min_val) / scale,
        dtype=np.float32,
    )

    common = {
        'pre_seq_length': pre_seq_length,
        'aft_seq_length': aft_seq_length,
        'normalization': normalization,
        'file_path': file_path,
        'axis_order': actual_axis_order,
    }

    train_set = NavierStokesT20Dataset(
        data=normalized[:960],
        trajectory_ids=range(960),
        split='train',
        **common,
    )

    test_set = NavierStokesT20Dataset(
        data=normalized[960:1200],
        trajectory_ids=range(960, 1200),
        split='test',
        **common,
    )

    print(
        f'[NS split] train={len(train_set)}, '
        f'test={len(test_set)}, '
        f'train_min={min_val}, train_max={max_val}',
        flush=True,
    )

    dataloader_train = create_loader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        is_training=True,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
        distributed=False,
        use_prefetcher=False,
    )

    dataloader_test = create_loader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
        distributed=False,
        use_prefetcher=False,
    )

    return dataloader_train, dataloader_test, dataloader_test