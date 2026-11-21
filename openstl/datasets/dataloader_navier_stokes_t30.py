from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NavierStokesDataset(Dataset):
    def __init__(
        self,
        data,
        mode='train',
        t_in=10,
        t_out=10,
        normalization=None,
    ):
        super().__init__()

        if mode == 'train':
            self.data = data[:8000]
        elif mode == 'test':
            self.data = data[8000:10000]
        else:
            raise ValueError(f'Unsupported mode: {mode}')

        if normalization is None:
            raise ValueError('Normalization parameters are required.')

        self.t_in = t_in
        self.t_out = t_out
        self.mean = 0.0
        self.std = 1.0
        self.normalization = normalization
        self.min_val = normalization['min']
        self.max_val = normalization['max']

        print(
            f'{mode} dataset loaded | '
            f'Samples: {len(self.data)} | '
            f'Task: {t_in}->{t_out}'
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.data[idx]).unsqueeze(1)
        input_seq = seq[:self.t_in]
        target_seq = seq[self.t_in:self.t_in + self.t_out]
        return input_seq, target_seq


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

    if axis_order not in (None, 'THWN'):
        raise ValueError(
            'The raw T30 data must use THWN axis order. '
            'Set ns_axis_order to None or THWN.'
        )

    if (pre_seq_length, aft_seq_length) != (10, 10):
        raise ValueError(
            'This T30 configuration requires 10 input frames '
            'and 10 prediction frames.'
        )

    if in_shape is not None:
        if list(in_shape) != [10, 1, 64, 64]:
            raise ValueError(
                f'Expected in_shape [10,1,64,64], got {in_shape}.'
            )

    if data_file is None:
        raise ValueError('Please set ns_data_file in the configuration.')

    path = Path(data_file).expanduser()

    if not path.is_absolute():
        path = Path(data_root).expanduser() / path

    if not path.is_file():
        raise FileNotFoundError(f'Data file not found: {path}')

    with h5py.File(path, 'r') as f:
        if 'u' not in f:
            raise KeyError(f'Dataset "u" was not found in: {path}')

        data = np.array(f['u'], dtype=np.float32)

    print(f'Raw u.shape: {data.shape}')

    if data.shape != (50, 64, 64, 10000):
        raise ValueError(
            'Expected raw T30 data with shape [50,64,64,10000], '
            f'got {data.shape}.'
        )

    data = data.transpose(3, 0, 1, 2)

    print(f'Transposed shape: {data.shape}')

    train_data = data[:8000]

    min_val = train_data.min()
    max_val = train_data.max()
    scale = max_val - min_val + 1e-8

    if not np.isfinite(min_val) or not np.isfinite(max_val):
        raise ValueError(
            'The training data contain NaN or infinite values.'
        )

    data = (data - min_val) / scale

    normalization = {
        'min': float(min_val),
        'max': float(max_val),
        'scale': float(scale),
        'fit_scope': 'training_trajectories',
    }

    train_set = NavierStokesDataset(
        data,
        mode='train',
        t_in=pre_seq_length,
        t_out=aft_seq_length,
        normalization=normalization,
    )

    test_set = NavierStokesDataset(
        data,
        mode='test',
        t_in=pre_seq_length,
        t_out=aft_seq_length,
        normalization=normalization,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader, test_loader