import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


class TaxibjDataset(Dataset):
    def __init__(self, X, Y, use_augment=False):
        super().__init__()

        self.X = (np.asarray(X, dtype=np.float32) + 1.0) / 2.0
        self.Y = (np.asarray(Y, dtype=np.float32) + 1.0) / 2.0
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1

    def _augment_seq(self, sequences):
        if random.randint(0, 1):
            sequences = torch.flip(sequences, dims=(3,))
        return sequences

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.from_numpy(self.X[index].copy())
        labels = torch.from_numpy(self.Y[index].copy())

        if self.use_augment:
            input_length = data.shape[0]
            sequences = self._augment_seq(
                torch.cat([data, labels], dim=0)
            )
            data = sequences[:input_length]
            labels = sequences[input_length:]

        return data, labels


def load_data(
    batch_size,
    val_batch_size,
    data_root,
    num_workers=4,
    pre_seq_length=4,
    aft_seq_length=4,
    in_shape=None,
    distributed=False,
    use_augment=False,
    use_prefetcher=False,
    drop_last=False,
):
    if pre_seq_length is None:
        pre_seq_length = 4

    if aft_seq_length is None:
        aft_seq_length = 4

    if pre_seq_length != 4 or aft_seq_length != 4:
        raise ValueError(
            "The TaxiBJ setting requires 4 input frames "
            "and 4 target frames."
        )

    dataset_path = os.path.join(
        data_root, "taxibj", "dataset.npz"
    )

    with np.load(dataset_path, allow_pickle=False) as dataset:
        X_train = dataset["X_train"]
        Y_train = dataset["Y_train"]
        X_test = dataset["X_test"]
        Y_test = dataset["Y_test"]

    expected_shapes = {
        "X_train": (20461, 4, 2, 32, 32),
        "Y_train": (20461, 4, 2, 32, 32),
        "X_test": (500, 4, 2, 32, 32),
        "Y_test": (500, 4, 2, 32, 32),
    }

    arrays = {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test,
    }

    for name, array in arrays.items():
        if array.shape != expected_shapes[name]:
            raise ValueError(
                f"{name} has shape {array.shape}; "
                f"expected {expected_shapes[name]}."
            )

    train_set = TaxibjDataset(
        X_train,
        Y_train,
        use_augment=use_augment,
    )

    test_set = TaxibjDataset(
        X_test,
        Y_test,
        use_augment=False,
    )

    dataloader_train = create_loader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        is_training=True,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
    )

    dataloader_vali = create_loader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
    )

    dataloader_test = create_loader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
    )

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == "__main__":
    train_loader, _, test_loader = load_data(
        batch_size=1,
        val_batch_size=1,
        data_root="data/",
        num_workers=4,
        pre_seq_length=4,
        aft_seq_length=4,
        use_augment=False,
        drop_last=False,
    )

    print(
        f"Training samples: {len(train_loader.dataset)}"
    )
    print(
        f"Test samples: {len(test_loader.dataset)}"
    )

    for data, labels in train_loader:
        print(
            f"Training batch: {data.shape}, {labels.shape}"
        )
        break

    for data, labels in test_loader:
        print(
            f"Test batch: {data.shape}, {labels.shape}"
        )
        break