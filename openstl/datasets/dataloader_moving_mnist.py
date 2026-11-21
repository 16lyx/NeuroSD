import gzip
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


def load_mnist(root, data_name='mnist'):
    if data_name != 'mnist':
        raise ValueError(
            'This loader supports the standard MovingMNIST dataset only.'
        )

    path = os.path.join(
        root,
        'moving_mnist',
        'train-images-idx3-ubyte.gz',
    )

    if not os.path.isfile(path):
        raise FileNotFoundError(f'MNIST image file not found: {path}')

    with gzip.open(path, 'rb') as f:
        content = f.read()

    images = np.frombuffer(content, dtype=np.uint8, offset=16)

    if images.size == 0 or images.size % (28 * 28) != 0:
        raise ValueError(f'Invalid MNIST image data: {path}')

    return images.reshape(-1, 28, 28)


def load_fixed_set(root, data_name='mnist'):
    if data_name != 'mnist':
        raise ValueError(
            'This loader supports the standard MovingMNIST dataset only.'
        )

    path = os.path.join(
        root,
        'moving_mnist',
        'mnist_test_seq.npy',
    )

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'Fixed MovingMNIST test file not found: {path}'
        )

    dataset = np.load(path, allow_pickle=False)

    if dataset.shape != (20, 10000, 64, 64):
        raise ValueError(
            'Expected fixed test data with shape [20,10000,64,64], '
            f'got {dataset.shape}.'
        )

    if dataset.dtype != np.uint8:
        raise ValueError(
            'Expected uint8 test images with pixel values in [0,255], '
            f'got {dataset.dtype}.'
        )

    return dataset[..., np.newaxis]


class MovingMNIST(Dataset):
    def __init__(
        self,
        root,
        is_train=True,
        data_name='mnist',
        n_frames_input=10,
        n_frames_output=10,
        image_size=64,
        num_objects=None,
        transform=None,
        use_augment=False,
    ):
        super().__init__()

        if num_objects is None:
            num_objects = [2]

        if data_name != 'mnist':
            raise ValueError(
                'This configuration requires data_name="mnist".'
            )

        if (n_frames_input, n_frames_output) != (10, 10):
            raise ValueError(
                'This configuration requires 10 input frames '
                'and 10 prediction frames.'
            )

        if image_size != 64:
            raise ValueError('This configuration requires 64x64 images.')

        if list(num_objects) != [2]:
            raise ValueError(
                'This configuration requires exactly two moving digits.'
            )

        if use_augment:
            raise ValueError(
                'Additional augmentation is disabled in this configuration.'
            )

        if transform is not None:
            raise ValueError(
                'Additional transforms are disabled in this configuration.'
            )

        self.is_train = is_train
        self.data_name = data_name
        self.num_objects = list(num_objects)
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = n_frames_input + n_frames_output
        self.image_size_ = image_size
        self.digit_size_ = 28
        self.step_length_ = 0.1
        self.mean = 0.0
        self.std = 1.0
        self.dataset = None
        self.mnist = None

        if self.is_train:
            self.mnist = load_mnist(root, data_name)
            self.length = 10000
        else:
            self.dataset = load_fixed_set(root, data_name)
            self.length = self.dataset.shape[1]

    def get_random_trajectory(self, seq_length):
        canvas_size = self.image_size_ - self.digit_size_

        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi

        velocity_x = np.cos(theta)
        velocity_y = np.sin(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)

        bounce_x = 1
        bounce_y = 1

        for i in range(seq_length):
            y += bounce_y * velocity_y * self.step_length_
            x += bounce_x * velocity_x * self.step_length_

            if x <= 0:
                x = 0
                bounce_x = -bounce_x

            if x >= 1.0:
                x = 1.0
                bounce_x = -bounce_x

            if y <= 0:
                y = 0
                bounce_y = -bounce_y

            if y >= 1.0:
                y = 1.0
                bounce_y = -bounce_y

            start_y[i] = y
            start_x[i] = x

        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)

        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        data = np.zeros(
            (
                self.n_frames_total,
                self.image_size_,
                self.image_size_,
            ),
            dtype=np.float32,
        )

        for _ in range(num_digits):
            start_y, start_x = self.get_random_trajectory(
                self.n_frames_total
            )

            digit_index = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[digit_index]

            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_

                data[i, top:bottom, left:right] = np.maximum(
                    data[i, top:bottom, left:right],
                    digit_image,
                )

        return data[..., np.newaxis]

    def __getitem__(self, idx):
        if self.is_train:
            num_digits = random.choice(self.num_objects)
            images = self.generate_moving_mnist(num_digits)
        else:
            images = self.dataset[:, idx, ...]

        images = images.transpose(0, 3, 1, 2)
        images = np.ascontiguousarray(images, dtype=np.float32)
        sequence = torch.from_numpy(images).div_(255.0)

        input_seq = sequence[:self.n_frames_input].contiguous()
        target_seq = sequence[
            self.n_frames_input:self.n_frames_total
        ].contiguous()

        return input_seq, target_seq

    def __len__(self):
        return self.length


def load_data(
    batch_size,
    val_batch_size,
    data_root,
    num_workers=4,
    data_name='mnist',
    pre_seq_length=10,
    aft_seq_length=10,
    in_shape=None,
    distributed=False,
    use_augment=False,
    use_prefetcher=False,
    drop_last=False,
    **kwargs,
):
    if in_shape is not None:
        if list(in_shape) != [10, 1, 64, 64]:
            raise ValueError(
                f'Expected in_shape [10,1,64,64], got {in_shape}.'
            )

    train_set = MovingMNIST(
        root=data_root,
        is_train=True,
        data_name=data_name,
        n_frames_input=pre_seq_length,
        n_frames_output=aft_seq_length,
        image_size=64,
        num_objects=[2],
        use_augment=use_augment,
    )

    test_set = MovingMNIST(
        root=data_root,
        is_train=False,
        data_name=data_name,
        n_frames_input=pre_seq_length,
        n_frames_output=aft_seq_length,
        image_size=64,
        num_objects=[2],
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
        drop_last=False,
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
        drop_last=False,
        num_workers=num_workers,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
    )

    return dataloader_train, dataloader_vali, dataloader_test