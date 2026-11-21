import random
from functools import partial
from itertools import repeat

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    IterableDataset,
    default_collate,
    get_worker_info,
)
from torch.utils.data.distributed import DistributedSampler

from timm.data.distributed_sampler import (
    OrderedDistributedSampler,
    RepeatAugSampler,
)


def worker_init(worker_id, worker_seeding="all"):
    worker_info = get_worker_info()

    if worker_info is None:
        return

    if worker_info.id != worker_id:
        raise RuntimeError("Unexpected worker ID.")

    if callable(worker_seeding):
        seed = int(worker_seeding(worker_info))
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
    elif worker_seeding == "all":
        np.random.seed(worker_info.seed % (2**32))
    elif worker_seeding != "part":
        raise ValueError(
            "worker_seeding must be 'all', 'part', or a callable."
        )


def fast_collate_for_prediction(batch):
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    if not isinstance(batch[0], (tuple, list)):
        raise TypeError("Each sample must contain inputs and targets.")

    if isinstance(batch[0][0], tuple):
        inner_size = len(batch[0][0])

        if inner_size == 0:
            raise ValueError("Input tuples cannot be empty.")

        if any(len(sample[0]) != inner_size for sample in batch):
            raise ValueError("Input tuples must have equal lengths.")

        inputs = []
        targets = []

        for j in range(inner_size):
            for sample in batch:
                inputs.append(
                    torch.as_tensor(
                        sample[0][j],
                        dtype=torch.float32,
                    )
                )
                targets.append(
                    torch.as_tensor(
                        sample[1],
                        dtype=torch.float32,
                    )
                )

        return torch.stack(inputs), torch.stack(targets)

    inputs = torch.stack(
        [
            torch.as_tensor(sample[0], dtype=torch.float32)
            for sample in batch
        ]
    )

    targets = torch.stack(
        [
            torch.as_tensor(sample[1], dtype=torch.float32)
            for sample in batch
        ]
    )

    return inputs, targets


def expand_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        return tuple(repeat(x, n))

    if len(x) == 1:
        return tuple(x) * n

    if len(x) != n:
        raise ValueError(
            "Normalization statistics must match the channel count."
        )

    return tuple(x)


class PrefetchLoader:
    def __init__(
        self,
        loader,
        mean=None,
        std=None,
        channels=3,
        fp16=False,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("PrefetchLoader requires CUDA.")

        if (mean is None) != (std is None):
            raise ValueError(
                "mean and std must be provided together."
            )

        self.loader = loader
        self.fp16 = fp16
        self.device = torch.device(
            "cuda",
            torch.cuda.current_device(),
        )
        self.dtype = torch.float16 if fp16 else torch.float32
        self.mean = None
        self.std = None

        if mean is not None:
            self.mean = torch.tensor(
                expand_to_chs(mean, channels),
                dtype=self.dtype,
                device=self.device,
            ) * 255.0

            self.std = torch.tensor(
                expand_to_chs(std, channels),
                dtype=self.dtype,
                device=self.device,
            ) * 255.0

            if torch.any(self.std == 0).item():
                raise ValueError(
                    "Normalization standard deviations must be nonzero."
                )

    def _prepare(self, tensor):
        tensor = tensor.to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=True,
        )

        if self.mean is not None:
            if tensor.ndim not in (4, 5):
                raise ValueError(
                    "Normalization requires [B,C,H,W] "
                    "or [B,T,C,H,W] tensors."
                )

            if tensor.shape[-3] != self.mean.numel():
                raise ValueError(
                    "Tensor channels do not match normalization statistics."
                )

            shape = (
                (1,) * (tensor.ndim - 3)
                + (self.mean.numel(), 1, 1)
            )

            tensor = tensor.sub(
                self.mean.view(shape)
            ).div(
                self.std.view(shape)
            )

        return tensor

    def __iter__(self):
        stream = torch.cuda.Stream(device=self.device)
        current_input = None
        current_target = None

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = self._prepare(next_input)
                next_target = self._prepare(next_target)

            if current_input is not None:
                yield current_input, current_target

            current_stream = torch.cuda.current_stream(
                device=self.device
            )
            current_stream.wait_stream(stream)

            current_input = next_input
            current_target = next_target

            current_input.record_stream(current_stream)
            current_target.record_stream(current_stream)

        if current_input is not None:
            yield current_input, current_target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def create_loader(
    dataset,
    batch_size,
    shuffle=True,
    is_training=False,
    mean=None,
    std=None,
    num_workers=1,
    num_aug_repeats=0,
    input_channels=1,
    use_prefetcher=False,
    distributed=False,
    pin_memory=False,
    drop_last=False,
    fp16=False,
    collate_fn=None,
    persistent_workers=True,
    worker_seeding="all",
):
    sampler = None
    is_iterable = isinstance(dataset, IterableDataset)

    if distributed and not is_iterable:
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(
                    dataset,
                    num_repeats=num_aug_repeats,
                )
            else:
                sampler = DistributedSampler(
                    dataset,
                    shuffle=shuffle,
                )
        else:
            sampler = OrderedDistributedSampler(dataset)
    elif num_aug_repeats:
        raise ValueError(
            "Repeated augmentation requires a distributed "
            "map-style dataset."
        )

    if collate_fn is None:
        collate_fn = default_collate

    loader_args = {
        "batch_size": batch_size,
        "shuffle": (
            shuffle
            and is_training
            and not is_iterable
            and sampler is None
        ),
        "num_workers": num_workers,
        "sampler": sampler,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "worker_init_fn": partial(
            worker_init,
            worker_seeding=worker_seeding,
        ),
        "persistent_workers": (
            persistent_workers and num_workers > 0
        ),
    }

    try:
        loader = DataLoader(dataset, **loader_args)
    except TypeError as error:
        if "persistent_workers" not in str(error):
            raise

        loader_args.pop("persistent_workers")
        loader = DataLoader(dataset, **loader_args)

    if use_prefetcher:
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            channels=input_channels,
            fp16=fp16,
        )

    return loader


def reshape_patch(img_tensor, patch_size):
    if img_tensor.ndim != 4:
        raise ValueError("Expected an array with shape [T,H,W,C].")

    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")

    seq_length, height, width, channels = img_tensor.shape

    if height % patch_size or width % patch_size:
        raise ValueError(
            "Image dimensions must be divisible by patch_size."
        )

    patches = np.reshape(
        img_tensor,
        (
            seq_length,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
            channels,
        ),
    )

    patches = np.transpose(
        patches,
        (0, 1, 3, 2, 4, 5),
    )

    return np.reshape(
        patches,
        (
            seq_length,
            height // patch_size,
            width // patch_size,
            patch_size * patch_size * channels,
        ),
    )


def reshape_patch_back(patch_tensor, patch_size):
    if patch_tensor.ndim != 5:
        raise ValueError("Expected an array with shape [B,T,H,W,C].")

    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")

    batch_size, seq_length, height, width, channels = (
        patch_tensor.shape
    )
    patch_area = patch_size * patch_size

    if channels % patch_area:
        raise ValueError(
            "Patch channels must be divisible by patch_size squared."
        )

    image_channels = channels // patch_area

    images = np.reshape(
        patch_tensor,
        (
            batch_size,
            seq_length,
            height,
            width,
            patch_size,
            patch_size,
            image_channels,
        ),
    )

    images = np.transpose(
        images,
        (0, 1, 2, 4, 3, 5, 6),
    )

    return np.reshape(
        images,
        (
            batch_size,
            seq_length,
            height * patch_size,
            width * patch_size,
            image_channels,
        ),
    )


def reshape_patch_back_tensor(patch_tensor, patch_size):
    if patch_tensor.ndim != 5:
        raise ValueError("Expected a tensor with shape [B,T,H,W,C].")

    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")

    batch_size, seq_length, height, width, channels = (
        patch_tensor.shape
    )
    patch_area = patch_size * patch_size

    if channels % patch_area:
        raise ValueError(
            "Patch channels must be divisible by patch_size squared."
        )

    image_channels = channels // patch_area

    images = patch_tensor.reshape(
        batch_size,
        seq_length,
        height,
        width,
        patch_size,
        patch_size,
        image_channels,
    )

    images = images.permute(
        0, 1, 2, 4, 3, 5, 6
    ).reshape(
        batch_size,
        seq_length,
        height * patch_size,
        width * patch_size,
        image_channels,
    )

    return images.permute(0, 1, 4, 2, 3)