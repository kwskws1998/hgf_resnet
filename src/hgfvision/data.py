"""Leakage-free CIFAR splits and deterministic, supervised online streams.

The official CIFAR test set is separate from every split returned by
``make_splits``. Stream metadata describes the experiment for reporting; callers
must pass only ``x`` and ``y`` to their updater. Image transformations apply
contrast, brightness, clipping, Gaussian noise, clipping, then normalization.
Noise is indexed by the source sample and seed, independent of access order.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
SPLIT_NAMES = ("source", "source_val", "stream", "dev")


def _integer(value: int, name: str, minimum: int = 0) -> int:
    """Validate integer arguments without silently accepting booleans or floats."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def make_splits(
    n: int = 50_000,
    seed: int = 0,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[int]]:
    """Partition training indices once, with no overlap and no discarded samples.

    Defaults are 30,000 / 5,000 / 10,000 / 5,000 for CIFAR's training set.
    Other dataset sizes use the same proportions, with rounding assigned to dev.
    Explicit sizes must specify exactly the four split names and sum to ``n``.
    Neither labels nor test data influence this partition.
    """
    n = _integer(n, "n")
    seed = _integer(seed, "seed")
    if sizes is None:
        counts = [3 * n // 5, n // 10, n // 5]
        counts.append(n - sum(counts))
    else:
        if set(sizes) != set(SPLIT_NAMES):
            raise ValueError(f"sizes must contain exactly {SPLIT_NAMES}")
        counts = [_integer(sizes[name], f"sizes[{name!r}]") for name in SPLIT_NAMES]
        if sum(counts) != n:
            raise ValueError("split sizes must sum to n")
    order = np.random.default_rng(seed).permutation(n).tolist()
    result = {}
    offset = 0
    for name, count in zip(SPLIT_NAMES, counts):
        result[name] = order[offset : offset + count]
        offset += count
    return result


@dataclass(frozen=True)
class Phase:
    """A fixed regime lasting ``steps`` minibatches, with pixel-scale noise.

    ``noise_std`` is measured on RGB intensities in [0, 1], before CIFAR
    normalization. Repeat brightness/contrast settings in adjacent phases to
    retain a domain shift while adding and removing a transient noise burst.
    """

    name: str
    steps: int
    brightness: float = 1.0
    contrast: float = 1.0
    noise_std: float = 0.0

    def __post_init__(self) -> None:
        """Reject ambiguous regimes and invalid transformation magnitudes."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("phase name must be a nonempty string")
        _integer(self.steps, "steps", minimum=1)
        for name in ("brightness", "contrast", "noise_std"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite nonnegative number")
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")


def _indices(base_dataset: Dataset, indices: Sequence[int] | None) -> tuple[int, ...]:
    """Validate a unique, in-range selection before a stream can start."""
    chosen = range(len(base_dataset)) if indices is None else indices
    result = tuple(_integer(index, "source index") for index in chosen)
    if len(result) != len(set(result)):
        raise ValueError("indices must be unique; source samples cannot repeat")
    if any(index >= len(base_dataset) for index in result):
        raise ValueError("source index is outside the base dataset")
    return result


def _position(index: int, length: int) -> int:
    """Implement ordinary dataset indexing, including valid negative indices."""
    if isinstance(index, bool) or not isinstance(index, Integral):
        raise TypeError("dataset index must be an integer")
    position = int(index)
    if position < 0:
        position += length
    if not 0 <= position < length:
        raise IndexError("dataset index out of range")
    return position


def _raw_tensor(image: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
    """Copy one unnormalized 32-pixel RGB image without modifying its source."""
    if isinstance(image, Image.Image):
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        result = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
    elif isinstance(image, np.ndarray):
        if image.shape != (32, 32, 3) or image.dtype != np.uint8:
            raise ValueError("numpy images must be uint8 with shape (32, 32, 3)")
        result = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
    elif isinstance(image, torch.Tensor):
        if tuple(image.shape) != (3, 32, 32):
            raise ValueError("tensor images must have shape (3, 32, 32)")
        if image.device.type != "cpu":
            raise ValueError("dataset images must be on the CPU")
        if image.dtype == torch.uint8:
            result = image.detach().clone().float().div_(255.0)
        elif image.is_floating_point():
            result = image.detach().clone().float()
            if not torch.isfinite(result).all() or result.min() < 0 or result.max() > 1:
                raise ValueError("floating tensor images must contain values in [0, 1]")
        else:
            raise ValueError("tensor images must be uint8 or floating point")
    else:
        raise TypeError("base images must be PIL images, uint8 arrays, or raw tensors")
    if tuple(result.shape) != (3, 32, 32):
        raise ValueError("images must be 32x32 RGB")
    return result.contiguous()


def _normalize(image: torch.Tensor) -> torch.Tensor:
    """Apply fixed CIFAR statistics; no statistics are fit on stream or test data."""
    mean = image.new_tensor(CIFAR_MEAN).view(3, 1, 1)
    std = image.new_tensor(CIFAR_STD).view(3, 1, 1)
    return (image - mean) / std


class CleanDataset(Dataset):
    """A raw-data subset with deterministic normalization and no augmentation."""

    def __init__(self, base_dataset: Dataset, indices: Sequence[int] | None = None):
        """Retain an immutable selection while leaving the original dataset intact."""
        self.base_dataset = base_dataset
        self.indices = _indices(base_dataset, indices)

    def __len__(self) -> int:
        """Return the number of selected samples."""
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return a normalized image and its supervised class label."""
        source_index = self.indices[_position(index, len(self))]
        image, label = self.base_dataset[source_index]
        return _normalize(_raw_tensor(image)), int(label)


class SourceDataset(CleanDataset):
    """Source-only training images with indexed, epoch-dependent augmentation.

    A crop of size 32 follows four-pixel zero padding, then an independent
    horizontal flip with probability 0.5. Call ``set_epoch`` before creating
    each epoch's loader iterator. Persistent loader workers must be recreated
    to receive a changed epoch, because each worker owns a dataset copy.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        indices: Sequence[int] | None = None,
        seed: int = 0,
        augment: bool = True,
    ):
        """Configure source preprocessing without touching any global RNG state."""
        super().__init__(base_dataset, indices)
        self.seed = _integer(seed, "seed")
        if not isinstance(augment, bool):
            raise ValueError("augment must be True or False")
        self.augment = augment
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Choose the reproducible augmentation realization for the next epoch."""
        self.epoch = _integer(epoch, "epoch")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Crop and flip using only the seed, epoch, and original sample index."""
        source_index = self.indices[_position(index, len(self))]
        raw_image, label = self.base_dataset[source_index]
        image = _raw_tensor(raw_image)
        if self.augment:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, source_index])
            )
            top, left = rng.integers(0, 9, size=2).tolist()
            padded = torch.nn.functional.pad(image, (4, 4, 4, 4))
            image = padded[:, top : top + 32, left : left + 32]
            if rng.random() < 0.5:
                image = image.flip(-1)
        return _normalize(image).contiguous(), int(label)


class StreamDataset(Dataset):
    """A finite, single-pass stream whose regime stays fixed within each step.

    Use a DataLoader with ``shuffle=False`` and this object's ``batch_size`` to
    preserve the intended schedule. Exactly the required prefix of ``indices``
    is consumed. Phase and index metadata are for reports, never updater inputs.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        indices: Sequence[int],
        phases: Sequence[Phase],
        batch_size: int = 8,
        seed: int = 0,
    ):
        """Validate the complete one-pass schedule without reading any images."""
        self.base_dataset = base_dataset
        self.batch_size = _integer(batch_size, "batch_size", minimum=1)
        self.seed = _integer(seed, "seed")
        self.phases = tuple(phases)
        if not self.phases or not all(isinstance(phase, Phase) for phase in self.phases):
            raise ValueError("phases must be a nonempty sequence of Phase objects")
        self._step_ends = []
        total_steps = 0
        for phase in self.phases:
            total_steps += phase.steps
            self._step_ends.append(total_steps)
        required = total_steps * self.batch_size
        chosen = _indices(base_dataset, indices)
        if len(chosen) < required:
            raise ValueError(f"stream requires {required} unique samples; got {len(chosen)}")
        self.indices = chosen[:required]

    def __len__(self) -> int:
        """Return exactly schedule steps times batch size, with no partial step."""
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        """Generate an indexed corruption independently of loader worker state."""
        position = _position(index, len(self))
        source_index = self.indices[position]
        step = position // self.batch_size
        phase = self.phases[bisect_right(self._step_ends, step)]
        raw_image, label = self.base_dataset[source_index]
        image = _raw_tensor(raw_image)
        channel_mean = image.mean(dim=(1, 2), keepdim=True)
        image = ((image - channel_mean) * phase.contrast + channel_mean) * phase.brightness
        image = image.clamp(0.0, 1.0)
        if phase.noise_std:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, source_index]))
            noise = torch.from_numpy(rng.standard_normal(tuple(image.shape)).astype(np.float32))
            image = (image + noise * phase.noise_std).clamp(0.0, 1.0)
        return {
            "x": _normalize(image),
            "y": int(label),
            "index": source_index,
            "phase": phase.name,
            "step": step,
        }


def load_cifar10(root: str | Path, download: bool = False) -> tuple[Dataset, Dataset]:
    """Load raw official train/test data; network downloads require explicit True."""
    if not isinstance(download, bool):
        raise ValueError("download must be explicitly True or False")
    from torchvision.datasets import CIFAR10

    train = CIFAR10(root=str(root), train=True, transform=None, download=download)
    test = CIFAR10(root=str(root), train=False, transform=None, download=download)
    return train, test


class FakeCIFAR10(Dataset):
    """Deterministic synthetic 10-class images for offline smoke tests only.

    These patterned images are not CIFAR-10, and results on this dataset provide
    no evidence of real-data accuracy, adaptation quality, or novelty.
    """

    def __init__(self, size: int = 100, seed: int = 0):
        """Configure synthetic sample count and a local deterministic noise seed."""
        self.size = _integer(size, "size")
        self.seed = _integer(seed, "seed")

    def __len__(self) -> int:
        """Return the synthetic dataset size."""
        return self.size

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        """Draw a label-specific colored stripe with deterministic pixel nuisance."""
        position = _position(index, len(self))
        label = position % 10
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, position]))
        image = rng.integers(8, 33, size=(32, 32, 3), dtype=np.uint8)
        column = 1 + 3 * label
        image[3:29, column : column + 2, :] = 150
        image[3:29, column : column + 2, label % 3] = 240
        return Image.fromarray(image), label
