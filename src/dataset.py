"""Dataset loading and preprocessing for MNIST, CIFAR-10, and a synthetic circle dataset."""

import math
from typing import Literal

import torch
import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class CircleDataset(Dataset):
    """Synthetic 28×28 grayscale images of a circle with random radius / position.

    Each sample is a white ring on a black background, normalised to [-1, 1].
    Useful for smoke-testing the diffusion model.
    """

    def __init__(
        self,
        size: int = 28,
        num_samples: int = 60000,
        min_radius: float = 0.15,
        max_radius: float = 0.40,
        thickness: float = 0.04,
        seed: int = 42,
    ):
        self.size = size
        self.num_samples = num_samples
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.thickness = thickness
        self.rng = torch.Generator().manual_seed(seed)

        # Pre-compute a coordinate grid once
        ys = torch.linspace(-1, 1, size)
        xs = torch.linspace(-1, 1, size)
        self.Y, self.X = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Random circle parameters
        r = self.min_radius + (self.max_radius - self.min_radius) * torch.rand(1, generator=self.rng).item()
        dx = 0.15 * torch.randn(1, generator=self.rng).item()
        dy = 0.15 * torch.randn(1, generator=self.rng).item()

        # Distance from centre (dx, dy)
        D = torch.sqrt((self.X - dx) ** 2 + (self.Y - dy) ** 2)

        # Ring: 1 inside the ring, 0 elsewhere
        inner = r - self.thickness / 2
        outer = r + self.thickness / 2
        img = ((D >= inner) & (D <= outer)).float()

        # Normalise to [-1, 1]
        img = img * 2.0 - 1.0
        return img.unsqueeze(0)  # (1, H, W)


class HFDiffusionDataset(Dataset):
    """Wrap a HuggingFace dataset and apply torchvision transforms per-sample."""

    def __init__(
        self,
        name: Literal["mnist", "cifar10"],
        image_size: int = 32,
        split: str = "train",
        cache_dir: str = "./datasets",
    ):
        self.name = name

        if name == "mnist":
            self.hf_dataset = load_dataset("ylecun/mnist", split=split, cache_dir=cache_dir)
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),  # [0,1] -> [-1,1] grayscale
            ])
            self._key = "image"
        elif name == "cifar10":
            self.hf_dataset = load_dataset("uoft-cs/cifar10", split=split, cache_dir=cache_dir)
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [0,1] -> [-1,1] RGB
            ])
            self._key = "img"
        else:
            raise ValueError(f"Unknown dataset: {name}")

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        example = self.hf_dataset[idx]
        img = example[self._key]
        return self.transform(img)


def create_dataloader(
    name: Literal["mnist", "cifar10", "circle"],
    image_size: int = 32,
    batch_size: int = 128,
    split: str = "train",
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    cache_dir: str = "./datasets",
) -> DataLoader:
    """Create a DataLoader for the specified dataset.

    Args:
        name: Dataset name ("mnist" or "cifar10").
        image_size: Target image size (square).
        batch_size: Number of samples per batch.
        split: "train" or "test".
        shuffle: Whether to shuffle examples.
        num_workers: Number of dataloader subprocesses (4 is good for T4 GPU).
        pin_memory: Pin CPU memory for faster CUDA transfers.
        cache_dir: HuggingFace dataset cache directory.

    Returns:
        DataLoader yielding (B, C, H, W) image tensors in [-1, 1].
    """
    if name == "circle":
        num = 60000 if split == "train" else 10000
        dataset = CircleDataset(size=image_size, num_samples=num)
    else:
        dataset = HFDiffusionDataset(name, image_size, split, cache_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
