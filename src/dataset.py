"""Dataset loading and preprocessing for MNIST and CIFAR-10 via HuggingFace `datasets`."""

from typing import Literal

import torch
import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


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
    name: Literal["mnist", "cifar10"],
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
    dataset = HFDiffusionDataset(name, image_size, split, cache_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
