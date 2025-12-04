import os
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from scripts.configs import TrainingConfig


def build_cifar10_datasets(
    root: str = "./data/",
    use_aug: bool = True
) -> Tuple[Dataset, Dataset]:
    """
    Build CIFAR-10 train/val datasets.

    - Train dataset: with data augmentation (if use_aug=True)
    - Val dataset:   no augmentation (clean images)

    Both are built from the CIFAR-10 training split; we then
    split indices into train/val subsets.
    """

    if use_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Cutout-like regularisation
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value=0.0,
                inplace=False
            )
        ])
    else:
        # No augmentation (for ablations)
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # We construct two copies of the CIFAR-10 training split:
    #  - one with train_transform for the train subset
    #  - one with val_transform for the val subset
    full_train_aug = torchvision.datasets.CIFAR10(
        root,
        train=True,
        download=True,
        transform=train_transform
    )

    full_train_clean = torchvision.datasets.CIFAR10(
        root,
        train=True,
        download=False,
        transform=val_transform
    )

    return full_train_aug, full_train_clean


def split_indices(
    n: int,
    valid_size: Union[int, float],
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Return (train_indices, valid_indices) given dataset size and valid_size.

    Args:
        n: Total number of examples in the dataset.
        valid_size: If int, number of validation examples.
                    If float, fraction of validation examples.
        shuffle: Whether to shuffle the indices before splitting.
        seed: Random seed for shuffling.

    Returns:
        train_indices: List of training set indices.
        valid_indices: List of validation set indices.

    Raises:
        ValueError: If valid_size is invalid.
    """
    if isinstance(valid_size, float):
        if not (0.0 < valid_size < 1.0):
            raise ValueError(
                f"valid_size as float must be in (0, 1); got {valid_size}"
            )
        valid_size = int(n * valid_size)

    if valid_size <= 0 or valid_size >= n:
        raise ValueError(
            f"valid_size={valid_size} must be > 0 and < dataset size {n}"
        )

    indices = list(range(n))

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        indices = torch.randperm(n, generator=generator).tolist()

    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]

    return train_indices, valid_indices


def make_dataloader(
    dataset: Dataset,
    indices: List[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool
) -> DataLoader:
    """Create a DataLoader from a subset of indices."""
    subset = Subset(dataset, indices)

    loader = DataLoader(
        dataset=subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

    return loader


def build_dataloaders(
    cfg: TrainingConfig,
    root: str = "./data/"
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders according to config,
    with data augmentation on the train set only.

    Returns:
        train_loader, valid_loader
    """
    use_aug = cfg.use_aug
    
    # Build base datasets (both from CIFAR-10 train split)
    ds_train_aug, ds_train_clean = build_cifar10_datasets(
        root=root,
        use_aug=use_aug
    )

    n = len(ds_train_aug)
    train_indices, valid_indices = split_indices(
        n=n,
        valid_size=cfg.valid_size,
        shuffle=True,
        seed=cfg.seed
    )
    train_loader = make_dataloader(
        dataset=ds_train_aug,
        indices=train_indices,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )
    valid_loader = make_dataloader(
        dataset=ds_train_clean,
        indices=valid_indices,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )

    return train_loader, valid_loader