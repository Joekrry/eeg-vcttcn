"""PyTorch dataset, pooled cross-subject split, and DataLoader construction.

The split is performed at *trial* granularity (not per epoch/window) so that no
trial ever straddles the train/val/test boundary -- this prevents optimistic
leakage when a trial is later expanded into several overlapping windows. The
split is stratified by class and pooled across all subjects.
"""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from cvttcn.config import AugmentConfig, Config, DataConfig
from cvttcn.data.preprocessing import EpochedData


class Augmentation:
    """Light, shape-preserving augmentation for one epoch of shape ``(1, C, T)``.

    Applies (optionally) a circular time shift, per-channel dropout, and additive
    Gaussian noise. A dedicated ``numpy`` Generator makes it reproducible.
    """

    def __init__(self, cfg: AugmentConfig, rng: "np.random.Generator | None" = None):
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.cfg.enabled:
            return x
        x = x.copy()
        if self.cfg.time_shift > 0:
            shift = int(self.rng.integers(-self.cfg.time_shift, self.cfg.time_shift + 1))
            x = np.roll(x, shift, axis=-1)
        if self.cfg.channel_dropout > 0:
            keep = (self.rng.random(x.shape[1]) >= self.cfg.channel_dropout)
            x = x * keep[None, :, None]
        if self.cfg.noise_std > 0:
            x = x + self.rng.normal(0.0, self.cfg.noise_std, size=x.shape)
        return x.astype(np.float32)


class EEGDataset(Dataset):
    """Wraps epoch arrays as ``(tensor x, tensor y)`` pairs.

    ``X`` is ``(N, 1, C, T)`` float32 and ``y`` is ``(N,)`` int64. An optional
    ``augment`` callable is applied to ``x`` (training split only).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: "Augmentation | None" = None):
        assert len(X) == len(y), "X and y must have the same length."
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.augment is not None:
            x = self.augment(x)
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        label = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, label


@dataclass
class DataSplit:
    """Epoch indices for each split (indices into the original arrays)."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def split_by_trial(data: EpochedData, cfg: DataConfig) -> DataSplit:
    """Stratified, pooled train/val/test split performed at trial granularity."""
    trials = data.trials
    uniq, first_idx = np.unique(trials, return_index=True)
    trial_labels = data.y[first_idx]  # label of each unique trial (aligned with uniq)

    positions = np.arange(len(uniq))
    tv_pos, test_pos = train_test_split(
        positions,
        test_size=cfg.test_size,
        random_state=cfg.split_seed,
        stratify=trial_labels,
    )
    # val size expressed relative to the remaining (train+val) trials
    rel_val = cfg.val_size / (1.0 - cfg.test_size)
    train_pos, val_pos = train_test_split(
        tv_pos,
        test_size=rel_val,
        random_state=cfg.split_seed,
        stratify=trial_labels[tv_pos],
    )

    def epochs_of(trial_positions: np.ndarray) -> np.ndarray:
        return np.where(np.isin(trials, uniq[trial_positions]))[0]

    return DataSplit(
        train=epochs_of(train_pos),
        val=epochs_of(val_pos),
        test=epochs_of(test_pos),
    )


@dataclass
class DataLoaders:
    """The three loaders plus the underlying split (for inspection / logging)."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    split: DataSplit


def make_datasets(data: EpochedData, split: DataSplit, cfg: Config):
    """Build train/val/test :class:`EEGDataset`s (augmentation on train only)."""
    augment = Augmentation(cfg.data.augment, rng=np.random.default_rng(cfg.train.seed))
    train_ds = EEGDataset(data.X[split.train], data.y[split.train], augment=augment)
    val_ds = EEGDataset(data.X[split.val], data.y[split.val], augment=None)
    test_ds = EEGDataset(data.X[split.test], data.y[split.test], augment=None)
    return train_ds, val_ds, test_ds


def build_dataloaders(data: EpochedData, cfg: Config) -> DataLoaders:
    """Split the pooled epochs and wrap each split in a DataLoader."""
    split = split_by_trial(data, cfg.data)
    train_ds, val_ds, test_ds = make_datasets(data, split, cfg)

    generator = torch.Generator().manual_seed(cfg.train.seed)
    common = dict(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=False, generator=generator, **common
    )
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return DataLoaders(train_loader, val_loader, test_loader, split)
