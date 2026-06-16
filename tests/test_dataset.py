"""Tests for the dataset, pooled split-by-trial, augmentation, and DataLoaders."""

import numpy as np
import pytest
import torch

from cvttcn.config import AugmentConfig, Config
from cvttcn.data.dataset import (
    Augmentation,
    EEGDataset,
    build_dataloaders,
    make_datasets,
    split_by_trial,
)
from cvttcn.data.preprocessing import EpochedData

C, T = 8, 16  # small spatial/temporal dims keep these tests fast


def _synthetic(n_trials=80, per_trial=2, seed=0) -> EpochedData:
    """EpochedData with several epochs per trial and balanced trial labels."""
    rng = np.random.default_rng(seed)
    n = n_trials * per_trial
    X = rng.standard_normal((n, 1, C, T)).astype(np.float32)
    trial_labels = np.array([t % 2 for t in range(n_trials)], dtype=np.int64)  # balanced
    y = np.repeat(trial_labels, per_trial)
    trials = np.repeat(np.arange(n_trials), per_trial).astype(np.int64)
    subjects = (np.repeat(np.arange(n_trials), per_trial) // 8).astype(np.int64)
    return EpochedData(X, y, subjects, trials)


# --- split_by_trial --------------------------------------------------------
def test_split_partitions_all_epochs_without_overlap():
    data = _synthetic()
    split = split_by_trial(data, Config().data)
    all_idx = np.concatenate([split.train, split.val, split.test])
    assert np.array_equal(np.sort(all_idx), np.arange(len(data)))
    # pairwise disjoint
    assert not (set(split.train) & set(split.val))
    assert not (set(split.train) & set(split.test))
    assert not (set(split.val) & set(split.test))


def test_split_has_no_trial_leakage():
    data = _synthetic()
    split = split_by_trial(data, Config().data)
    tr = set(data.trials[split.train])
    va = set(data.trials[split.val])
    te = set(data.trials[split.test])
    # a trial must live entirely in exactly one split
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(np.unique(data.trials))


def test_split_sizes_match_config_fractions():
    data = _synthetic(n_trials=200)
    cfg = Config().data
    split = split_by_trial(data, cfg)
    n = len(data)
    assert abs(len(split.test) / n - cfg.test_size) < 0.03
    assert abs(len(split.val) / n - cfg.val_size) < 0.03


def test_split_is_stratified_and_balanced():
    data = _synthetic()
    split = split_by_trial(data, Config().data)
    for idx in (split.train, split.val, split.test):
        labels = data.y[idx]
        assert set(np.unique(labels)) == {0, 1}
        # roughly balanced (started 50/50)
        assert abs(labels.mean() - 0.5) < 0.2


def test_split_is_deterministic_for_a_seed():
    data = _synthetic()
    s1 = split_by_trial(data, Config().data)
    s2 = split_by_trial(data, Config().data)
    assert np.array_equal(s1.train, s2.train)
    assert np.array_equal(s1.test, s2.test)


def test_split_changes_with_seed():
    data = _synthetic()
    a = split_by_trial(data, Config.from_dict({"data": {"split_seed": 1}}).data)
    b = split_by_trial(data, Config.from_dict({"data": {"split_seed": 2}}).data)
    assert not np.array_equal(a.test, b.test)


# --- EEGDataset ------------------------------------------------------------
def test_dataset_item_shape_and_dtype():
    data = _synthetic(n_trials=4)
    ds = EEGDataset(data.X, data.y)
    x, y = ds[0]
    assert x.shape == (1, C, T)
    assert x.dtype == torch.float32
    assert y.dtype == torch.long and y.item() in (0, 1)
    assert len(ds) == len(data)


# --- Augmentation ----------------------------------------------------------
def test_augmentation_preserves_shape():
    aug = Augmentation(AugmentConfig(), rng=np.random.default_rng(0))
    x = np.random.default_rng(1).standard_normal((1, C, T)).astype(np.float32)
    out = aug(x)
    assert out.shape == x.shape and out.dtype == np.float32


def test_augmentation_disabled_is_identity():
    aug = Augmentation(AugmentConfig(enabled=False))
    x = np.random.default_rng(2).standard_normal((1, C, T)).astype(np.float32)
    assert np.array_equal(aug(x), x)


def test_augmentation_is_deterministic_with_seed():
    cfg = AugmentConfig()
    x = np.random.default_rng(3).standard_normal((1, C, T)).astype(np.float32)
    a = Augmentation(cfg, rng=np.random.default_rng(42))(x.copy())
    b = Augmentation(cfg, rng=np.random.default_rng(42))(x.copy())
    assert np.allclose(a, b)


def test_augmentation_changes_input():
    aug = Augmentation(AugmentConfig(), rng=np.random.default_rng(0))
    x = np.ones((1, C, T), dtype=np.float32)
    assert not np.array_equal(aug(x), x)


def test_train_dataset_augments_val_does_not():
    data = _synthetic()
    cfg = Config()
    split = split_by_trial(data, cfg.data)
    train_ds, val_ds, _ = make_datasets(data, split, cfg)
    assert train_ds.augment is not None
    assert val_ds.augment is None


# --- build_dataloaders -----------------------------------------------------
def test_build_dataloaders_batches_have_expected_shapes():
    data = _synthetic()
    cfg = Config.from_dict({"train": {"batch_size": 16, "num_workers": 0}})
    loaders = build_dataloaders(data, cfg)
    xb, yb = next(iter(loaders.train))
    assert xb.shape[1:] == (1, C, T)
    assert xb.shape[0] == yb.shape[0] <= 16
    assert xb.dtype == torch.float32 and yb.dtype == torch.long


def test_build_dataloaders_no_leakage_between_loaders():
    data = _synthetic()
    loaders = build_dataloaders(data, Config())
    tr = set(data.trials[loaders.split.train])
    te = set(data.trials[loaders.split.test])
    assert not (tr & te)


def test_build_dataloaders_covers_every_epoch_once():
    data = _synthetic()
    loaders = build_dataloaders(data, Config())
    total = len(loaders.split.train) + len(loaders.split.val) + len(loaders.split.test)
    assert total == len(data)
