"""Tests for the training utilities and the Trainer."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cvttcn.config import Config
from cvttcn.models.cvt_tcn import build_model
from cvttcn.training.trainer import Trainer
from cvttcn.training.utils import resolve_device, set_seed


# --- utils -----------------------------------------------------------------
def test_set_seed_makes_torch_reproducible():
    set_seed(123)
    a = torch.randn(10)
    set_seed(123)
    b = torch.randn(10)
    assert torch.equal(a, b)


def test_resolve_device_explicit_and_auto():
    assert resolve_device("cpu").type == "cpu"
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto").type == expected


# --- a tiny, easily separable problem --------------------------------------
def _toy_config(**train_overrides):
    train = {"batch_size": 16, "device": "cpu", "amp": False, "warmup_epochs": 1}
    train.update(train_overrides)
    return Config.from_dict(
        {
            "data": {"n_channels": 8},
            "model": {
                "cvt": {"embed_dim": 16, "depth": 1, "num_heads": 2},
                "tcn": {"channels": [16]},
            },
            "train": train,
        }
    )


def _toy_loaders(n_per_class=32, n_time=64, seed=0):
    """Two classes separated by a constant offset -> trivially learnable."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    n_ch = 8
    x0 = rng.normal(0.0, 0.1, size=(n_per_class, 1, n_ch, n_time))
    x1 = rng.normal(1.5, 0.1, size=(n_per_class, 1, n_ch, n_time))
    X = np.concatenate([x0, x1]).astype(np.float32)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int64)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    return loader


def test_single_train_epoch_reduces_loss():
    set_seed(0)
    cfg = _toy_config(epochs=1)
    trainer = Trainer(build_model(cfg), cfg)
    loader = _toy_loaders()
    before = trainer.evaluate(loader).loss
    trainer.train_epoch(loader)
    after = trainer.evaluate(loader).loss
    assert after < before


def test_fit_learns_the_toy_task():
    set_seed(0)
    cfg = _toy_config(epochs=8, early_stopping_patience=100)
    trainer = Trainer(build_model(cfg), cfg)
    loader = _toy_loaders()
    history = trainer.fit(loader, loader)
    assert history[-1]["train"].loss < history[0]["train"].loss
    assert history[-1]["val"].accuracy > 0.9  # should master a trivial split


def test_early_stopping_triggers_on_stagnation():
    set_seed(0)
    cfg = _toy_config(epochs=50, early_stopping_patience=2, lr=0.0)  # frozen -> no gains
    trainer = Trainer(build_model(cfg), cfg)
    loader = _toy_loaders()
    history = trainer.fit(loader, loader)
    assert len(history) < cfg.train.epochs  # stopped early


def test_checkpoint_round_trip(tmp_path):
    set_seed(0)
    cfg = _toy_config(epochs=1)
    trainer = Trainer(build_model(cfg), cfg)
    loader = _toy_loaders()
    trainer.train_epoch(loader)

    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path)

    fresh = Trainer(build_model(cfg), cfg)
    fresh.load_checkpoint(path)

    trainer.model.eval()
    fresh.model.eval()
    x = torch.randn(4, 1, 8, 64)
    with torch.no_grad():
        assert torch.allclose(trainer.model(x), fresh.model(x), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fit_with_amp_on_cuda():
    set_seed(0)
    cfg = _toy_config(epochs=2, amp=True, device="cuda")
    trainer = Trainer(build_model(cfg), cfg)
    assert trainer.use_amp
    loader = _toy_loaders()
    history = trainer.fit(loader, loader)
    assert len(history) >= 1
    assert np.isfinite(history[-1]["train"].loss)
