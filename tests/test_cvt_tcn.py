"""Tests for the hybrid CvT + TCN model."""

import pytest
import torch

from cvttcn.config import Config
from cvttcn.models.cvt_tcn import CvTTCN, build_model


def test_forward_produces_class_logits():
    model = build_model(Config()).eval()
    out = model(torch.randn(4, 1, 64, 640))
    assert out.shape == (4, 2)
    assert torch.isfinite(out).all()


def test_num_classes_is_configurable():
    cfg = Config.from_dict({"model": {"num_classes": 4}})
    model = build_model(cfg).eval()
    assert model(torch.randn(2, 1, 64, 640)).shape == (2, 4)


def test_variable_epoch_length_via_global_pool():
    model = build_model(Config()).eval()
    # Global average pooling over time -> classifier is length-agnostic.
    assert model(torch.randn(2, 1, 64, 320)).shape == (2, 2)
    assert model(torch.randn(2, 1, 64, 800)).shape == (2, 2)


def test_channel_count_follows_data_config():
    cfg = Config.from_dict({"data": {"n_channels": 32}})
    model = build_model(cfg).eval()
    assert model(torch.randn(2, 1, 32, 640)).shape == (2, 2)


def test_backward_populates_finite_gradients():
    model = build_model(Config())
    logits = model(torch.randn(3, 1, 64, 640))
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0]))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


def test_eval_forward_is_deterministic():
    model = build_model(Config()).eval()
    x = torch.randn(2, 1, 64, 640)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x))


def test_has_trainable_parameters():
    model = build_model(Config())
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_on_cuda():
    model = build_model(Config()).cuda().eval()
    out = model(torch.randn(4, 1, 64, 640, device="cuda"))
    assert out.shape == (4, 2)
    assert out.is_cuda and torch.isfinite(out).all()
