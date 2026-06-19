"""Tests for the Temporal Convolutional Network module."""

import torch

from cvttcn.config import TCNConfig
from cvttcn.models.tcn import Chomp1d, TemporalBlock, TemporalConvNet, build_tcn


def test_chomp1d_removes_trailing_samples():
    x = torch.arange(2 * 3 * 10, dtype=torch.float32).reshape(2, 3, 10)
    out = Chomp1d(4)(x)
    assert out.shape == (2, 3, 6)
    assert torch.equal(out, x[:, :, :6])


def test_chomp1d_zero_is_identity():
    x = torch.randn(1, 2, 5)
    assert torch.equal(Chomp1d(0)(x), x)


def test_temporal_block_preserves_length_and_sets_channels():
    block = TemporalBlock(4, 8, kernel_size=3, dilation=2)
    x = torch.randn(2, 4, 50)
    out = block(x)
    assert out.shape == (2, 8, 50)  # length preserved, channels -> 8


def test_tcn_output_shape():
    tcn = TemporalConvNet(16, [32, 32, 32], kernel_size=3)
    x = torch.randn(4, 16, 640)
    out = tcn(x)
    assert out.shape == (4, 32, 640)
    assert tcn.num_outputs == 32


def test_tcn_is_causal():
    """Output at time t must not depend on inputs after t."""
    torch.manual_seed(0)
    tcn = TemporalConvNet(8, [16, 16, 16], kernel_size=3, dropout=0.0).eval()
    x = torch.randn(1, 8, 64)
    t = 30
    x2 = x.clone()
    x2[:, :, t + 1:] = torch.randn_like(x2[:, :, t + 1:])  # change only the future
    with torch.no_grad():
        out1 = tcn(x)
        out2 = tcn(x2)
    # everything up to and including t must be identical
    assert torch.allclose(out1[:, :, : t + 1], out2[:, :, : t + 1], atol=1e-6)
    # and the change after t should actually have an effect
    assert not torch.allclose(out1[:, :, t + 1:], out2[:, :, t + 1:], atol=1e-6)


def test_receptive_field_formula():
    k, n_layers = 3, 4
    tcn = TemporalConvNet(8, [16] * n_layers, kernel_size=k)
    expected = 1 + 2 * (k - 1) * (2 ** n_layers - 1)  # dilations 1,2,4,8
    assert tcn.receptive_field == expected


def test_gradients_flow_through_tcn():
    tcn = TemporalConvNet(8, [16, 16], kernel_size=3, dropout=0.0)
    x = torch.randn(2, 8, 32, requires_grad=True)
    out = tcn(x)
    out.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = [p.grad for p in tcn.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_build_tcn_from_config():
    cfg = TCNConfig(channels=[64, 64, 64], kernel_size=3, dropout=0.2)
    tcn = build_tcn(num_inputs=64, cfg=cfg)
    assert isinstance(tcn, TemporalConvNet)
    assert tcn.num_outputs == 64
    out = tcn(torch.randn(2, 64, 128))
    assert out.shape == (2, 64, 128)
