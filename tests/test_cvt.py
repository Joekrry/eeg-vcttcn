"""Tests for the Convolutional Vision Transformer building blocks."""

import pytest
import torch

from cvttcn.models.cvt import (
    ConvAttention,
    ConvEmbedding,
    ConvProjection,
    ConvTransformerBlock,
)


# --- ConvEmbedding ---------------------------------------------------------
def test_conv_embedding_collapses_channels_and_tokenizes_time():
    embed = ConvEmbedding(in_channels=1, embed_dim=64, n_eeg_channels=64, kernel_t=7, stride_t=4)
    x = torch.randn(2, 1, 64, 640)
    out = embed(x)
    # time 640 with kernel 7, stride 4, pad 3 -> 160 tokens; channels collapsed.
    assert out.shape == (2, 160, 64)


def test_conv_embedding_token_count_follows_stride():
    embed = ConvEmbedding(1, 32, n_eeg_channels=64, kernel_t=7, stride_t=2)
    out = embed(torch.randn(1, 1, 64, 640))
    assert out.shape == (1, 320, 32)  # stride 2 -> 320 tokens


# --- ConvProjection --------------------------------------------------------
def test_conv_projection_stride1_preserves_length():
    proj = ConvProjection(dim=16, kernel_size=3, stride=1)
    out = proj(torch.randn(2, 50, 16))
    assert out.shape == (2, 50, 16)


def test_conv_projection_stride2_subsamples_tokens():
    proj = ConvProjection(dim=16, kernel_size=3, stride=2)
    out = proj(torch.randn(2, 50, 16))
    assert out.shape == (2, 25, 16)  # ceil(50/2)


# --- ConvAttention ---------------------------------------------------------
def test_conv_attention_output_matches_query_length():
    attn = ConvAttention(dim=32, num_heads=8, kernel_size=3, stride_q=1, stride_kv=2)
    x = torch.randn(2, 40, 32)
    out = attn(x)
    assert out.shape == (2, 40, 32)  # output length == Q length (stride_q=1)


def test_conv_attention_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="divisible"):
        ConvAttention(dim=30, num_heads=8, kernel_size=3, stride_q=1, stride_kv=2)


# --- ConvTransformerBlock --------------------------------------------------
def test_block_preserves_shape():
    block = ConvTransformerBlock(
        dim=32, num_heads=4, mlp_ratio=4.0, kernel_size=3, stride_q=1, stride_kv=2
    )
    x = torch.randn(2, 40, 32)
    assert block(x).shape == (2, 40, 32)


def test_stacked_blocks_preserve_tokens():
    blocks = torch.nn.Sequential(
        *[
            ConvTransformerBlock(32, 4, 4.0, 3, stride_q=1, stride_kv=2)
            for _ in range(3)
        ]
    )
    x = torch.randn(2, 40, 32)
    assert blocks(x).shape == (2, 40, 32)


def test_block_has_no_positional_embedding():
    block = ConvTransformerBlock(32, 4, 4.0, 3, stride_q=1, stride_kv=2)
    names = [n for n, _ in block.named_parameters()]
    assert not any("pos" in n.lower() for n in names)


def test_block_gradient_flow():
    block = ConvTransformerBlock(32, 4, 4.0, 3, stride_q=1, stride_kv=2)
    x = torch.randn(2, 40, 32, requires_grad=True)
    block(x).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = [p.grad for p in block.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


# --- end-to-end embedding + blocks -----------------------------------------
def test_embedding_then_blocks_chain():
    embed = ConvEmbedding(1, 64, n_eeg_channels=64, kernel_t=7, stride_t=4)
    block = ConvTransformerBlock(64, 8, 4.0, 3, stride_q=1, stride_kv=2)
    out = block(embed(torch.randn(2, 1, 64, 640)))
    assert out.shape == (2, 160, 64)  # ready to transpose to (B, 64, 160) for the TCN
