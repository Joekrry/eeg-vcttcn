"""Convolutional Vision Transformer (CvT) building blocks, adapted for EEG.

Follows the three ideas of CvT (Wu et al., 2021): a convolutional token
embedding, convolutional (depthwise) projections for Q/K/V instead of linear
ones, and no explicit positional embedding (locality comes from the convolutions).

The input EEG epoch is shaped ``(B, 1, n_channels, T)``. The token embedding
applies a convolution whose height kernel spans all channels, collapsing the
spatial (channel) axis and tokenizing along time into ``(B, T', embed_dim)``.
The transformer blocks then operate on that temporal token sequence and preserve
its length, so blocks stack and the output feeds the TCN branch in the hybrid
model.
"""

import torch
import torch.nn as nn
from einops import rearrange


class ConvEmbedding(nn.Module):
    """Convolutional token embedding: collapse channels, tokenize along time.

    ``(B, in_channels, n_eeg_channels, T)`` -> ``(B, T', embed_dim)``.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        n_eeg_channels: int,
        kernel_t: int,
        stride_t: int,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(n_eeg_channels, kernel_t),
            stride=(1, stride_t),
            padding=(0, kernel_t // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)          # (B, embed_dim, 1, T')
        x = x.squeeze(2)          # (B, embed_dim, T')
        x = x.transpose(1, 2)     # (B, T', embed_dim)
        return self.norm(x)


class ConvProjection(nn.Module):
    """Depthwise-separable conv projection used for Q/K/V (CvT).

    Operates on a temporal token sequence ``(B, T, C)``; a stride > 1 subsamples
    the tokens (used for K and V to cut attention cost).
    """

    def __init__(self, dim: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)         # (B, C, T)
        x = self.bn(self.conv(x))     # (B, C, T')
        return x.transpose(1, 2)      # (B, T', C)


class ConvAttention(nn.Module):
    """Multi-head self-attention with convolutional Q/K/V projections.

    Q keeps the full token length (stride_q, typically 1); K and V are subsampled
    (stride_kv). The output length equals the Q length, so a residual connection
    around this module is valid.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        stride_q: int,
        stride_kv: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.conv_q = ConvProjection(dim, kernel_size, stride_q)
        self.conv_k = ConvProjection(dim, kernel_size, stride_kv)
        self.conv_v = ConvProjection(dim, kernel_size, stride_kv)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.to_q(self.conv_q(x))    # (B, Tq, C)
        k = self.to_k(self.conv_k(x))    # (B, Tk, C)
        v = self.to_v(self.conv_v(x))    # (B, Tk, C)

        q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = torch.matmul(attn, v)      # (B, h, Tq, d)

        out = rearrange(out, "b h t d -> b t (h d)")
        return self.proj_drop(self.proj(out))


class ConvTransformerBlock(nn.Module):
    """A CvT block: conv-attention and an MLP, each with a pre-norm residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        kernel_size: int,
        stride_q: int,
        stride_kv: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ConvAttention(
            dim,
            num_heads,
            kernel_size,
            stride_q=stride_q,
            stride_kv=stride_kv,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
