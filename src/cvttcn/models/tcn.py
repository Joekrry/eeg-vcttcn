"""Temporal Convolutional Network (TCN).

A standard dilated-causal TCN (Bai, Kolter & Koltun, 2018): a stack of residual
blocks, each with two weight-normalised dilated 1-D convolutions made causal by
left-padding and chomping the right overhang. Dilation doubles per block so the
receptive field grows exponentially with depth.

In the hybrid model the TCN consumes the CvT token sequence ``(B, d_model, T')``
and models long-range temporal dependencies before global pooling.
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from cvttcn.config import TCNConfig


class Chomp1d(nn.Module):
    """Remove the right-hand overhang introduced by symmetric padding.

    A dilated conv padded by ``(k-1)*dilation`` on both sides produces an output
    ``(k-1)*dilation`` samples longer than the input; chomping that many samples
    from the end keeps the length and makes the convolution strictly causal.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Two dilated-causal convolutions with a residual connection."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.kernel_size: int = kernel_size
        self.dilation: int = dilation
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(n_in, n_out, kernel_size, padding=padding, dilation=dilation)
        )
        self.conv2 = weight_norm(
            nn.Conv1d(n_out, n_out, kernel_size, padding=padding, dilation=dilation)
        )
        self.net = nn.Sequential(
            self.conv1,
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            self.conv2,
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # 1x1 conv to match channels on the residual path when they differ.
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stack of :class:`TemporalBlock`s with exponentially increasing dilation."""

    def __init__(
        self,
        num_inputs: int,
        num_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if not num_channels:
            raise ValueError("num_channels must list at least one layer width.")
        self.kernel_size: int = kernel_size
        self.dilations: list[int] = [2 ** i for i in range(len(num_channels))]
        blocks = []
        for i, out_ch in enumerate(num_channels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            blocks.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    dilation=self.dilations[i],
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.num_outputs: int = num_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, num_inputs, T)`` -> ``(B, num_channels[-1], T)``."""
        for block in self.blocks:
            x = block(x)
        return x

    @property
    def receptive_field(self) -> int:
        """Number of past time steps each output position can see."""
        rf = 1
        for dilation in self.dilations:
            rf += 2 * (self.kernel_size - 1) * dilation
        return rf


def build_tcn(num_inputs: int, cfg: TCNConfig) -> TemporalConvNet:
    """Construct a :class:`TemporalConvNet` from a :class:`TCNConfig`."""
    return TemporalConvNet(
        num_inputs,
        list(cfg.channels),
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
    )
