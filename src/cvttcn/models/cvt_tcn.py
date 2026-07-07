"""The hybrid CvT + TCN model for EEG motor-imagery classification.

Data flow for an input epoch ``(B, 1, n_channels, T)``:

  ConvEmbedding    collapse the channel axis, tokenize along time -> (B, T', d)
  CvT blocks       depth conv-transformer blocks, token length preserved
  transpose        (B, T', d) -> (B, d, T') so the sequence is channel-first
  TCN              dilated-causal temporal modelling             -> (B, d_tcn, T')
  global avg pool  average over time                             -> (B, d_tcn)
  LayerNorm + head classify                                      -> (B, num_classes)

The CvT branch acts as a spatial/spectral tokenizer and encoder; the TCN branch
models long-range temporal structure. Global average pooling makes the classifier
agnostic to the exact number of tokens, so the model accepts variable-length
epochs.
"""

import torch
import torch.nn as nn

from cvttcn.config import Config, ModelConfig
from cvttcn.models.cvt import ConvEmbedding, ConvTransformerBlock
from cvttcn.models.tcn import build_tcn


class CvTTCN(nn.Module):
    """Convolutional Vision Transformer followed by a Temporal Convolutional Net."""

    def __init__(self, model_cfg: ModelConfig, n_eeg_channels: int):
        super().__init__()
        cvt = model_cfg.cvt

        self.embedding = ConvEmbedding(
            in_channels=model_cfg.in_channels,
            embed_dim=cvt.embed_dim,
            n_eeg_channels=n_eeg_channels,
            kernel_t=cvt.embed_kernel,
            stride_t=cvt.embed_stride,
        )
        self.blocks = nn.ModuleList(
            ConvTransformerBlock(
                dim=cvt.embed_dim,
                num_heads=cvt.num_heads,
                mlp_ratio=cvt.mlp_ratio,
                kernel_size=cvt.proj_kernel,
                stride_q=cvt.proj_stride_q,
                stride_kv=cvt.proj_stride_kv,
                dropout=cvt.dropout,
                attn_dropout=cvt.attn_dropout,
            )
            for _ in range(cvt.depth)
        )
        self.tcn = build_tcn(num_inputs=cvt.embed_dim, cfg=model_cfg.tcn)
        self.norm = nn.LayerNorm(self.tcn.num_outputs)
        self.head = nn.Linear(self.tcn.num_outputs, model_cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, 1, n_channels, T)`` -> logits ``(B, num_classes)``."""
        x = self.embedding(x)          # (B, T', d)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)          # (B, d, T')
        x = self.tcn(x)                # (B, d_tcn, T')
        x = x.mean(dim=-1)             # global average pool over time
        x = self.norm(x)
        return self.head(x)


def build_model(cfg: Config) -> CvTTCN:
    """Construct a :class:`CvTTCN` from a full :class:`Config`.

    The number of EEG channels is taken from the data config so the token
    embedding's channel-collapsing kernel matches the preprocessed epochs.
    """
    return CvTTCN(cfg.model, n_eeg_channels=cfg.data.n_channels)
