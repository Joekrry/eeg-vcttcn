"""Typed configuration schema for the CVTTCN project.

The whole pipeline (data acquisition, preprocessing, model, trainer) reads its
settings from a single nested ``Config`` object so there is one source of truth.
Configs can be built from defaults, loaded from / written to YAML, or overridden
with a partial dictionary.

Note: ``from __future__ import annotations`` is intentionally *not* used here so
that ``dataclasses.fields(cls)[i].type`` yields real class objects, which lets
``_build`` detect nested dataclasses by reflection.
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Union

import yaml


@dataclass
class AugmentConfig:
    """Training-time data augmentation (applied to the train split only)."""

    enabled: bool = True
    time_shift: int = 20          # max |samples| to circularly roll along time
    noise_std: float = 0.1        # gaussian noise std (data is ~unit-scale post z-score)
    channel_dropout: float = 0.1  # per-channel zeroing probability


@dataclass
class DataConfig:
    """Dataset selection, signal preprocessing, and split settings."""

    # Where MNE caches the raw EDF recordings (project-local, git-ignored).
    data_root: str = "data"
    # ``None`` selects every valid subject (1..n_subjects_max minus exclusions).
    subjects: Union[list[int], None] = None
    # Subjects with inconsistent annotation timing / sampling rate in EEGMMIDB.
    excluded_subjects: list[int] = field(default_factory=lambda: [88, 89, 92, 100])
    # Runs 4, 8, 12 = imagined opening/closing of the left or right fist.
    runs: list[int] = field(default_factory=lambda: [4, 8, 12])
    n_subjects_max: int = 109

    # Signal / epoching.
    sfreq: float = 160.0          # native EEGMMIDB sampling rate (Hz)
    l_freq: float = 8.0           # band-pass low cutoff (mu rhythm)
    h_freq: float = 30.0          # band-pass high cutoff (beta rhythm)
    tmin: float = 0.0             # epoch start relative to cue (s)
    tmax: float = 4.0             # epoch end relative to cue (s)
    n_channels: int = 64          # EEG channels in EEGMMIDB
    normalize: str = "zscore"     # per-epoch, per-channel; "zscore" or "none"

    # Pooled cross-subject split, applied per *trial* to avoid window leakage.
    val_size: float = 0.15
    test_size: float = 0.15
    split_seed: int = 42

    augment: AugmentConfig = field(default_factory=AugmentConfig)


@dataclass
class CvTConfig:
    """Convolutional Vision Transformer branch hyper-parameters."""

    embed_dim: int = 64           # token embedding dimension
    depth: int = 4                # number of conv-transformer blocks
    num_heads: int = 8
    mlp_ratio: float = 4.0
    embed_kernel: int = 7         # conv token-embedding kernel (time axis)
    embed_stride: int = 4         # token-embedding temporal stride
    proj_kernel: int = 3          # depthwise conv-projection kernel for Q/K/V
    proj_stride_kv: int = 2       # spatial reduction stride for K/V
    proj_stride_q: int = 1        # stride for Q
    dropout: float = 0.1
    attn_dropout: float = 0.1


@dataclass
class TCNConfig:
    """Temporal Convolutional Network branch hyper-parameters."""

    channels: list[int] = field(default_factory=lambda: [64, 64, 64])
    kernel_size: int = 3
    dropout: float = 0.2


@dataclass
class ModelConfig:
    """Full hybrid model configuration."""

    in_channels: int = 1
    num_classes: int = 2          # binary: left vs right fist
    cvt: CvTConfig = field(default_factory=CvTConfig)
    tcn: TCNConfig = field(default_factory=TCNConfig)


@dataclass
class TrainConfig:
    """Optimization, scheduling, regularization, and runtime settings."""

    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    label_smoothing: float = 0.0
    early_stopping_patience: int = 15
    amp: bool = True              # mixed precision (CUDA only)
    grad_clip: float = 1.0
    num_workers: int = 0          # 0 is safest on Windows
    seed: int = 42
    device: str = "auto"          # "auto" | "cuda" | "cpu"
    checkpoint_dir: str = "checkpoints"


@dataclass
class Config:
    """Top-level configuration aggregating every section."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # -- construction helpers ------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Build a ``Config`` from a (possibly partial) nested dictionary."""
        return _build(cls, data)

    @classmethod
    def from_yaml(cls, path: "str | Path") -> "Config":
        """Load and validate a ``Config`` from a YAML file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data).validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: "str | Path") -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    # -- validation ----------------------------------------------------------
    def validate(self) -> "Config":
        """Sanity-check interdependent fields; raise ``ValueError`` on misuse."""
        d, m, t = self.data, self.model, self.train
        if not (0.0 < d.val_size < 1.0) or not (0.0 < d.test_size < 1.0):
            raise ValueError("val_size and test_size must each be in (0, 1).")
        if d.val_size + d.test_size >= 1.0:
            raise ValueError("val_size + test_size must leave room for training data.")
        if d.normalize not in {"zscore", "none"}:
            raise ValueError(f"Unknown normalize mode: {d.normalize!r}")
        if d.h_freq <= d.l_freq:
            raise ValueError("h_freq must be greater than l_freq.")
        if d.tmax <= d.tmin:
            raise ValueError("tmax must be greater than tmin.")
        if not (0.0 <= d.augment.channel_dropout < 1.0):
            raise ValueError("augment.channel_dropout must be in [0, 1).")
        if d.augment.noise_std < 0.0:
            raise ValueError("augment.noise_std must be non-negative.")
        if d.augment.time_shift < 0:
            raise ValueError("augment.time_shift must be non-negative.")
        if m.num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if t.device not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"Unknown device: {t.device!r}")
        return self


def _build(cls: type, data: Any) -> Any:
    """Recursively construct a dataclass ``cls`` from a dictionary.

    Nested dataclass fields are built recursively; unknown keys raise so typos in
    a YAML file fail loudly instead of being silently ignored.
    """
    if not is_dataclass(cls):
        return data
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping for {cls.__name__}, got {type(data).__name__}.")

    valid = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(valid)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        ftype = valid[name].type
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _build(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: "str | Path | None" = None) -> Config:
    """Return the default config, or one loaded (and validated) from ``path``."""
    if path is None:
        return Config().validate()
    return Config.from_yaml(path)
