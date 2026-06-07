"""Preprocessing and epoching of EEGMMIDB recordings.

Turns raw EDF files into model-ready epoch tensors for the binary imagined
left-vs-right fist task. Per run the pipeline is:

  read EDF -> standardize channel names -> band-pass to the mu/beta band ->
  extract T1/T2 events -> epoch around the cue -> crop to a fixed length ->
  per-epoch, per-channel z-score.

Epochs from a subject's runs (and across subjects) are concatenated. The result
is an :class:`EpochedData` holding ``X`` of shape ``(N, 1, n_channels, n_times)``
plus integer labels and subject / trial ids used for the pooled split later.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

import mne
import numpy as np
from mne.datasets import eegbci

from cvttcn.config import DataConfig
from cvttcn.data.download import download_subject, subjects_from_config

# Annotation -> integer class label, valid for the imagined-fist runs (4, 8, 12).
# T0 = rest (dropped); T1 = left fist; T2 = right fist.
ANNOT_LABELS = {"T1": 0, "T2": 1}
CLASS_NAMES = ("left_fist", "right_fist")

_ZSCORE_EPS = 1e-12  # std below this means a constant channel (avoid div-by-zero)


@dataclass
class EpochedData:
    """Epoched EEG ready for tensor conversion.

    Attributes:
        X: float32 array, shape ``(N, 1, n_channels, n_times)``.
        y: int64 array, shape ``(N,)``, values in ``{0, 1}``.
        subjects: int64 array, shape ``(N,)``, subject id of each epoch.
        trials: int64 array, shape ``(N,)``, unique trial id of each epoch
            (currently one epoch == one trial; used for split-by-trial).
    """

    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    trials: np.ndarray

    def __len__(self) -> int:
        return int(self.X.shape[0])


def expected_n_times(cfg: DataConfig) -> int:
    """Number of time samples per epoch after cropping: ``round((tmax-tmin)*sfreq)``."""
    return int(round((cfg.tmax - cfg.tmin) * cfg.sfreq))


def zscore_epochs(X: np.ndarray) -> np.ndarray:
    """Per-epoch, per-channel z-score over the time axis. ``X`` is ``(N, ch, t)``.

    Scale-robust: constant channels (std ~ 0) map to zeros instead of being divided
    by a scale-dependent epsilon, which matters because EEG is stored in volts.
    """
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    denom = np.where(std < _ZSCORE_EPS, 1.0, std)
    return (X - mean) / denom


def load_raw(
    edf_path: Union[str, Path],
    cfg: DataConfig,
    verbose: Union[str, bool, None] = "ERROR",
) -> mne.io.BaseRaw:
    """Read one EDF run and apply channel standardization, resampling, filtering."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=verbose)
    eegbci.standardize(raw)  # 'Fc5.' -> 'FC5', consistent naming across runs
    if abs(float(raw.info["sfreq"]) - cfg.sfreq) > 1e-3:
        raw.resample(cfg.sfreq, verbose=verbose)
    raw.filter(cfg.l_freq, cfg.h_freq, fir_design="firwin", verbose=verbose)
    return raw


def epochs_from_raw(
    raw: mne.io.BaseRaw,
    cfg: DataConfig,
    verbose: Union[str, bool, None] = "ERROR",
) -> mne.Epochs:
    """Extract fixed-length T1/T2 epochs (rest T0 dropped) from a filtered raw."""
    events, event_id = mne.events_from_annotations(raw, verbose=verbose)
    wanted = {name: code for name, code in event_id.items() if str(name) in ANNOT_LABELS}
    if not wanted:
        raise ValueError(
            "No T1/T2 annotations found; this run is not an imagined-fist run."
        )
    return mne.Epochs(
        raw,
        events,
        event_id=wanted,
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        baseline=None,
        picks="eeg",
        preload=True,
        verbose=verbose,
    )


def _labels_from_epochs(epochs: mne.Epochs) -> np.ndarray:
    """Map each epoch's annotation code back to its 0/1 class label."""
    code_to_name = {int(code): str(name) for name, code in epochs.event_id.items()}
    return np.array(
        [ANNOT_LABELS[code_to_name[int(c)]] for c in epochs.events[:, 2]],
        dtype=np.int64,
    )


def epochs_to_arrays(epochs: mne.Epochs, cfg: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    """Convert epochs to ``(X, y)`` with ``X`` of shape ``(N, 1, ch, n_times)``."""
    n_times = expected_n_times(cfg)
    data = epochs.get_data(copy=True).astype(np.float32)  # (N, ch, t)
    if data.shape[-1] < n_times:
        raise ValueError(
            f"Epoch length {data.shape[-1]} is shorter than expected {n_times}."
        )
    data = data[:, :, :n_times]
    if cfg.normalize == "zscore":
        data = zscore_epochs(data)
    X = data[:, np.newaxis, :, :].astype(np.float32)  # add the singleton "image" channel
    y = _labels_from_epochs(epochs)
    return X, y


def preprocess_subject(
    subject: int,
    cfg: DataConfig,
    force_update: bool = False,
    verbose: Union[str, bool, None] = "ERROR",
) -> EpochedData:
    """Download (if needed) and epoch all configured runs for one subject."""
    paths = download_subject(subject, cfg.runs, cfg.data_root, force_update, verbose)
    X_parts, y_parts = [], []
    for path in paths:
        raw = load_raw(path, cfg, verbose)
        epochs = epochs_from_raw(raw, cfg, verbose)
        X, y = epochs_to_arrays(epochs, cfg)
        X_parts.append(X)
        y_parts.append(y)
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    subjects = np.full(len(y), subject, dtype=np.int64)
    trials = np.arange(len(y), dtype=np.int64)
    return EpochedData(X, y, subjects, trials)


def preprocess_subjects(
    subjects: Iterable[int],
    cfg: DataConfig,
    force_update: bool = False,
    verbose: Union[str, bool, None] = "ERROR",
) -> EpochedData:
    """Epoch several subjects and concatenate, assigning globally-unique trial ids."""
    parts = [preprocess_subject(s, cfg, force_update, verbose) for s in subjects]
    X = np.concatenate([p.X for p in parts], axis=0)
    y = np.concatenate([p.y for p in parts], axis=0)
    subjects_arr = np.concatenate([p.subjects for p in parts], axis=0)
    trials = np.arange(len(y), dtype=np.int64)  # one epoch == one trial, globally unique
    return EpochedData(X, y, subjects_arr, trials)


def build_epochs(
    cfg: DataConfig,
    force_update: bool = False,
    verbose: Union[str, bool, None] = "ERROR",
) -> EpochedData:
    """Epoch every subject selected by ``cfg`` (the full pooled dataset)."""
    return preprocess_subjects(subjects_from_config(cfg), cfg, force_update, verbose)
