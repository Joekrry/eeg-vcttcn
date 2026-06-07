"""Tests for preprocessing & epoching.

Logic is tested against a synthetic in-memory MNE Raw so the suite stays offline.
A single real-data integration test is opt-in via CVTTCN_NETWORK_TESTS=1.
"""

import os

import mne
import numpy as np
import pytest

from cvttcn.config import Config
from cvttcn.data import preprocessing
from cvttcn.data.preprocessing import (
    ANNOT_LABELS,
    CLASS_NAMES,
    EpochedData,
    epochs_from_raw,
    epochs_to_arrays,
    expected_n_times,
    preprocess_subjects,
    zscore_epochs,
)

mne.set_log_level("ERROR")


def _make_raw(cfg, n_each=3, seed=0):
    """Synthetic 64-channel raw with interleaved T0/T1/T2 annotations.

    Returns (raw, task_order) where task_order lists the T1/T2 sequence in time.
    """
    rng = np.random.default_rng(seed)
    n_ch = cfg.n_channels
    sfreq = cfg.sfreq
    spacing = (cfg.tmax - cfg.tmin) + 1.0  # leave a gap between cues

    onsets, descs, task_order = [], [], []
    t = 1.0
    for _ in range(n_each):
        for desc in ("T1", "T2"):
            onsets.append(t)
            descs.append("T0")  # a rest marker before each task cue
            t += 1.0
            onsets.append(t)
            descs.append(desc)
            task_order.append(desc)
            t += spacing

    duration_s = t + cfg.tmax + 2.0
    n_samp = int(duration_s * sfreq)
    data = rng.standard_normal((n_ch, n_samp)).astype(np.float64) * 1e-5  # ~volts
    info = mne.create_info([f"EEG{i:03d}" for i in range(n_ch)], sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(onset=onsets, duration=[0.0] * len(onsets), description=descs)
    )
    return raw, task_order


# --- pure helpers ----------------------------------------------------------
def test_label_constants_are_consistent():
    assert ANNOT_LABELS == {"T1": 0, "T2": 1}
    assert CLASS_NAMES[0] == "left_fist" and CLASS_NAMES[1] == "right_fist"


def test_expected_n_times_default():
    cfg = Config().data
    assert expected_n_times(cfg) == 640  # 4.0 s * 160 Hz


def test_zscore_epochs_normalizes_per_channel():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=3.0, size=(4, 64, 640)).astype(np.float32)
    Z = zscore_epochs(X)
    assert np.allclose(Z.mean(axis=-1), 0.0, atol=1e-4)
    assert np.allclose(Z.std(axis=-1), 1.0, atol=1e-3)


def test_zscore_handles_constant_channel_without_nan():
    X = np.ones((1, 64, 640), dtype=np.float32)
    assert np.isfinite(zscore_epochs(X)).all()


# --- epoching on synthetic raw (offline) -----------------------------------
def test_epochs_from_raw_drops_rest_and_keeps_t1_t2():
    cfg = Config().data
    raw, task_order = _make_raw(cfg, n_each=3)
    epochs = epochs_from_raw(raw, cfg)
    assert len(epochs) == len(task_order)  # only T1/T2, the T0 rest cues dropped
    assert set(map(str, epochs.event_id)) == {"T1", "T2"}


def test_labels_follow_annotation_order():
    cfg = Config().data
    raw, task_order = _make_raw(cfg, n_each=3)
    epochs = epochs_from_raw(raw, cfg)
    _, y = epochs_to_arrays(epochs, cfg)
    expected = np.array([ANNOT_LABELS[d] for d in task_order], dtype=np.int64)
    assert np.array_equal(y, expected)


def test_epochs_to_arrays_shape_dtype_and_finiteness():
    cfg = Config().data
    raw, task_order = _make_raw(cfg, n_each=4)
    epochs = epochs_from_raw(raw, cfg)
    X, y = epochs_to_arrays(epochs, cfg)
    assert X.shape == (len(task_order), 1, cfg.n_channels, expected_n_times(cfg))
    assert X.dtype == np.float32 and y.dtype == np.int64
    assert np.isfinite(X).all()
    assert set(np.unique(y)).issubset({0, 1})
    # z-score applied: per epoch & channel mean ~0, std ~1
    flat = X[:, 0]  # (N, ch, t)
    assert np.allclose(flat.mean(axis=-1), 0.0, atol=1e-4)
    assert np.allclose(flat.std(axis=-1), 1.0, atol=1e-3)


def test_no_normalization_when_disabled():
    cfg = Config.from_dict({"data": {"normalize": "none"}}).data
    raw, _ = _make_raw(cfg, n_each=2)
    epochs = epochs_from_raw(raw, cfg)
    X, _ = epochs_to_arrays(epochs, cfg)
    # Raw synthetic signal is on a ~1e-5 scale, so it is clearly not z-scored.
    assert np.abs(X).max() < 1e-3


# --- aggregation across subjects (mocked, offline) -------------------------
def test_preprocess_subjects_concatenates_with_global_trial_ids(monkeypatch):
    cfg = Config().data
    n_times = expected_n_times(cfg)

    def fake_subject(subject, c, *args, **kwargs):
        n = 2
        X = np.zeros((n, 1, c.n_channels, n_times), dtype=np.float32)
        y = np.array([0, 1], dtype=np.int64)
        return EpochedData(X, y, np.full(n, subject, dtype=np.int64), np.arange(n))

    monkeypatch.setattr(preprocessing, "preprocess_subject", fake_subject)
    data = preprocess_subjects([1, 2], cfg)
    assert len(data) == 4
    assert list(data.subjects) == [1, 1, 2, 2]
    assert list(data.trials) == [0, 1, 2, 3]  # globally unique across subjects


# --- real data integration (opt-in) ----------------------------------------
@pytest.mark.skipif(
    os.environ.get("CVTTCN_NETWORK_TESTS") != "1",
    reason="set CVTTCN_NETWORK_TESTS=1 to download and preprocess real EEGMMIDB data",
)
def test_preprocess_real_subject_one():
    cfg = Config().data
    data = preprocessing.preprocess_subject(1, cfg)
    assert data.X.ndim == 4
    assert data.X.shape[1:] == (1, cfg.n_channels, expected_n_times(cfg))
    assert len(data) == len(data.y) == len(data.subjects)
    assert set(np.unique(data.y)).issubset({0, 1})
    assert np.isfinite(data.X).all()
