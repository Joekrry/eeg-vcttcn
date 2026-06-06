"""Tests for the data-acquisition layer.

The MNE loader is monkeypatched so the suite never touches the network.
"""

from pathlib import Path

import pytest

from cvttcn.config import Config
from cvttcn.data import constants, download


# --- subject resolution ----------------------------------------------------
def test_resolve_subjects_default_excludes_bad_subjects():
    subjects = download.resolve_subjects(
        None, excluded=constants.DEFAULT_EXCLUDED_SUBJECTS
    )
    assert subjects == sorted(
        s for s in range(1, constants.N_SUBJECTS + 1)
        if s not in constants.DEFAULT_EXCLUDED_SUBJECTS
    )
    assert len(subjects) == constants.N_SUBJECTS - len(constants.DEFAULT_EXCLUDED_SUBJECTS)
    assert not set(subjects) & set(constants.DEFAULT_EXCLUDED_SUBJECTS)


def test_resolve_subjects_respects_explicit_list_and_exclusions():
    assert download.resolve_subjects([3, 1, 2, 88], excluded=[88]) == [1, 2, 3]


def test_resolve_subjects_deduplicates():
    assert download.resolve_subjects([5, 5, 1], excluded=[]) == [1, 5]


@pytest.mark.parametrize("bad", [[0], [110], [-1]])
def test_resolve_subjects_rejects_out_of_range(bad):
    with pytest.raises(ValueError, match="out of valid range"):
        download.resolve_subjects(bad)


def test_resolve_subjects_empty_after_exclusion_raises():
    with pytest.raises(ValueError, match="No subjects remain"):
        download.resolve_subjects([88], excluded=[88])


def test_subjects_from_config_uses_defaults():
    subjects = download.subjects_from_config(Config().data)
    assert len(subjects) == constants.N_SUBJECTS - len(constants.DEFAULT_EXCLUDED_SUBJECTS)


# --- consistency between dataset facts and config defaults -----------------
def test_constants_consistent_with_config_defaults():
    data = Config().data
    assert data.n_subjects_max == constants.N_SUBJECTS
    assert data.n_channels == constants.N_CHANNELS
    assert data.sfreq == constants.SFREQ
    assert data.runs == list(constants.IMAGINED_FIST_RUNS)
    assert data.excluded_subjects == list(constants.DEFAULT_EXCLUDED_SUBJECTS)


# --- download wrappers (mocked loader) -------------------------------------
@pytest.fixture
def fake_loader(monkeypatch):
    """Replace eegbci.load_data with a recorder that returns fake EDF paths."""
    calls = []

    def _fake(subjects, runs, *, path, update_path, force_update, verbose):
        calls.append(
            dict(
                subjects=subjects,
                runs=runs,
                path=path,
                update_path=update_path,
                force_update=force_update,
                verbose=verbose,
            )
        )
        return [
            f"{path}/MNE-eegbci-data/files/S{subjects:03d}/S{subjects:03d}R{run:02d}.edf"
            for run in runs
        ]

    monkeypatch.setattr(download.eegbci, "load_data", _fake)
    return calls


def test_download_subject_passes_correct_arguments(fake_loader, tmp_path):
    paths = download.download_subject(1, [4, 8, 12], data_root=tmp_path)
    assert len(paths) == 3
    assert all(isinstance(p, Path) for p in paths)
    (call,) = fake_loader
    assert call["subjects"] == 1
    assert call["runs"] == [4, 8, 12]
    # Must never prompt or mutate global MNE config.
    assert call["update_path"] is False


def test_download_subject_creates_data_root(fake_loader, tmp_path):
    target = tmp_path / "nested" / "data"
    assert not target.exists()
    download.download_subject(1, [4], data_root=target)
    assert target.exists()


def test_download_dataset_iterates_all_subjects(fake_loader, tmp_path):
    cfg = Config.from_dict(
        {"data": {"subjects": [1, 2], "data_root": str(tmp_path)}}
    ).data
    result = download.download_dataset(cfg)
    assert set(result) == {1, 2}
    assert len(fake_loader) == 2
    assert all(len(paths) == len(cfg.runs) for paths in result.values())
