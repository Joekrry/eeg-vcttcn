"""Acquisition of EEGMMIDB recordings via MNE's ``eegbci`` loader.

This module is responsible only for *fetching* the raw EDF files and for deciding
which subjects/runs to use. Filtering, epoching, and tensor creation live in
``preprocessing.py`` (next commit).

Downloads are cached under the project-local ``data/`` directory. ``load_data`` is
always called with ``update_path=False`` so it neither prompts interactively nor
mutates the global MNE configuration.
"""

from pathlib import Path
from typing import Iterable, Optional, Union

from mne.datasets import eegbci

from cvttcn.config import DataConfig
from cvttcn.data import constants


def resolve_subjects(
    subjects: Optional[Iterable[int]] = None,
    excluded: Optional[Iterable[int]] = None,
    n_max: int = constants.N_SUBJECTS,
) -> list[int]:
    """Return the sorted list of subject ids to use.

    ``subjects=None`` selects every subject in ``1..n_max`` minus ``excluded``;
    otherwise the explicit list is validated against the valid range and then has
    ``excluded`` removed.
    """
    excluded_set = set(excluded or ())
    if subjects is None:
        candidates: Iterable[int] = range(1, n_max + 1)
    else:
        candidates = list(subjects)
        for s in candidates:
            if not (1 <= s <= n_max):
                raise ValueError(f"Subject {s} out of valid range 1..{n_max}.")
    result = sorted(s for s in set(candidates) if s not in excluded_set)
    if not result:
        raise ValueError("No subjects remain after applying exclusions.")
    return result


def subjects_from_config(cfg: DataConfig) -> list[int]:
    """Resolve the subject list described by a :class:`DataConfig`."""
    return resolve_subjects(cfg.subjects, cfg.excluded_subjects, cfg.n_subjects_max)


def data_root_path(data_root: Union[str, Path]) -> Path:
    """Return an absolute, existing directory for cached downloads."""
    path = Path(data_root).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_subject(
    subject: int,
    runs: Iterable[int],
    data_root: Union[str, Path] = "data",
    force_update: bool = False,
    verbose: Union[str, bool, None] = "ERROR",
) -> list[Path]:
    """Fetch the EDF files for one subject and return their local paths."""
    root = data_root_path(data_root)
    paths = eegbci.load_data(
        subjects=subject,
        runs=list(runs),
        path=str(root),
        update_path=False,        # never prompt; never touch global MNE config
        force_update=force_update,
        verbose=verbose,
    )
    return [Path(p) for p in paths]


def download_dataset(
    cfg: DataConfig,
    force_update: bool = False,
    verbose: Union[str, bool, None] = "ERROR",
) -> dict[int, list[Path]]:
    """Fetch every configured subject/run and return ``{subject: [edf paths]}``."""
    subjects = subjects_from_config(cfg)
    return {
        subject: download_subject(subject, cfg.runs, cfg.data_root, force_update, verbose)
        for subject in subjects
    }
