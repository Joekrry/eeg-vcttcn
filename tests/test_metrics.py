"""Tests for classification metrics, verified against scikit-learn."""

import numpy as np
import pytest
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
)

from cvttcn.training.metrics import (
    ClassificationMetrics,
    accuracy,
    cohen_kappa_from_confusion,
    compute_metrics,
    confusion_matrix,
    macro_f1_from_confusion,
)


@pytest.fixture
def labels():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=200)
    y_pred = y_true.copy()
    flip = rng.random(200) < 0.3  # corrupt ~30% so metrics are non-trivial
    y_pred[flip] = rng.integers(0, 3, size=int(flip.sum()))
    return y_true, y_pred


def test_confusion_matrix_matches_sklearn(labels):
    y_true, y_pred = labels
    cm = confusion_matrix(y_true, y_pred, num_classes=3)
    expected = sk_confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    assert np.array_equal(cm, expected)


def test_accuracy_matches_sklearn(labels):
    y_true, y_pred = labels
    assert accuracy(y_true, y_pred) == pytest.approx(accuracy_score(y_true, y_pred))


def test_macro_f1_matches_sklearn(labels):
    y_true, y_pred = labels
    cm = confusion_matrix(y_true, y_pred, num_classes=3)
    expected = f1_score(y_true, y_pred, average="macro")
    assert macro_f1_from_confusion(cm) == pytest.approx(expected)


def test_cohen_kappa_matches_sklearn(labels):
    y_true, y_pred = labels
    cm = confusion_matrix(y_true, y_pred, num_classes=3)
    assert cohen_kappa_from_confusion(cm) == pytest.approx(cohen_kappa_score(y_true, y_pred))


def test_perfect_prediction_gives_unit_scores():
    y = np.array([0, 1, 1, 0, 1])
    m = compute_metrics(y, y, num_classes=2)
    assert m.accuracy == 1.0
    assert m.macro_f1 == pytest.approx(1.0)
    assert m.kappa == pytest.approx(1.0)


def test_accepts_torch_tensors():
    y_true = torch.tensor([0, 1, 1, 0])
    y_pred = torch.tensor([0, 1, 0, 0])
    m = compute_metrics(y_true, y_pred, num_classes=2)
    assert m.accuracy == pytest.approx(0.75)
    assert isinstance(m, ClassificationMetrics)
    assert m.confusion.shape == (2, 2)
