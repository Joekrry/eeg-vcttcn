"""Classification metrics computed from a confusion matrix.

Implemented directly with numpy (rather than calling scikit-learn) so the unit
tests can verify the formulas against scikit-learn as an independent reference.
All functions accept either numpy arrays or torch tensors of integer labels.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor, list]


def _to_int_array(a: ArrayLike) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    return np.asarray(a).astype(np.int64).ravel()


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, num_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    yt = _to_int_array(y_true)
    yp = _to_int_array(y_pred)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (yt, yp), 1)
    return cm


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt = _to_int_array(y_true)
    yp = _to_int_array(y_pred)
    if yt.size == 0:
        return 0.0
    return float((yt == yp).mean())


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """Unweighted mean of per-class F1 scores (F1 is 0 for empty classes)."""
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    actual = cm.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def cohen_kappa_from_confusion(cm: np.ndarray) -> float:
    """Cohen's kappa: agreement corrected for chance."""
    n = cm.sum()
    if n == 0:
        return 0.0
    observed = np.trace(cm) / n
    expected = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / (n * n)
    if abs(1.0 - expected) < 1e-12:
        return 0.0
    return float((observed - expected) / (1.0 - expected))


@dataclass
class ClassificationMetrics:
    """Bundle of the metrics tracked each epoch."""

    accuracy: float
    macro_f1: float
    kappa: float
    confusion: np.ndarray


def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike, num_classes: int) -> ClassificationMetrics:
    """Compute accuracy, macro-F1, Cohen's kappa, and the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    return ClassificationMetrics(
        accuracy=accuracy(y_true, y_pred),
        macro_f1=macro_f1_from_confusion(cm),
        kappa=cohen_kappa_from_confusion(cm),
        confusion=cm,
    )
