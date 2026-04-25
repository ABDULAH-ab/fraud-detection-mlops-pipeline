from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(slots=True)
class MetricBundle:
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    pr_auc: float
    false_positive_rate: float
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, float | list[list[int]]]:
        return asdict(self)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> MetricBundle:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    return MetricBundle(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1_score=float(f1_score(y_true, y_pred, zero_division=0)),
        auc_roc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(pr_auc),
        false_positive_rate=float(fp / max((fp + tn), 1)),
        confusion_matrix=cm.astype(int).tolist(),
    )
