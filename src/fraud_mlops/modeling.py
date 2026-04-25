from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from fraud_mlops.evaluation import MetricBundle, compute_classification_metrics


@dataclass(slots=True)
class ModelRunResult:
    model_name: str
    metrics: MetricBundle


def build_model(
    model_name: str,
    positive_class_weight: float,
    random_state: int = 42,
) -> object:
    name = model_name.lower()

    if name == "xgboost":
        return XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.75,
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
            scale_pos_weight=positive_class_weight,
        )

    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=150,
            learning_rate=0.08,
            num_leaves=31,
            min_child_samples=40,
            n_jobs=-1,
            random_state=random_state,
            class_weight={0: 1.0, 1: positive_class_weight},
        )

    if name == "hybrid_rf_fs":
        k = 50
        selector = SelectKBest(score_func=f_classif, k=min(k, 1000))
        rf = RandomForestClassifier(
            n_estimators=120,
            max_depth=None,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=random_state,
            class_weight={0: 1.0, 1: positive_class_weight},
        )
        return Pipeline(steps=[("feature_selection", selector), ("rf", rf)])

    raise ValueError(f"Unsupported model: {model_name}")


def fit_evaluate_model(
    model: object,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> MetricBundle:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)[:, 1]
    else:
        scores = model.decision_function(x_test)
        y_proba = 1.0 / (1.0 + np.exp(-scores))

    return compute_classification_metrics(y_test.to_numpy(), y_pred, y_proba)
