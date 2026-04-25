from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


@dataclass(slots=True)
class PreparedDataset:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    positive_class_weight: float


def _sanitize_feature_names(columns: list[str]) -> list[str]:
    sanitized: list[str] = []
    used: dict[str, int] = {}

    for col in columns:
        name = re.sub(r"[^0-9a-zA-Z_]", "_", str(col))
        name = re.sub(r"_+", "_", name).strip("_")
        if not name:
            name = "feature"
        if name[0].isdigit():
            name = f"f_{name}"

        if name in used:
            used[name] += 1
            name = f"{name}_{used[name]}"
        else:
            used[name] = 0

        sanitized.append(name)

    return sanitized


def _target_encode(
    train_series: pd.Series,
    test_series: pd.Series,
    y_train: pd.Series,
    smoothing: float = 20.0,
) -> tuple[pd.Series, pd.Series]:
    df = pd.DataFrame({"feature": train_series, "target": y_train})
    global_mean = y_train.mean()
    stats = df.groupby("feature")["target"].agg(["mean", "count"])
    smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
        stats["count"] + smoothing
    )

    train_encoded = train_series.map(smooth).fillna(global_mean)
    test_encoded = test_series.map(smooth).fillna(global_mean)
    return train_encoded, test_encoded


def preprocess_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    id_cols: list[str] | None = None,
    high_cardinality_threshold: int = 30,
) -> PreparedDataset:
    if id_cols is None:
        id_cols = []

    y_train = train_df[target_col].astype(int)
    y_test = test_df[target_col].astype(int)

    x_train = train_df.drop(columns=[target_col, *id_cols], errors="ignore").copy()
    x_test = test_df.drop(columns=[target_col, *id_cols], errors="ignore").copy()

    num_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [col for col in x_train.columns if col not in num_cols]

    if num_cols:
        imputer = IterativeImputer(max_iter=8, random_state=42)
        x_train[num_cols] = imputer.fit_transform(x_train[num_cols])
        x_test[num_cols] = imputer.transform(x_test[num_cols])

    for col in cat_cols:
        x_train[col] = x_train[col].astype(str).fillna("__missing__")
        x_test[col] = x_test[col].astype(str).fillna("__missing__")

    low_card_cols: list[str] = []
    high_card_cols: list[str] = []
    for col in cat_cols:
        if x_train[col].nunique() > high_cardinality_threshold:
            high_card_cols.append(col)
        else:
            low_card_cols.append(col)

    for col in high_card_cols:
        tr, te = _target_encode(x_train[col], x_test[col], y_train)
        x_train[col] = tr
        x_test[col] = te

    if low_card_cols:
        x_train = pd.get_dummies(x_train, columns=low_card_cols, dummy_na=False)
        x_test = pd.get_dummies(x_test, columns=low_card_cols, dummy_na=False)
        x_train, x_test = x_train.align(x_test, join="left", axis=1, fill_value=0)

    # Guard against infinities/overflow before tree models (notably XGBoost) are trained.
    x_train = x_train.replace([np.inf, -np.inf], np.nan)
    x_test = x_test.replace([np.inf, -np.inf], np.nan)

    medians = x_train.median(numeric_only=True)
    x_train = x_train.fillna(medians).fillna(0)
    x_test = x_test.fillna(medians).fillna(0)

    finite_limit = np.finfo(np.float32).max
    x_train = x_train.clip(lower=-finite_limit, upper=finite_limit).astype(np.float32)
    x_test = x_test.clip(lower=-finite_limit, upper=finite_limit).astype(np.float32)

    safe_columns = _sanitize_feature_names([str(c) for c in x_train.columns])
    x_train.columns = safe_columns
    x_test.columns = safe_columns

    class_counts = y_train.value_counts()
    pos_weight = float(class_counts.get(0, 1) / max(class_counts.get(1, 1), 1))

    return PreparedDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        positive_class_weight=pos_weight,
    )


def apply_imbalance_strategy(
    x: pd.DataFrame,
    y: pd.Series,
    strategy: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    strategy = strategy.lower()
    if strategy == "smote":
        sampler = SMOTE(random_state=random_state)
        x_res, y_res = sampler.fit_resample(x, y)
        return pd.DataFrame(x_res, columns=x.columns), pd.Series(y_res)

    if strategy == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)
        x_res, y_res = sampler.fit_resample(x, y)
        return pd.DataFrame(x_res, columns=x.columns), pd.Series(y_res)

    if strategy in {"class_weight", "none"}:
        return x, y

    raise ValueError(f"Unknown imbalance strategy: {strategy}")
