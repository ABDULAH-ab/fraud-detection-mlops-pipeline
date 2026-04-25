from __future__ import annotations

import numpy as np
import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    time_col: str,
    train_fraction: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(time_col)
    split_idx = int(len(ordered) * train_fraction)
    train = ordered.iloc[:split_idx].copy()
    test = ordered.iloc[split_idx:].copy()
    return train, test


def population_stability_index(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
) -> float:
    expected = expected.astype(float)
    actual = actual.astype(float)

    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) < 3:
        return 0.0

    exp_hist, _ = np.histogram(expected, bins=cuts)
    act_hist, _ = np.histogram(actual, bins=cuts)

    exp_pct = np.clip(exp_hist / max(exp_hist.sum(), 1), 1e-6, 1)
    act_pct = np.clip(act_hist / max(act_hist.sum(), 1), 1e-6, 1)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def compute_drift_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    psi_scores: dict[str, float] = {}
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            psi_scores[col] = population_stability_index(train_df[col], test_df[col])

    avg_psi = float(np.mean(list(psi_scores.values()))) if psi_scores else 0.0
    top_shifted = sorted(psi_scores.items(), key=lambda kv: kv[1], reverse=True)[:10]

    return {
        "average_psi": avg_psi,
        "top_shifted_features": [{"feature": k, "psi": v} for k, v in top_shifted],
    }
