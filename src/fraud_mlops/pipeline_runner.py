from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fraud_mlops.cost_sensitive import compare_standard_vs_cost_sensitive
from fraud_mlops.data_processing import apply_imbalance_strategy, preprocess_features
from fraud_mlops.drift import compute_drift_report, time_based_split
from fraud_mlops.explainability import (
    generate_feature_importance_report,
    try_generate_shap_summary,
)
from fraud_mlops.modeling import build_model, fit_evaluate_model
from fraud_mlops.retraining import RetrainingPolicy, compare_retraining_strategies


def _cap_rows(frame: pd.DataFrame, label_col: str, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame

    positives = frame[frame[label_col] == 1]
    negatives = frame[frame[label_col] == 0]

    pos_target = max(1, int(max_rows * 0.15))
    neg_target = max_rows - pos_target

    sampled_pos = positives.sample(n=min(len(positives), pos_target), random_state=random_state)
    sampled_neg = negatives.sample(n=min(len(negatives), neg_target), random_state=random_state)

    capped = pd.concat([sampled_pos, sampled_neg], axis=0).sample(frac=1.0, random_state=random_state)
    return capped.reset_index(drop=True)


def run_pipeline(
    data_path: Path,
    output_dir: Path,
    max_rows: int = 100_000,
    fast_mode: bool = False,
    skip_shap: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)

    train_df, test_df = time_based_split(df, time_col="TransactionDT", train_fraction=0.7)
    train_df = _cap_rows(train_df, "isFraud", max_rows=max_rows, random_state=42)
    test_df = _cap_rows(test_df, "isFraud", max_rows=max(10_000, max_rows // 3), random_state=24)

    prepared = preprocess_features(
        train_df=train_df,
        test_df=test_df,
        target_col="isFraud",
        id_cols=["TransactionID", "TransactionDT"],
    )

    imbalance_results: dict[str, dict] = {}
    imbalance_strategies = ["undersample", "class_weight"] if fast_mode else ["smote", "class_weight"]
    for strategy in imbalance_strategies:
        x_res, y_res = apply_imbalance_strategy(prepared.x_train, prepared.y_train, strategy=strategy)
        model = build_model(
            model_name="lightgbm",
            positive_class_weight=prepared.positive_class_weight if strategy == "class_weight" else 1.0,
        )
        metrics = fit_evaluate_model(model, x_res, y_res, prepared.x_test, prepared.y_test)
        imbalance_results[strategy] = metrics.to_dict()

    model_results: dict[str, dict] = {}
    best_model = None
    best_recall = -1.0

    candidate_models = ["xgboost", "lightgbm", "hybrid_rf_fs"]
    if fast_mode:
        candidate_models = ["lightgbm", "hybrid_rf_fs", "xgboost"]

    for name in candidate_models:
        model = build_model(name, prepared.positive_class_weight)
        metrics = fit_evaluate_model(
            model,
            prepared.x_train,
            prepared.y_train,
            prepared.x_test,
            prepared.y_test,
        )
        model_results[name] = metrics.to_dict()
        if metrics.recall > best_recall:
            best_recall = metrics.recall
            best_model = model

    cost_sensitive = compare_standard_vs_cost_sensitive(
        model_name="xgboost",
        x_train=prepared.x_train,
        y_train=prepared.y_train,
        x_test=prepared.x_test,
        y_test=prepared.y_test,
        positive_class_weight=prepared.positive_class_weight,
    )

    drift_report = compute_drift_report(
        train_df=train_df,
        test_df=test_df,
        feature_columns=[
            c for c in train_df.columns if c not in {"isFraud", "TransactionID", "TransactionDT"}
        ],
    )

    feature_importance = generate_feature_importance_report(
        best_model,
        prepared.x_test,
        output_dir / "feature_importance.csv",
    )
    if skip_shap:
        (output_dir / "shap_summary.csv").write_text("feature,mean_abs_shap\n", encoding="utf-8")
        shap_status = "skipped"
    else:
        shap_status = try_generate_shap_summary(best_model, prepared.x_test, output_dir / "shap_summary.csv")

    retraining_history = [
        {"recall": 0.88, "drift": 0.05, "days_since_last_train": 3},
        {"recall": 0.81, "drift": 0.12, "days_since_last_train": 6},
        {"recall": 0.75, "drift": 0.25, "days_since_last_train": 9},
    ]
    retraining_comparison = compare_retraining_strategies(
        retraining_history,
        policy=RetrainingPolicy(),
    )

    result = {
        "imbalance_comparison": imbalance_results,
        "model_comparison": model_results,
        "cost_sensitive": cost_sensitive,
        "drift_report": drift_report,
        "retraining_comparison": retraining_comparison,
        "feature_importance_preview": feature_importance[:10],
        "shap_status": shap_status,
    }

    with (output_dir / "experiment_report.json").open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()
    run_pipeline(
        args.data,
        args.output,
        max_rows=args.max_rows,
        fast_mode=args.fast_mode,
        skip_shap=args.skip_shap,
    )


if __name__ == "__main__":
    main()
