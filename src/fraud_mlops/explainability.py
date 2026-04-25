from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_feature_importance_report(
    model: object,
    x_reference: pd.DataFrame,
    output_path: Path,
    top_k: int = 20,
) -> list[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    if importances is None and hasattr(model, "named_steps"):
        rf = model.named_steps.get("rf")
        if rf is not None and hasattr(rf, "feature_importances_"):
            importances = rf.feature_importances_

    if importances is None:
        rows: list[dict] = []
    else:
        rows = [
            {"feature": feature, "importance": float(score)}
            for feature, score in sorted(
                zip(x_reference.columns, importances),
                key=lambda item: item[1],
                reverse=True,
            )[:top_k]
        ]

    pd.DataFrame(rows).to_csv(output_path, index=False)
    return rows


def try_generate_shap_summary(
    model: object,
    x_sample: pd.DataFrame,
    output_path: Path,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import shap

        sample = x_sample.sample(min(500, len(x_sample)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        shap_df = pd.DataFrame(
            {
                "feature": sample.columns,
                "mean_abs_shap": pd.Series(abs(shap_values).mean(axis=0)),
            }
        ).sort_values("mean_abs_shap", ascending=False)

        shap_df.to_csv(output_path, index=False)
        return "shap_generated"
    except Exception:
        pd.DataFrame(columns=["feature", "mean_abs_shap"]).to_csv(output_path, index=False)
        return "shap_unavailable"
