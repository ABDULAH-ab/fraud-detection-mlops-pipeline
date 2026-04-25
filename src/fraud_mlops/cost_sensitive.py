from __future__ import annotations

import pandas as pd

from fraud_mlops.modeling import build_model, fit_evaluate_model


def compare_standard_vs_cost_sensitive(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    positive_class_weight: float,
) -> dict:
    standard_model = build_model(model_name=model_name, positive_class_weight=1.0)
    standard_metrics = fit_evaluate_model(standard_model, x_train, y_train, x_test, y_test)

    cost_sensitive_model = build_model(
        model_name=model_name,
        positive_class_weight=positive_class_weight,
    )
    cost_sensitive_metrics = fit_evaluate_model(
        cost_sensitive_model,
        x_train,
        y_train,
        x_test,
        y_test,
    )

    business_impact = {
        "standard_false_negatives": standard_metrics.confusion_matrix[1][0],
        "cost_sensitive_false_negatives": cost_sensitive_metrics.confusion_matrix[1][0],
        "standard_false_positives": standard_metrics.confusion_matrix[0][1],
        "cost_sensitive_false_positives": cost_sensitive_metrics.confusion_matrix[0][1],
    }

    return {
        "standard": standard_metrics.to_dict(),
        "cost_sensitive": cost_sensitive_metrics.to_dict(),
        "business_impact": business_impact,
    }
