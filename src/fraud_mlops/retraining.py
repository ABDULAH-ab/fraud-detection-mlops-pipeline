from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RetrainingPolicy:
    recall_threshold: float = 0.80
    drift_threshold: float = 0.20
    periodic_days: int = 7


def should_retrain(
    latest_recall: float,
    latest_drift: float,
    days_since_last_train: int,
    policy: RetrainingPolicy,
    mode: str = "hybrid",
) -> bool:
    mode = mode.lower()
    threshold_hit = latest_recall < policy.recall_threshold or latest_drift > policy.drift_threshold
    periodic_hit = days_since_last_train >= policy.periodic_days

    if mode == "threshold":
        return threshold_hit
    if mode == "periodic":
        return periodic_hit
    if mode == "hybrid":
        return threshold_hit or periodic_hit

    raise ValueError(f"Unsupported retraining mode: {mode}")


def compare_retraining_strategies(history: list[dict], policy: RetrainingPolicy) -> dict:
    strategies = ["threshold", "periodic", "hybrid"]
    summary: dict[str, dict] = {}

    for strategy in strategies:
        triggers = 0
        for row in history:
            if should_retrain(
                latest_recall=row["recall"],
                latest_drift=row["drift"],
                days_since_last_train=row["days_since_last_train"],
                policy=policy,
                mode=strategy,
            ):
                triggers += 1

        summary[strategy] = {
            "trigger_count": triggers,
            "stability_score": max(0.0, 1.0 - triggers / max(len(history), 1)),
            "estimated_compute_cost": triggers * 1.0,
        }

    return summary
