from fraud_mlops.retraining import RetrainingPolicy, should_retrain


def test_hybrid_policy_triggers_on_low_recall() -> None:
    policy = RetrainingPolicy(recall_threshold=0.8, drift_threshold=0.2, periodic_days=7)
    assert should_retrain(0.7, 0.1, 1, policy, mode="hybrid")


def test_periodic_policy_waits_before_threshold() -> None:
    policy = RetrainingPolicy(recall_threshold=0.8, drift_threshold=0.2, periodic_days=7)
    assert not should_retrain(0.9, 0.1, 3, policy, mode="periodic")
