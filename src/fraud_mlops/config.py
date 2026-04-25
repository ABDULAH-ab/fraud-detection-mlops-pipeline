from dataclasses import dataclass


@dataclass(slots=True)
class TrainingConfig:
    target_column: str = "isFraud"
    id_column: str = "TransactionID"
    time_column: str = "TransactionDT"
    random_state: int = 42
    test_size: float = 0.2
    recall_alert_threshold: float = 0.80
    drift_alert_threshold: float = 0.20
