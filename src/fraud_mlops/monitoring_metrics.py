from __future__ import annotations

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


def push_model_and_data_metrics(
    pushgateway_url: str,
    job_name: str,
    recall: float,
    precision: float,
    false_positive_rate: float,
    drift_score: float,
    missing_ratio: float,
    anomaly_rate: float,
    detection_rate: float,
    confidence_mean: float,
) -> None:
    registry = CollectorRegistry()

    gauges = {
        "fraud_model_recall": Gauge("fraud_model_recall", "Fraud recall", registry=registry),
        "fraud_model_precision": Gauge("fraud_model_precision", "Fraud precision", registry=registry),
        "fraud_model_false_positive_rate": Gauge(
            "fraud_model_false_positive_rate", "Fraud false positive rate", registry=registry
        ),
        "fraud_data_drift_score": Gauge(
            "fraud_data_drift_score", "Average data drift (PSI)", registry=registry
        ),
        "fraud_missing_value_ratio": Gauge(
            "fraud_missing_value_ratio", "Missing value ratio", registry=registry
        ),
        "fraud_input_anomaly_rate": Gauge(
            "fraud_input_anomaly_rate", "Input anomaly rate", registry=registry
        ),
        "fraud_detection_rate": Gauge(
            "fraud_detection_rate", "Predicted fraud rate", registry=registry
        ),
        "fraud_prediction_confidence_mean": Gauge(
            "fraud_prediction_confidence_mean", "Average prediction confidence", registry=registry
        ),
    }

    gauges["fraud_model_recall"].set(recall)
    gauges["fraud_model_precision"].set(precision)
    gauges["fraud_model_false_positive_rate"].set(false_positive_rate)
    gauges["fraud_data_drift_score"].set(drift_score)
    gauges["fraud_missing_value_ratio"].set(missing_ratio)
    gauges["fraud_input_anomaly_rate"].set(anomaly_rate)
    gauges["fraud_detection_rate"].set(detection_rate)
    gauges["fraud_prediction_confidence_mean"].set(confidence_mean)

    push_to_gateway(pushgateway_url, job=job_name, registry=registry)
