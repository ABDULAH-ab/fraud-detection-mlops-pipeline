# Fraud Detection MLOps Pipeline

An end-to-end, production-oriented MLOps platform for financial transaction fraud detection. Built around the IEEE-CIS Fraud Detection dataset, the system covers the full machine learning lifecycle: automated data preprocessing and class-imbalance handling, model benchmarking, cost-sensitive learning, explainability, PSI-based drift detection, intelligent retraining policies, Kubeflow Pipelines orchestration, FastAPI model serving, Prometheus/Grafana observability, and alert-driven CI/CD.

---

## Architecture Overview

```
Raw / Streaming Data
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Kubeflow Pipeline (7-Step DAG with Retries & Conditional Deploy) │
│  Ingest ─▶ Validate ─▶ Preprocess ─▶ Features ─▶ Train ─▶ Eval    │
│                                                            │      │
│                                      (AUC ≥ 0.75 & Recall ≥ 0.80) │
│                                                            ▼      │
│                                                    Model Registry │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────┐      /metrics      ┌───────────────────────┐
│  FastAPI Inference   │ ─────────────────▶ │  Prometheus & Grafana │
│  (/predict, /health) │                    │  (Performance, Drift, │
└──────────────────────┘                    │   System Health)      │
                                            └──────────┬────────────┘
                                                       │ Alertmanager
                                                       ▼
┌──────────────────────┐  repository_dispatch ┌─────────────────────┐
│  GitHub Actions CI/CD│ ◀─────────────────── │  Alert Bridge       │
│  (Lint, Test, Build, │                      │  (alert_to_github)  │
│   Retrain & Deploy)  │                      └─────────────────────┘
└──────────────────────┘
```

---

## Key Capabilities

### 1. Data Preprocessing & Class-Imbalance Handling
- **Missing Value Imputation**: Multivariate imputation via `IterativeImputer` for numerical features and explicit missing-category handling (`__missing__`) for categorical columns (`src/fraud_mlops/data_processing.py`).
- **Feature Encoding**: Target encoding for high-cardinality categorical features and one-hot encoding for low-cardinality features.
- **Imbalance Strategies**: Built-in benchmarking across `smote`, `undersample`, and `class_weight` strategies.

### 2. Model Benchmarking & Cost-Sensitive Learning
- **Candidate Models** (`src/fraud_mlops/modeling.py`): Evaluates `xgboost`, `lightgbm`, and `hybrid_rf_fs` (Random Forest with `SelectKBest` feature selection).
- **Evaluation Metrics** (`src/fraud_mlops/evaluation.py`): Tracks precision, recall, F1, AUC-ROC, PR-AUC, and confusion matrices.
- **Cost-Sensitive Optimization** (`src/fraud_mlops/cost_sensitive.py`): Compares standard vs. cost-sensitive training with business-impact proxies (false negatives vs. false positives).

### 3. Drift Detection & Intelligent Retraining
- **Time-Based Splitting & PSI** (`src/fraud_mlops/drift.py`): Splits transactions chronologically on `TransactionDT` and computes Population Stability Index (PSI) across features to quantify data drift.
- **Retraining Policies** (`src/fraud_mlops/retraining.py`): Evaluates and compares threshold-based, periodic, and hybrid retraining triggers against recall decay, drift severity, and time since last training.

### 4. Model Explainability
- **Feature Importance & SHAP** (`src/fraud_mlops/explainability.py`): Generates ranked feature importance (`artifacts/feature_importance.csv`) and SHAP summary attributions (`artifacts/shap_summary.csv`).

### 5. Kubeflow Orchestration & Kubernetes
- **7-Step KFP v2 Pipeline** (`kubeflow/pipeline.py`): Implements `data_ingestion`, `data_validation`, `data_preprocessing`, `feature_engineering`, `model_training`, `model_evaluation`, and `conditional_deployment` (deployed only when `auc_roc >= 0.75` and `recall >= 0.80`), with per-step retry policies.
- **Cluster Manifests** (`kubernetes/`): Includes namespace, resource quota, artifact PVC, data-loader pod, and alert-bridge deployment specs.

### 6. Model Serving, Observability & Alert-Driven CI/CD
- **Inference API** (`src/fraud_mlops/inference_api.py`): FastAPI service exposing `/health`, `/predict`, and Prometheus `/metrics` (`fraud_api_latency_seconds`, `fraud_api_requests_total`, `fraud_api_errors_total`).
- **Monitoring & Alerting** (`monitoring/`): Prometheus scrape configs and alert rules (`FraudRecallLow`, `DataDriftHigh`, `FraudApiLatencyHigh`), Alertmanager routing, and pre-built Grafana dashboards (`system_health_dashboard.json`, `model_performance_dashboard.json`, `data_drift_dashboard.json`).
- **Automated CI/CD** (`.github/workflows/mlops-ci-cd.yml`): Runs `ruff` linting, `pytest` unit tests, data validation (`ci/data_validation_check.py`), Docker image builds for training, inference, and alert-bridge services, and supports `repository_dispatch` triggers (`model_performance_drop`, `data_drift_alert`) via `scripts/alert_to_github.py`.

---

## Repository Structure

```text
.
├── .github/workflows/
│   └── mlops-ci-cd.yml              # CI/CD pipeline (lint, test, Docker, Kubeflow triggers)
├── ci/
│   ├── data_validation_check.py     # Schema and dataset integrity checks
│   ├── deploy_inference_service.sh  # Inference deployment helper script
│   └── trigger_kubeflow_run.py      # Programmatic Kubeflow run trigger
├── data/
│   ├── README.md                    # Dataset layout instructions
│   └── train_demo.csv               # Sample dataset for quick local runs
├── docker/
│   ├── alert-bridge.Dockerfile      # Alertmanager-to-GitHub dispatch bridge image
│   ├── inference.Dockerfile         # FastAPI inference service image
│   └── training.Dockerfile          # Training & experiment runner image
├── kubeflow/
│   ├── compile_pipeline.py          # Compiles KFP pipeline to YAML
│   ├── fraud_pipeline.yaml          # Compiled Kubeflow pipeline definition
│   └── pipeline.py                  # 7-step Kubeflow v2 pipeline definition
├── kubernetes/
│   ├── alert-bridge.yaml            # Alert bridge Kubernetes deployment
│   ├── data-loader-pod.yaml         # Data loader pod manifest
│   ├── namespace.yaml               # Dedicated K8s namespace (fraud-mlops)
│   ├── pvc-artifacts.yaml           # PersistentVolumeClaim for pipeline artifacts
│   └── resource-quota.yaml          # Namespace CPU/memory resource quotas
├── monitoring/
│   ├── alert_rules.yml              # Prometheus alert rules (recall, drift, latency)
│   ├── alertmanager.yml             # Alertmanager webhook routing
│   ├── prometheus.yml               # Prometheus scrape configuration
│   └── grafana/
│       ├── data_drift_dashboard.json
│       ├── model_performance_dashboard.json
│       └── system_health_dashboard.json
├── scripts/
│   ├── alert_to_github.py           # Webhook bridge firing GitHub repository_dispatch
│   ├── deploy_fraud_pipeline.py     # Pipeline deployment helper
│   ├── prepare_ieee_train.py        # Merges IEEE-CIS transaction & identity CSVs
│   └── submit_pipeline.py           # Submits compiled pipeline to Kubeflow
├── src/fraud_mlops/
│   ├── config.py                    # Shared configuration defaults
│   ├── cost_sensitive.py            # Standard vs. cost-sensitive evaluation
│   ├── data_processing.py           # Imputation, encoding, imbalance handling
│   ├── drift.py                     # Time-based split & PSI drift calculation
│   ├── evaluation.py                # Classification & PR/ROC evaluation metrics
│   ├── explainability.py            # Feature importance & SHAP reporting
│   ├── inference_api.py             # FastAPI serving app with Prometheus metrics
│   ├── modeling.py                  # XGBoost, LightGBM, and Hybrid RF models
│   ├── monitoring_metrics.py        # Model & drift metric exporter
│   ├── pipeline_runner.py           # End-to-end local experiment CLI
│   └── retraining.py                # Threshold, periodic, and hybrid retraining logic
├── tests/
│   └── test_retraining.py           # Unit tests for retraining policies
├── pyproject.toml                   # Project metadata, Ruff, and Pytest config
└── requirements.txt                 # Python dependencies
```

---

## Getting Started

### Prerequisites
- Python `>=3.10`
- Docker & `kubectl` (optional, for containerized serving and Kubeflow deployment)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset Preparation

The pipeline is designed for the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data) dataset. Place the raw Kaggle CSVs in `data/raw/` (`train_transaction.csv` and `train_identity.csv`) and merge them into a single training file:

```bash
python3 scripts/prepare_ieee_train.py --raw-dir data/raw --output data/train_merged.csv
```

For quick verification without downloading the full Kaggle dataset, a lightweight sample file is available at `data/train_demo.csv`.

---

## Running Locally

### 1. Run the Experiment Pipeline

```bash
PYTHONPATH=src python -m fraud_mlops.pipeline_runner \
  --data data/train_merged.csv \
  --output artifacts
```

Optional flags:
- `--max-rows`: Cap training rows (default: `100000`)
- `--fast-mode`: Use faster imbalance strategies (`undersample`, `class_weight`)
- `--skip-shap`: Skip SHAP computation for faster iteration

Generated artifacts in `artifacts/`:
- `experiment_report.json` — Imbalance comparison, model benchmarks, cost-sensitive results, PSI drift report, and retraining strategy comparison
- `feature_importance.csv` — Ranked model feature importances
- `shap_summary.csv` — Mean absolute SHAP values per feature

### 2. Run the Inference API

```bash
PYTHONPATH=src uvicorn fraud_mlops.inference_api:APP --host 0.0.0.0 --port 8000
```

Available endpoints:
- `GET /health` — Service health check
- `POST /predict` — Fraud prediction and probability (`{"features": {...}}`)
- `GET /metrics` — Prometheus metrics endpoint

### 3. Lint and Test

```bash
ruff check .
pytest -q
```

---

## Kubeflow & Kubernetes Deployment

1. **Provision Kubernetes resources**:

```bash
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/resource-quota.yaml
kubectl apply -f kubernetes/pvc-artifacts.yaml
```

2. **Compile the Kubeflow pipeline**:

```bash
cd kubeflow
python compile_pipeline.py
```

3. **Run the pipeline**:
Upload `kubeflow/fraud_pipeline.yaml` to the Kubeflow Pipelines UI (or submit programmatically via `scripts/submit_pipeline.py`) and provide your `data_uri` and `model_registry_uri` parameters.

---

## Monitoring & Automated Retraining

- **Prometheus & Alertmanager**: Configured via `monitoring/prometheus.yml`, `monitoring/alert_rules.yml`, and `monitoring/alertmanager.yml` to monitor recall drops (`FraudRecallLow`), feature drift (`DataDriftHigh`), and API tail latency (`FraudApiLatencyHigh`).
- **Grafana Dashboards**: Import the JSON templates in `monitoring/grafana/` to visualize system health, model performance, and PSI data drift.
- **Alert-to-GitHub Bridge**: `scripts/alert_to_github.py` receives Alertmanager webhooks and dispatches `repository_dispatch` events (`model_performance_drop` or `data_drift_alert`) to trigger automated retraining in GitHub Actions.
