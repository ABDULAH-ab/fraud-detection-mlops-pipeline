# Fraud Detection MLOps Pipeline (Assignment 4)

This repository is a full starter implementation for the Assignment 4 requirements:

- Kubeflow pipeline with conditional deployment and retries
- Data challenge handling (missing values, high-cardinality encoding, imbalance strategies)
- Model comparison (XGBoost, LightGBM, hybrid RF + feature selection)
- Cost-sensitive learning comparison
- CI/CD pipeline with intelligent monitoring-triggered retraining hooks
- Observability stack (Prometheus + Grafana + alert rules)
- Drift simulation (time-based split + PSI)
- Intelligent retraining strategy comparison
- Explainability (feature importance + SHAP fallback)

## 1. Project Structure

```
.
├── .github/workflows/mlops-ci-cd.yml
├── ci/data_validation_check.py
├── docker/
│   ├── inference.Dockerfile
│   └── training.Dockerfile
├── kubeflow/
│   ├── compile_pipeline.py
│   └── pipeline.py
├── kubernetes/
│   ├── namespace.yaml
│   ├── pvc-artifacts.yaml
│   └── resource-quota.yaml
├── monitoring/
│   ├── alert_rules.yml
│   ├── alertmanager.yml
│   ├── prometheus.yml
│   └── grafana/
│       ├── data_drift_dashboard.json
│       ├── model_performance_dashboard.json
│       └── system_health_dashboard.json
├── scripts/alert_to_github.py
├── src/fraud_mlops/
│   ├── cost_sensitive.py
│   ├── data_processing.py
│   ├── drift.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── inference_api.py
│   ├── modeling.py
│   ├── monitoring_metrics.py
│   ├── pipeline_runner.py
│   └── retraining.py
└── tests/test_retraining.py
```

## 2. Task Mapping

### Task 1: Kubeflow Environment Setup

- Namespace, resource quota, PVC:
	- `kubernetes/namespace.yaml`
	- `kubernetes/resource-quota.yaml`
	- `kubernetes/pvc-artifacts.yaml`
- Kubeflow pipeline with 7 steps and retries:
	- `kubeflow/pipeline.py`
- Pipeline compile script:
	- `kubeflow/compile_pipeline.py`

### Task 2: Data Challenges Handling

- Advanced missing values with `IterativeImputer`
- High-cardinality handling with target encoding
- Low-cardinality one-hot encoding
- Imbalance strategies: `SMOTE`, `undersample`, `class_weight`
- Comparison ready in experiment runner
	- `src/fraud_mlops/data_processing.py`
	- `src/fraud_mlops/pipeline_runner.py`

### Task 3: Model Complexity

- Models:
	- `xgboost`
	- `lightgbm`
	- `hybrid_rf_fs` (RF + SelectKBest)
- Metrics:
	- precision, recall, f1, AUC-ROC, confusion matrix, PR-AUC
	- `src/fraud_mlops/modeling.py`
	- `src/fraud_mlops/evaluation.py`

### Task 4: Cost-Sensitive Learning

- Standard vs cost-sensitive comparison:
	- `src/fraud_mlops/cost_sensitive.py`
- Includes business-impact proxies:
	- false negatives and false positives comparison

### Task 5: CI/CD with Intelligent Triggers

- Workflow:
	- `push`, `pull_request`, `repository_dispatch` triggers
	- lint, tests, data validation checks
	- docker build/push for training + inference images
	- compile/deploy Kubeflow step placeholders
	- `/.github/workflows/mlops-ci-cd.yml`
- Monitoring-triggered dispatch bridge:
	- `scripts/alert_to_github.py`

### Task 6: Observability & Monitoring

- Prometheus scrape + rule configs:
	- `monitoring/prometheus.yml`
	- `monitoring/alert_rules.yml`
- Alertmanager integration:
	- `monitoring/alertmanager.yml`
- Grafana dashboard templates:
	- `monitoring/grafana/system_health_dashboard.json`
	- `monitoring/grafana/model_performance_dashboard.json`
	- `monitoring/grafana/data_drift_dashboard.json`
- Model/data metric exporter:
	- `src/fraud_mlops/monitoring_metrics.py`

### Task 7: Drift Simulation

- Time-based split + PSI drift report:
	- `src/fraud_mlops/drift.py`

### Task 8: Intelligent Retraining

- Threshold / periodic / hybrid trigger logic + comparison:
	- `src/fraud_mlops/retraining.py`

### Task 9: Explainability

- Feature importance report
- SHAP summary (if available)
	- `src/fraud_mlops/explainability.py`

## 3. Local Setup

1. Create and activate virtual environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 4. Dataset Setup (IEEE-CIS Fraud Detection)

Competition source:

- https://www.kaggle.com/competitions/ieee-fraud-detection/data

Prepare data locally:

```bash
./scripts/download_ieee_data.sh
python3 scripts/prepare_ieee_train.py --raw-dir data/raw --output data/train_merged.csv
```

If you cannot download from Kaggle, confirm that:

1. You accepted the competition rules on Kaggle.
2. Your API token is configured at `~/.kaggle/kaggle.json`.

3. Run syntax checks/tests:

```bash
ruff check .
pytest -q
```

## 5. Run Experiment Pipeline (Local)

Use IEEE-CIS data CSV (for example merged train file):

```bash
PYTHONPATH=src python -m fraud_mlops.pipeline_runner \
	--data data/train_merged.csv \
	--output artifacts
```

Key output artifacts:

- `artifacts/experiment_report.json`
- `artifacts/feature_importance.csv`
- `artifacts/shap_summary.csv`

## 6. Kubeflow Usage

1. Apply Kubernetes setup:

```bash
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/resource-quota.yaml
kubectl apply -f kubernetes/pvc-artifacts.yaml
```

2. Compile pipeline:

```bash
cd kubeflow
python compile_pipeline.py
```

3. Upload `kubeflow/fraud_pipeline.yaml` to Kubeflow Pipelines UI and run with your data URI.

## 7. Evidence to Capture for Submission

- CI/CD workflow run logs from GitHub Actions
- Kubeflow run screenshots/logs
- Prometheus alerts + Grafana dashboard screenshots
- `experiment_report.json` comparison outputs
- Explainability artifacts (`feature_importance.csv`, SHAP output)
- Research report PDF

## 8. Notes

- This scaffold is intentionally production-oriented but still assignment-friendly.
- You should tune thresholds and hyperparameters based on your final dataset experiments.
- Replace placeholder deployment commands in CI with your actual cluster/Kubeflow endpoint commands.