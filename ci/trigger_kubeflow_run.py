from __future__ import annotations

import argparse
import os
import sys
import time

from kfp import Client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trigger a Kubeflow pipeline run from CI")
    parser.add_argument("--pipeline", required=True, help="Path to compiled pipeline YAML")
    parser.add_argument("--experiment", default="fraud-detection", help="Experiment name")
    parser.add_argument("--namespace", default="fraud-mlops", help="Kubernetes namespace")
    parser.add_argument("--host", default=os.getenv("KFP_HOST", ""), help="Kubeflow host URL")
    parser.add_argument(
        "--token", default=os.getenv("KFP_TOKEN", ""), help="Kubeflow auth token (optional)"
    )
    parser.add_argument("--run-prefix", default="ci", help="Run name prefix")
    parser.add_argument("--data-uri", default="/mnt/artifacts/train_merged.csv")
    parser.add_argument("--model-registry-uri", default="/tmp/model_registry")
    parser.add_argument(
        "--allow-skip",
        action="store_true",
        help="Exit 0 when KFP host is not configured",
    )
    return parser.parse_args()


def create_client(host: str, token: str) -> Client:
    if token:
        return Client(host=host, existing_token=token)
    return Client(host=host)


def get_or_create_experiment(client: Client, name: str, namespace: str):
    try:
        return client.get_experiment(experiment_name=name, namespace=namespace)
    except Exception:
        return client.create_experiment(name=name, namespace=namespace)


def main() -> int:
    args = parse_args()

    if not args.host:
        message = "KFP_HOST is not configured. Skipping Kubeflow trigger step."
        if args.allow_skip:
            print(message)
            return 0
        print(message)
        return 1

    try:
        client = create_client(args.host, args.token)
        experiment = get_or_create_experiment(client, args.experiment, args.namespace)

        run_name = f"{args.run_prefix}-{int(time.time())}"
        run = client.create_run_from_pipeline_package(
            pipeline_file=args.pipeline,
            arguments={
                "data_uri": args.data_uri,
                "model_registry_uri": args.model_registry_uri,
            },
            run_name=run_name,
            experiment_name=experiment.name,
            namespace=args.namespace,
        )
    except Exception as exc:  # pragma: no cover
        print(f"Failed to trigger Kubeflow run: {exc}")
        return 1

    run_id = getattr(run.run_id, "__str__", lambda: run.run_id)()
    print(f"Triggered Kubeflow run: {run_name}")
    print(f"Run ID: {run_id}")
    print(f"Experiment: {experiment.name}")
    print(f"Namespace: {args.namespace}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
