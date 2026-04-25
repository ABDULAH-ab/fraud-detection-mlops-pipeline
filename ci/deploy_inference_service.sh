#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${1:-}"
NAMESPACE="${2:-fraud-mlops}"
DEPLOYMENT_NAME="fraud-inference-api"
SERVICE_NAME="fraud-api"

if [[ -z "$IMAGE_TAG" ]]; then
  echo "Usage: $0 <image-tag> [namespace]"
  exit 1
fi

if ! kubectl get ns "$NAMESPACE" >/dev/null 2>&1; then
  echo "Namespace $NAMESPACE not found; creating it"
  kubectl create ns "$NAMESPACE"
fi

cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${DEPLOYMENT_NAME}
  namespace: ${NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${DEPLOYMENT_NAME}
  template:
    metadata:
      labels:
        app: ${DEPLOYMENT_NAME}
    spec:
      containers:
      - name: api
        image: ${IMAGE_TAG}
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 512Mi
---
apiVersion: v1
kind: Service
metadata:
  name: ${SERVICE_NAME}
  namespace: ${NAMESPACE}
spec:
  selector:
    app: ${DEPLOYMENT_NAME}
  ports:
  - name: http
    port: 8000
    targetPort: 8000
EOF

kubectl -n "$NAMESPACE" rollout status deployment/"$DEPLOYMENT_NAME" --timeout=180s
echo "Inference API deployed with image: $IMAGE_TAG"
