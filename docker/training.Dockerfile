FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY ci ./ci
ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "fraud_mlops.pipeline_runner"]
