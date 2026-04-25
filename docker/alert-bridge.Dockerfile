FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/alert_to_github.py ./alert_to_github.py

EXPOSE 8080
CMD ["uvicorn", "alert_to_github:APP", "--host", "0.0.0.0", "--port", "8080"]
