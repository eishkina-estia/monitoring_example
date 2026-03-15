FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY config.yml ./config.yml

# Data used by training
COPY data/raw ./data/raw

# Data used by offline monitoring
COPY data/monitoring/drift_scenarios ./data/monitoring/drift_scenarios