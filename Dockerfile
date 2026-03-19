#FROM python:3.10-slim AS base
FROM python:3.10-slim-bookworm AS base

WORKDIR /app

COPY requirements ./requirements

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements/base.txt

COPY src ./src
COPY config.yml ./config.yml

# Data used by training
COPY data/raw ./data/raw

# Data used by offline monitoring
COPY data/monitoring/drift_scenarios ./data/monitoring/drift_scenarios


# Training image
FROM base AS training
RUN pip install --no-cache-dir -r requirements/training.txt

# Serving image
FROM base AS serving
RUN pip install --no-cache-dir -r requirements/serving.txt

# Offline monitoring image
FROM base AS monitoring
RUN pip install --no-cache-dir -r requirements/monitoring_offline.txt
