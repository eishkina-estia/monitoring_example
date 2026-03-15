# ML Monitoring

This project demonstrates a complete ML workflow with MLflow and monitoring tools.

The dataset used in this project comes from [ics.uci.edu](http://archive.ics.uci.edu/ml/datasets/Wine+Quality)

## Project Structure
```
monitoring_example/
│
├── config.yml                 # Project configuration (paths, MLflow settings)
├── requirements.txt           # Python dependencies
│
├── data/
│   ├── raw/
│   │   └── wine-quality.csv   # Raw dataset
│   └── processed/
│       └── processed.pkl      # Preprocessed dataset (generated)
│
├── mlflow.db                  # MLflow SQLite backend (generated)
├── mlruns/                    # MLflow artifacts (generated)
│
├── src/
│   ├── common.py              # Loads config.yml and resolves paths
│   ├── mlflow_utils.py        # MLflow configuration
│   │
│   ├── training/
│   │   ├── preprocess_data.py # Data preprocessing
│   │   ├── train_model.py     # Model training and experiment tracking
│   │   ├── test_model_load.py # Load latest model and run predictions on a sample from test dataset
│   │   └── run_training_pipeline.py # Runs the full training pipeline
│   │
│   ├── serving/
│   │   ├── api.py             # FastAPI application
│   │   ├── inference.py       # Model loading + prediction logic
│   │   └── schemas.py         # API request/response schemas
│   │
│   └── monitoring/
│       └── prometheus
│           ├── metrics.py     # Prometheus metrics definitions
│           └── prometheus.yml # Prometheus configuration
│
└── ...
```

## Configuration

Project configuration (paths, MLflow settings) is defined in `config.yml`.
The configuration is automatically loaded by `src/common.py`.

## Training Workflow

The training workflow consists of the following steps:

1. **Load raw data**
   - read the raw CSV dataset
   - split data into train and test sets

2. **Preprocess data**
   - check missing values
   - apply feature scaling with `StandardScaler`
   - save preprocessing artifacts to `data/processed/processed.pkl`

3. **Train model**
   - run a small hyperparameter search with ElasticNet
   - log parameters, metrics, and model artifacts to MLflow
   - register the best model in the MLflow model registry

4. **Save monitoring reference data**
   - save the raw training reference dataset to `data/monitoring/reference.pkl`
   - this file is intended to be used later by Evidently for drift detection

5. **Load and test the latest model**
   - load the latest registered model
   - run predictions on a small test sample

The pipeline entry point is:
```
src/training/run_training_pipeline.py
```

Training should be executed inside Docker so that the same environment is used for both training and serving.
```shell
docker compose --profile training run --rm train
```

Artifacts are stored persistently on the host:
```
data/processed/
mlflow.db
mlruns/
```

Example output:
```
      fixed acidity  volatile acidity  citric acid  residual sugar  ...  sulphates    alcohol  target-true  target-pred
4656            6.0              0.29         0.41            10.8  ...       0.59  10.966667            7     6.306061
3659            5.4              0.53         0.16             2.7  ...       0.53  13.200000            8     6.590940
907             7.1              0.25         0.39             2.1  ...       0.43  12.200000            8     6.364277
4352            7.3              0.28         0.35             1.6  ...       0.47  10.700000            5     5.751044
3271            6.5              0.32         0.34             5.7  ...       0.60  12.000000            7     6.415782

[5 rows x 13 columns]
```

You can inspect experiments and registered models using the MLflow UI:
```shell
$ mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open the UI:
```
http://localhost:5000
 ```

## Model Serving with FastAPI

After training and registering the model in MLflow, the model can be exposed as a prediction service using FastAPI.

The API:
* loads the latest registered model, 
* loads the preprocessing pipeline,
* exposes prediction endpoint,
* exposes monitoring endpoints.

Running the Serving + Monitoring stack
```shell
docker compose up api prometheus
```

This launches:
* FastAPI inference service
* Prometheus monitoring service

Open the interactive API documentation:
```
http://localhost:8000/docs
```

### Available Endpoints

#### Root
```
GET /
```
Returns a simple service description.

#### Health check
```
GET /health
```

Example response:
```
{
  "status": "ok",
  "model_name": "wine_quality_model",
  "model_version": "1"
}
```

#### Prediction

```
POST /predict
```

Example request:
```
{
  "fixed acidity": 7.4,
  "volatile acidity": 0.7,
  "citric acid": 0.0,
  "residual sugar": 1.9,
  "chlorides": 0.076,
  "free sulfur dioxide": 11.0,
  "total sulfur dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

Example response:
```
{
  "predicted_quality": 6.12,
  "model_name": "wine_quality_model",
  "model_version": "1"
}
```

## Monitoring with Prometheus

[Prometheus](https://prometheus.io/) metrics are exposed by the FastAPI application.

Metrics include:
* prediction_requests_total: number of inference requests
* prediction_request_errors_total: number of inference errors
* prediction_latency_seconds: request latency
* model_info: service health metrics

Metrics endpoint:
```
GET /metrics
```

Example:
```
http://localhost:8000/metrics
```

Open the Prometheus dashboard:
```
http://localhost:9090
```

## Grafana Dashboards

[Grafana](https://grafana.com/) is used to visualize the metrics collected by Prometheus and to provide dashboards for monitoring the application and the infrastructure.

Grafana will be available at:
```
http://localhost:3000
```
Default credentials:
```
username: admin
password: admin
```

## Monitoring with Evidently

This project includes a simple data drift monitoring pipeline using [Evidently](https://evidentlyai.com/).

The monitoring workflow compares new data batches with a reference dataset generated during training.

### Reference dataset

During training, a reference dataset is saved:
```
data/monitoring/reference.pkl
```

The reference file contains:
```
X_ref        # feature values used for training
y_ref_true   # true target values
y_ref_pred   # predictions produced by the trained model
```
This dataset represents the baseline distribution used to detect drift in production data.

### Drift scenarios

Synthetic drift datasets are generated for testing monitoring:
```
data/monitoring/drift_scenarios/
 ├─ feature_drift.csv   # distribution shift in several features
 ├─ target_drift.csv    # change in target distribution 
 └─ concept_drift.csv   # strong feature shift and degraded target relationship 
```
These files are generated locally:
```shell
python -m data.monitoring.drift_scenarios.generate_drift_scenarios
```

### Generate drift reports

Run the Evidently monitoring script:
```shell
python -m src.monitoring.evidently.generate_report
```
or inside Docker:
```shell
docker compose --profile monitoring run --rm evidently-report
```

This will:
* Load the reference dataset
* Load each drift scenario CSV
* Run Evidently drift analysis
* Generate HTML reports

Reports are saved to:
```
./reports/evidently/
```

Each report includes:
* Dataset drift detection
* Feature-level drift statistics
* Target drift
* Summary statistics

The monitoring report uses [DataDriftPreset](https://docs.evidentlyai.com/metrics/preset_data_drift) and [DataSummaryPreset](https://docs.evidentlyai.com/metrics/preset_data_summary) from Evidently.

## Workflow Summary

Typical workflow:

1. Train the model
   ```shell
   docker compose --profile training run --rm train
   ```
   This step:
   * trains the ML model
   * registers it in MLflow
   * generates the monitoring reference dataset
   
   Generated artifacts:
   ```
   data/processed/processed.pkl
   data/monitoring/reference.pkl
   # mlflow model artifacts
   ```
2. Start serving and runtime monitoring   
   Start the application stack:
   * FastAPI for model serving
   * Prometheus for metrics collection
   * Grafana for monitoring dashboards
   ```shell
   docker compose up
   ```

   Access services
   * **FastAPI API**   
     Interactive API documentation:
     ```
     http://localhost:8000/docs
     ```
   * **Prometheus**   
     Metrics and scraping configuration:
     ```
     http://localhost:9090
     ```
   * **Grafana**   
     Monitoring dashboards:
     ```
     http://localhost:3000
     ```
   Default credentials (if unchanged):
   ```
   admin / admin
   ```

3. Generate offline drift reports

   Drift reports are generated offline using Evidently.   
   ```shell
   docker compose --profile monitoring run --rm evidently-report
   ```

   This step compares new data with the reference dataset generated during training.
   For demonstration purposes, the project currently uses pre-generated drift scenario datasets.
   
   Reports are saved to:
   ```
   ./reports/evidently/
   ```