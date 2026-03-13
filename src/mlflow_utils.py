import os
from pathlib import Path
import mlflow
from src import common as common

MLFLOW_DB = common.CONFIG["paths"]["mlflow_db"]
MLFLOW_ARTIFACTS_DIR = common.CONFIG["paths"]["mlruns"]
EXPERIMENT_NAME = common.CONFIG["mlflow"]["experiment_name"]

def configure_mlflow():
    """Configure MLflow tracking with a SQLite backend and a local artifact directory."""

    print("Configure MLflow tracking...")
    os.makedirs(MLFLOW_ARTIFACTS_DIR, exist_ok=True)
    db_dir = os.path.dirname(MLFLOW_DB)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    tracking_uri = f"sqlite:///{MLFLOW_DB}"
    mlflow.set_tracking_uri(tracking_uri)

    # artifact_uri = f"file:///{MLFLOW_ARTIFACTS_DIR}"
    artifact_uri = Path(MLFLOW_ARTIFACTS_DIR).resolve().as_uri()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        mlflow.create_experiment(name=EXPERIMENT_NAME, artifact_location=artifact_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow artifact location: {artifact_uri}")
