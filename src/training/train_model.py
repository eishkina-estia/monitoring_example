import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
import mlflow

from src import common as common
from src.mlflow_utils import configure_mlflow

DATA_PROC_PATH = common.CONFIG['paths']['data_processed']

RANDOM_STATE = common.CONFIG['ml']['random_state']

MODEL_NAME = common.CONFIG['mlflow']['model_name']
ARTIFACT_PATH = common.CONFIG['mlflow']['artifact_path']

def train_and_log_model(model_params, X_train, X_test, y_train, y_test):
    """Train the model, log it to MLflow, and evaluate it."""

    model = ElasticNet(random_state=RANDOM_STATE, **model_params)
    model.fit(X_train, y_train)

    # Infer an MLflow model signature from the training data (input),
    # model predictions (output) and parameters (for inference).
    signature = mlflow.models.infer_signature(X_train, y_train)

    # Log model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=ARTIFACT_PATH,
        signature=signature)

    # Log model params
    mlflow.log_params(model_params)

    # Log metrics to MLflow tracking server
    eval_data = pd.concat([X_test,y_test], axis=1)

    results = mlflow.evaluate(
        model_info.model_uri,
        data=eval_data,
        targets=y_test.name,
        model_type="regressor",
        evaluators=["default"]
    )

    return results

def train_model(X_train_processed, X_test_processed, y_train, y_test):
    configure_mlflow()

    params_alpha = [0.01, 0.1, 1, 10]
    params_l1_ratio = np.arange(0.0, 1.1, 0.5)

    num_iterations = len(params_alpha) * len(params_l1_ratio)

    run_name = "elasticnet"
    k = 0
    best_score = float('inf')
    best_run_id = None

    # Test all the defined combinations of hyperparams
    # Log each run
    # Register the best model
    with mlflow.start_run(run_name=run_name, description=run_name):
        for alpha in params_alpha:
            for l1_ratio in params_l1_ratio:
                k += 1
                print(f"\n***** ITERATION {k} from {num_iterations} *****")
                child_run_name = f"{run_name}_{k:02}"
                with mlflow.start_run(run_name=child_run_name, nested=True) as child_run:
                    model_params = {"alpha": alpha, "l1_ratio": l1_ratio}
                    results = train_and_log_model(model_params, X_train_processed, X_test_processed, y_train, y_test)

                    rmse = results.metrics["root_mean_squared_error"]
                    r2 = results.metrics["r2_score"]
                    print(f"rmse: {rmse}")
                    print(f"r2: {r2}")

                    if rmse < best_score:
                        best_score = rmse
                        best_run_id = child_run.info.run_id

    print("#" * 20)
    # Register the best model in the model registry
    model_uri = f"runs:/{best_run_id}/{ARTIFACT_PATH}"
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    print("Model saved to the model registry:")
    print(f"Name: {mv.name}")
    print(f"Version: {mv.version}")
    print(f"Source: {mv.source}")

