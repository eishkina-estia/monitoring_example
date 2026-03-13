import os
import pickle
import mlflow

from src import common as common
from src.mlflow_utils import configure_mlflow

DATA_PROC_PATH = common.CONFIG['paths']['data_processed']
MODEL_NAME = common.CONFIG['mlflow']['model_name']

def load_preprocessor():
    """Load the fitted preprocessor from the preprocessing artifacts.

    Returns:
        fitted preprocessing object
    """
    print("Load preprocessor")

    with open(DATA_PROC_PATH, "rb") as file:
        _, _, _, _, preprocessor = pickle.load(file)

    return preprocessor

def load_latest_model():
    """Load the latest registered version of a model from the MLflow model registry."""

    configure_mlflow()
    client = mlflow.MlflowClient()
    model_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])

    if not model_versions:
        raise ValueError(f"No registered versions found for model '{MODEL_NAME}'.")

    model_uri = f"models:/{MODEL_NAME}/{model_versions[0].version}"

    print("Load model from the model registry")
    print(f"Model URI: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri=model_uri)

    signature = model.metadata.signature
    if signature is None or signature.inputs is None:
        raise ValueError("The loaded model does not contain an input signature.")

    print("Model input signature:")
    print(signature.inputs)

    return model, signature

def load_test_data_sample(n_samples):
    """Load a small sample from the preprocessed test set."""
    print(f"Load preprocessed test data: {n_samples} samples")
    with open(DATA_PROC_PATH, "rb") as file:
        _, X_test, _, y_test, _ = pickle.load(file)
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    X_test_sample = X_test.iloc[:n_samples]
    y_test_sample = y_test.iloc[:n_samples]
    return X_test_sample, y_test_sample