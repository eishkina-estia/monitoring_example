import pickle

import mlflow
import pandas as pd

from src import common as common
from src.mlflow_utils import configure_mlflow

DATA_PROC_PATH = common.CONFIG["paths"]["data_processed"]
MODEL_NAME = common.CONFIG["mlflow"]["model_name"]

# ---------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------

def load_preprocessor():
    """Load the fitted preprocessor from the preprocessing artifacts.

    Returns:
        fitted preprocessing object
    """
    print("Load preprocessor")

    with open(DATA_PROC_PATH, "rb") as file:
        _, _, _, _, preprocessor = pickle.load(file)

    return preprocessor


def get_latest_model_version(model_name):
    """Get the latest registered model version from the MLflow model registry.

    Args:
        model_name: registered model name

    Returns:
        latest model version as string
    """
    print("Get latest model version")

    client = mlflow.MlflowClient()
    model_versions = client.get_latest_versions(model_name, stages=["None"])

    if not model_versions:
        raise ValueError(f"No registered versions found for model '{model_name}'.")

    latest_model_version = str(model_versions[0].version)
    print(f"Latest model version: {latest_model_version}")

    return latest_model_version


def load_model(model_name, model_version):
    """Load a registered model from MLflow and extract its input signature.

    Args:
        model_name: registered model name
        model_version: registered model version

    Returns:
        loaded MLflow pyfunc model
        MLflow model signature
    """
    print("Load model from the model registry")

    model_uri = f"models:/{model_name}/{model_version}"
    print(f"Model URI: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri=model_uri)

    signature = model.metadata.signature
    if signature is None or signature.inputs is None:
        raise ValueError("The loaded model does not contain an input signature.")

    print("Model input signature:")
    print(signature.inputs)

    return model, signature



def load_inference_artifacts():
    """Load all artifacts needed by the inference API.

    Returns:
        dict with:
        - model
        - preprocessor
        - signature
        - model_name
        - model_version
    """
    print("Step 1 - Configure MLflow")
    configure_mlflow()

    print("Step 2 - Load latest model")
    model_version = get_latest_model_version(MODEL_NAME)
    model, signature = load_model(MODEL_NAME, model_version)

    print("Step 3 - Load preprocessor")
    preprocessor = load_preprocessor()

    print("Inference artifacts loaded")

    return {
        "model": model,
        "preprocessor": preprocessor,
        "model_signature": signature,
        "model_name": MODEL_NAME,
        "model_version": model_version,
    }


# ---------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------

def build_input_dataframe(payload, model_signature):
    """Convert validated request data into a one-row pandas DataFrame.

    The output DataFrame matches the input schema expected by the model
    signature (before preprocessing).

    Args:
        payload: validated WineFeatures object
        model_signature: MLflow model signature

    Returns:
        one-row DataFrame with raw input features ordered according to the model signature
    """
    if model_signature is None or model_signature.inputs is None:
        raise ValueError("The model signature does not contain input columns.")

    # Extract ordered feature names from the signature
    signature_columns = [col.name for col in model_signature.inputs]

    # Convert payload to dict using aliases (feature names used during training)
    data = payload.model_dump(by_alias=True)

    # Build dataframe with correct column order
    x_raw = pd.DataFrame([data], columns=signature_columns)

    return x_raw


def predict_quality(payload, preprocessor, model, model_signature):
    """Apply preprocessing and run model inference.

    Args:
        payload: validated WineFeatures object
        preprocessor: fitted preprocessing object
        model: loaded MLflow model
        model_signature: MLflow model input signature

    Returns:
        predicted quality as float
    """
    print("Run prediction")

    # Build raw model input from request payload
    x_raw = build_input_dataframe(payload, model_signature)

    # Apply preprocessing and rebuild a dataframe with a correct signature
    x_processed = preprocessor.transform(x_raw)
    signature_columns = [col.name for col in model_signature.inputs]
    x_processed = pd.DataFrame(x_processed, columns=signature_columns)

    # Run model inference
    y_pred = model.predict(x_processed)

    return float(y_pred[0])