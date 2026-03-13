import pickle
import mlflow
from src import common as common
from src.mlflow_utils import configure_mlflow

DATA_PROC_PATH = common.CONFIG['paths']['data_processed']
MODEL_NAME = common.CONFIG['mlflow']['model_name']
N_SAMPLES = 5

def load_data_sample():
    """Load a small sample from the preprocessed test set."""
    print(f"Load preprocessed test data: {N_SAMPLES} samples")
    with open(DATA_PROC_PATH, "rb") as file:
        _, X_test, _, y_test, _ = pickle.load(file)
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    X_test_sample = X_test.iloc[:N_SAMPLES]
    y_test_sample = y_test.iloc[:N_SAMPLES]
    return X_test_sample, y_test_sample

def load_latest_model(model_name):
    """Load the latest registered version of a model from the MLflow model registry."""

    client = mlflow.MlflowClient()
    model_versions = client.get_latest_versions(model_name, stages=["None"])

    if not model_versions:
        raise ValueError(f"No registered versions found for model '{model_name}'.")

    model_uri = f"models:/{model_name}/{model_versions[0].version}"

    print("Load model from the model registry")
    print(f"Model URI: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    return model

def test_model_load():
    """Load the latest registered model and run predictions on a few test samples."""
    configure_mlflow()
    model = load_latest_model(MODEL_NAME)

    X_test_sample, y_test_sample = load_data_sample()
    y_pred = model.predict(X_test_sample)

    # Display results
    prediction_preview = X_test_sample
    prediction_preview["target_true"] = y_test_sample
    prediction_preview["target_pred"] = y_pred

    print("Prediction preview:")
    print(prediction_preview)

if __name__ == "__main__":
    test_model_load()

