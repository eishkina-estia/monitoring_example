import os
import pickle
import pandas as pd

from src import common as common
DATA_MONITORING_REF_PATH = common.CONFIG['paths']['monitoring']['data_reference']

from src.training.training_utils import load_latest_model, load_preprocessor

def save_monitoring_reference_data(X_raw, y):
    """
    Save reference data for drift monitoring.

    The following objects are stored in DATA_MONITORING_PATH as a pickle file:
    (X_train_raw, y_train, y_pred)
    """

    preprocessor = load_preprocessor()
    model, model_signature = load_latest_model()

    X_processed = preprocessor.transform(X_raw)
    signature_columns = [col.name for col in model_signature.inputs]
    X_processed = pd.DataFrame(X_processed, columns=signature_columns)
    y_pred = model.predict(X_processed)
    y_pred = pd.Series(y_pred, name="prediction")

    reference_data = {
        "X_ref": X_raw.reset_index(drop=True),
        "y_ref_true": y.reset_index(drop=True),
        "y_ref_pred": y_pred
    }

    monitoring_dir = os.path.dirname(DATA_MONITORING_REF_PATH)
    os.makedirs(monitoring_dir, exist_ok=True)

    with open(DATA_MONITORING_REF_PATH, "wb") as file:
        pickle.dump(reference_data, file)

    print(f"Reference monitoring data saved to {DATA_MONITORING_REF_PATH}")