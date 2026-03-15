import os
import pickle

import numpy as np
import pandas as pd

from src import common as common
DATA_MONITORING_REF_PATH = common.CONFIG['paths']['monitoring']['data_reference']
OUTPUT_DIR = os.path.dirname(__file__)

np.random.seed(42)

def load_reference_data():
    """Load monitoring reference dataset."""
    with open(DATA_MONITORING_REF_PATH, "rb") as f:
        ref = pickle.load(f)

    X_ref = ref["X_ref"].copy()
    y_ref = ref["y_ref_true"].copy()

    return X_ref, y_ref

def generate_feature_drift(X_ref, y_ref):
    """Scenario 1: moderate feature drift."""
    print("Scenario 1: moderate feature drift.")

    X_drift = X_ref.copy()

    numeric_cols = X_drift.select_dtypes(include="number").columns
    nb_cols_drift = min(3, len(numeric_cols))
    selected_cols = np.random.choice(numeric_cols, size=nb_cols_drift, replace=False)

    for col in selected_cols:
        X_drift[col] = X_drift[col] * 1.25 + np.random.normal(0, 0.1, len(X_drift))

    df = pd.concat([X_drift, y_ref.rename("target")], axis=1)

    path = os.path.join(OUTPUT_DIR, "feature_drift.csv")
    df.to_csv(path, index=False)
    print(f"Data saved {path}")

def generate_target_drift(X_ref, y_ref):
    """Scenario 2: target distribution drift."""
    print("Scenario 2: target distribution drift.")

    y_drift = y_ref.copy()
    y_drift = y_drift + np.random.normal(0.5, 0.2, len(y_drift))

    df = pd.concat([X_ref, y_drift.rename("target")], axis=1)

    path = os.path.join(OUTPUT_DIR, "target_drift.csv")
    df.to_csv(path, index=False)
    print(f"Saved {path}")

def generate_concept_drift(X_ref, y_ref):
    """Scenario 3: strong feature + target shift."""
    print("Scenario 3: strong feature + target shift.")

    X_drift = X_ref.copy()
    y_drift = y_ref.copy()

    numeric_cols = X_drift.select_dtypes(include="number").columns
    nb_cols_drift = min(5, len(numeric_cols))
    selected_cols = np.random.choice(numeric_cols, size=nb_cols_drift, replace=False)

    for col in selected_cols:
        X_drift[col] = X_drift[col] * 1.6 + np.random.normal(0, 0.3, len(X_drift))

    y_drift = y_drift + np.random.normal(1.0, 0.5, len(y_drift))

    df = pd.concat([X_drift, y_drift.rename("target")], axis=1)

    path = os.path.join(OUTPUT_DIR, "concept_drift.csv")
    df.to_csv(path, index=False)
    print(f"Saved {path}")

if __name__ == "__main__":

    X_ref, y_ref = load_reference_data()

    generate_feature_drift(X_ref, y_ref)
    generate_target_drift(X_ref, y_ref)
    generate_concept_drift(X_ref, y_ref)

    print("Drift scenarios generated successfully.")
