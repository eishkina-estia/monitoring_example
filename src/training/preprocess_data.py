import os
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import common as common

DATA_PROC_PATH = common.CONFIG['paths']['data_processed']

def preprocess_data(X_train, X_test, y_train, y_test):
    """
    Load the raw splitted data, apply preprocessing and save the resulting artifacts.

    The following objects are stored in DATA_PROC_PATH as a pickle file:
    (X_train_processed, X_test_processed, y_train, y_test, preprocessor)

    where:
        X_train_processed : processed training features
        X_test_processed  : processed test features
        y_train           : training target values
        y_test            : test target values
        preprocessor      : fitted preprocessing object applied to the features
                            (e.g., scaler, column transformer, etc.)
    """

    print("Check missing values")
    print(f"Missing values in X_train:\n{X_train.isna().sum()}")
    print(f"Missing values in X_test:\n{X_test.isna().sum()}")

    # Scale features
    print("Apply feature scaling (StandardScaler)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns, index=X_test.index)

    # Save results
    print(f"Save the preprocessed training and test sets to {DATA_PROC_PATH}...")
    model_dir = os.path.dirname(DATA_PROC_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(DATA_PROC_PATH, "wb") as file:
        pickle.dump((X_train_scaled, X_test_scaled, y_train, y_test, scaler), file)

    print("Preprocessing completed.")

    return (X_train_scaled, X_test_scaled, y_train, y_test)
