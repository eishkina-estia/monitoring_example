import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import common as common

DATA_RAW_PATH = common.CONFIG['paths']['data_raw']
DATA_PROC_PATH = common.CONFIG['paths']['data_processed']

TARGET = common.CONFIG['ml']['target_name']
RANDOM_STATE = common.CONFIG['ml']['random_state']
TEST_SIZE = 0.25

def load_data(data_path):
    """Load the raw dataset and split it into features X and target y."""

    print("Load raw data...")
    data = pd.read_csv(data_path)
    print(f"{len(data)} rows loaded from {data_path}")

    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y

def preprocess_data():
    """
    Load the raw data, split it into training and test sets,
    apply preprocessing and save the resulting artifacts.

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

    # Load raw data
    X, y = load_data(DATA_RAW_PATH)

    # Split the dataset
    print("Split the dataset into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Train set: {len(X_train)} rows")
    print(f"Test set: {len(X_test)} rows")

    print("Check missing values")
    print(f"Missing values in X_train:\n{X_train.isna().sum()}")
    print(f"Missing values in X_test:\n{X_test.isna().sum()}")

    # Scale features
    print("Apply feature scaling (StandardScaler)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

    # Save results
    print(f"Save the preprocessed training and test sets to {DATA_PROC_PATH}...")
    model_dir = os.path.dirname(DATA_PROC_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(DATA_PROC_PATH, "wb") as file:
        pickle.dump((X_train_scaled, X_test_scaled, y_train, y_test, scaler), file)

    print("Preprocessing completed.")

if __name__ == "__main__":

    preprocess_data()