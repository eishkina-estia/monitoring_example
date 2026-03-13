import pandas as pd
from sklearn.model_selection import train_test_split

from src import common as common

DATA_RAW_PATH = common.CONFIG['paths']['data_raw']

TARGET = common.CONFIG['ml']['target_name']
RANDOM_STATE = common.CONFIG['ml']['random_state']
TEST_SIZE = 0.25

def load_raw_data():
    """Load the raw dataset and split it into features X and target y."""

    print("Load raw data...")
    data = pd.read_csv(DATA_RAW_PATH)
    print(f"{len(data)} rows loaded from {DATA_RAW_PATH}")

    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Split the dataset
    print("Split the dataset into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Train set: {len(X_train)} rows")
    print(f"Test set: {len(X_test)} rows")

    return X_train, X_test, y_train, y_test
