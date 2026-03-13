from src.training.load_raw_data import load_raw_data
from src.training.preprocess_data import preprocess_data
from src.training.train_model import train_model
from src.training.save_monitoring_data import save_monitoring_reference_data
from src.training.test_model_load import test_model_load

def run_pipeline():
    print("STEP 1 - Load raw data")
    X_train_raw, X_test_raw, y_train, y_test = load_raw_data()

    print("\nSTEP 2 - Preprocess data")
    X_train_processed, X_test_processed, y_train, y_test = preprocess_data(X_train_raw, X_test_raw, y_train, y_test)

    print("\nSTEP 3 - Train model")
    train_model(X_train_processed, X_test_processed, y_train, y_test)

    print("\nSTEP 4 - Save monitoring reference data")
    save_monitoring_reference_data(X_train_raw, y_train)

    print("\nSTEP 5 - Load and test best model")
    test_model_load()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()