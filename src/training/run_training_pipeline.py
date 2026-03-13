from src.training.preprocess_data import preprocess_data
from src.training.train_model import train_model
from src.training.test_model_load import test_model_load

def run_pipeline():
    print("STEP 1 - Preprocess data")
    preprocess_data()

    print("\nSTEP 2 - Train model")
    train_model()

    print("\nSTEP 3 - Load and test best model")
    test_model_load()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()