from src.training.training_utils import load_latest_model, load_test_data_sample

N_SAMPLES = 5

def test_model_load():
    """Load the latest registered model and run predictions on a few test samples."""
    model, _ = load_latest_model()

    X_test_sample, y_test_sample = load_test_data_sample(N_SAMPLES)
    y_pred = model.predict(X_test_sample)

    # Display results
    prediction_preview = X_test_sample
    prediction_preview["target_true"] = y_test_sample
    prediction_preview["target_pred"] = y_pred

    print("Prediction preview:")
    print(prediction_preview)

