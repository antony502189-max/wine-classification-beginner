from train import train_and_evaluate


def test_model_reaches_reasonable_accuracy(tmp_path):
    metrics = train_and_evaluate(tmp_path)

    assert metrics["accuracy"] >= 0.85
    assert (tmp_path / "wine_model.joblib").exists()
    assert (tmp_path / "confusion_matrix.png").exists()

