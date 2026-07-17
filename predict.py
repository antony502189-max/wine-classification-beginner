"""Predict a wine class with the trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.datasets import load_wine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/wine_model.joblib"))
    parser.add_argument("--features", type=float, nargs=13, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = joblib.load(args.model)
    dataset = load_wine()
    class_id = int(model.predict([args.features])[0])
    probabilities = model.predict_proba([args.features])[0]
    print(f"Predicted class: {dataset.target_names[class_id]}")
    print(f"Confidence: {probabilities[class_id]:.1%}")

