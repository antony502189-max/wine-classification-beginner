"""Train and evaluate a beginner-friendly wine classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def load_data():
    """Return a reproducible train/test split of the wine dataset."""
    dataset = load_wine()
    return train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=dataset.target,
    ), dataset.target_names


def build_model() -> Pipeline:
    """Create a scaling and classification pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1_000)),
        ]
    )


def train_and_evaluate(output_dir: Path) -> dict[str, float]:
    """Train the model, save artifacts, and return the main metrics."""
    (x_train, x_test, y_train, y_test), target_names = load_data()
    model = build_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    metrics = {"accuracy": round(float(accuracy), 4)}

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "wine_model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=target_names,
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title("Wine classification: confusion matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(display.figure_)

    print(f"Accuracy: {accuracy:.3f}")
    print(classification_report(y_test, predictions, target_names=target_names))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts"), help="Artifact folder"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args.output_dir)

