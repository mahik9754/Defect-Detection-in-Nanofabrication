
"""Evaluate frozen calibration and review routing on the test split."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from config import ROOT, MANIFEST
from calibration import softmax
from review_router import route_for_review
from fit_temperature import (
    expected_calibration_error,
    negative_log_likelihood,
)

CHECKPOINT = ROOT / "models" / "original" / "original_seed42.keras"
RUN_CONFIG = ROOT / "results" / "original" / "original_seed42_run_config.json"
CALIBRATION_CONFIG = ROOT / "stage3_calibration" / "calibration_config.json"
TEST_DIR = ROOT / "dataset" / "processed_original" / "test"
OUTPUT_DIR = ROOT / "stage3_calibration"


def get_test_data():
    """Load test images using their original manifest row indices."""

    df = pd.read_csv(MANIFEST, dtype={"class": str})
    df = df[df["split"] == "test"].copy()

    images = []
    labels = []

    for index, row in df.iterrows():
        image_path = (
            TEST_DIR
            / str(row["class"])
            / f"{index:05d}.png"
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Test image not found: {image_path}"
            )

        with tf.keras.utils.load_img(
            image_path,
            target_size=(224, 224),
            color_mode="rgb",
        ) as image:
            images.append(tf.keras.utils.img_to_array(image))

        labels.append(int(row["class"]) - 1)

    if not images:
        raise ValueError("No test images were found.")

    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def main():
    for path in (CHECKPOINT, RUN_CONFIG, CALIBRATION_CONFIG):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    with open(RUN_CONFIG, "r", encoding="utf-8") as file:
        run_config = json.load(file)

    with open(CALIBRATION_CONFIG, "r", encoding="utf-8") as file:
        calibration_config = json.load(file)

    class_names = run_config["class_names"]
    temperature = float(calibration_config["temperature"])
    threshold = float(calibration_config["review_threshold"])

    if class_names != ["1", "2", "3", "4", "5", "6"]:
        raise ValueError("Unexpected class mapping.")

    if temperature <= 0 or not 0 <= threshold <= 1:
        raise ValueError("Invalid saved temperature or threshold.")

    print("Loading the existing model...")
    model = tf.keras.models.load_model(CHECKPOINT)
    final_layer = model.layers[-1]

    if (
        not isinstance(final_layer, tf.keras.layers.Dense)
        or final_layer.activation != tf.keras.activations.softmax
    ):
        raise ValueError("Expected a Dense softmax output layer.")

    print("Loading held-out test images...")
    images, labels = get_test_data()
    print(f"Test samples: {len(labels)}")

    feature_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=final_layer.input,
    )

    print("Extracting test features...")
    features = feature_model.predict(
        images,
        batch_size=32,
        verbose=1,
    )

    logits = (
        features @ final_layer.kernel.numpy()
        + final_layer.bias.numpy()
    )

    if logits.shape != (len(labels), 6):
        raise ValueError(f"Unexpected logits shape: {logits.shape}")

    # Apply the temperature fitted on validation data.
    raw_probabilities = softmax(logits, temperature=1.0)
    calibrated_probabilities = softmax(
        logits, temperature=temperature
    )


    raw_predictions = np.argmax(raw_probabilities, axis=1)

    routing = route_for_review(calibrated_probabilities, threshold)

    predictions = routing["predicted_class"]
    reviewed = routing["needs_review"]
    accepted = ~reviewed
    correct = predictions == labels

    accepted_count = int(np.sum(accepted))
    review_count = int(np.sum(reviewed))

    coverage = float(np.mean(accepted))
    accepted_error = (
        float(np.mean(~correct[accepted]))
        if accepted_count > 0
        else None
    )

    

    metrics = {
        "test_samples": int(len(labels)),
        "temperature_from_validation": temperature,
        "threshold_from_validation": threshold,
        "raw_accuracy": float(np.mean(raw_predictions == labels)),
        "calibrated_accuracy": float(np.mean(correct)),
        "raw_nll": negative_log_likelihood(raw_probabilities, labels),
        "calibrated_nll": negative_log_likelihood(
            calibrated_probabilities, labels
        ),
        "raw_ece_15_bins": expected_calibration_error(
            raw_probabilities, labels
        ),
        "calibrated_ece_15_bins": expected_calibration_error(
            calibrated_probabilities, labels
        ),
        "automatic_acceptance_coverage": coverage,
        "accepted_count": accepted_count,
        "sent_for_review_count": review_count,
        "accepted_prediction_error": accepted_error,
        "review_rate": float(np.mean(reviewed)),
        "test_split": "dataset/processed_original/test",
        "test_used_for_tuning": False,
    }

    output_path = OUTPUT_DIR / "test_evaluation_metrics.csv"
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    ).to_csv(output_path, index=False)

    print("\n--- Final Test Evaluation ---")
    print(f"Test samples: {len(labels)}")
    print(f"Temperature (frozen): {temperature:.6f}")
    print(f"Threshold (frozen): {threshold:.2f}")
    print(f"Raw accuracy: {metrics['raw_accuracy']:.6f}")
    print(f"Calibrated accuracy: {metrics['calibrated_accuracy']:.6f}")
    print(f"Raw NLL: {metrics['raw_nll']:.6f}")
    print(f"Calibrated NLL: {metrics['calibrated_nll']:.6f}")
    print(f"Raw ECE: {metrics['raw_ece_15_bins']:.6f}")
    print(f"Calibrated ECE: {metrics['calibrated_ece_15_bins']:.6f}")
    print(f"Automatic acceptance coverage: {coverage:.4f}")
    print(f"Accepted predictions: {accepted_count}")
    print(f"Sent for review: {review_count}")

    if accepted_error is not None:
        print(f"Accepted-prediction error: {accepted_error:.6f}")
    else:
        print("Accepted-prediction error: undefined (no accepted images)")

    print(f"\nSaved test metrics to: {output_path}")


if __name__ == "__main__":
    main()
