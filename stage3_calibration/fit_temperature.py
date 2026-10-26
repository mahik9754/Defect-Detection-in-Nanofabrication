
"""Fit temperature scaling and select a human-review threshold."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import ROOT, MANIFEST
from calibration import softmax


CHECKPOINT = ROOT / "models" / "original" / "original_seed42.keras"
RUN_CONFIG = ROOT / "results" / "original" / "original_seed42_run_config.json"
VALIDATION_DIR = ROOT / "dataset" / "processed_original" / "val"
OUTPUT_DIR = ROOT / "stage3_calibration"

MINIMUM_COVERAGE = 0.80
TEMPERATURE_MIN = 0.05
TEMPERATURE_MAX = 10.0
TEMPERATURE_CANDIDATES = 500


def get_validation_data():
    """Load the existing validation images and their labels."""

    df = pd.read_csv(MANIFEST, dtype={"class": str})
    df = df[df["split"] == "val"].copy()

    images = []
    labels = []

    for index, row in df.iterrows():
        image_path = (
            VALIDATION_DIR
            / str(row["class"])
            / f"{index:05d}.png"
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Validation image not found: {image_path}"
            )

        with tf.keras.utils.load_img(
            image_path,
            target_size=(224, 224),
            color_mode="rgb",
        ) as image:
            image_array = tf.keras.utils.img_to_array(image)

        images.append(image_array)
        labels.append(int(row["class"]) - 1)

    if not images:
        raise ValueError("No validation images were found.")

    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def negative_log_likelihood(probabilities, labels):
    """Calculate the mean negative log-likelihood."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    true_class_probabilities = probabilities[
        np.arange(len(labels)), labels
    ]

    true_class_probabilities = np.clip(
        true_class_probabilities, 1e-12, 1.0
    )

    return float(-np.mean(np.log(true_class_probabilities)))


def expected_calibration_error(probabilities, labels, bins=15):
    """Calculate ECE using maximum confidence and prediction accuracy."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    confidence = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = predictions == labels

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    for bin_index in range(bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

        if bin_index == bins - 1:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence >= lower) & (confidence < upper)

        if np.any(in_bin):
            bin_accuracy = np.mean(correct[in_bin])
            bin_confidence = np.mean(confidence[in_bin])
            bin_fraction = np.mean(in_bin)

            ece += bin_fraction * abs(
                bin_accuracy - bin_confidence
            )

    return float(ece)


def save_reliability_diagram(
    raw_probabilities,
    calibrated_probabilities,
    labels,
    output_path,
    bins=15,
):
    """Plot confidence against accuracy before and after calibration."""

    labels = np.asarray(labels)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    plt.figure(figsize=(7, 6))

    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    for probabilities, name in [
        (raw_probabilities, "Before calibration"),
        (calibrated_probabilities, "After calibration"),
    ]:
        confidence = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        correct = predictions == labels

        bin_confidence = []
        bin_accuracy = []

        for i in range(bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]

            if i == bins - 1:
                in_bin = (
                    (confidence >= lower)
                    & (confidence <= upper)
                )
            else:
                in_bin = (
                    (confidence >= lower)
                    & (confidence < upper)
                )

            if np.any(in_bin):
                bin_confidence.append(
                    float(np.mean(confidence[in_bin]))
                )
                bin_accuracy.append(
                    float(np.mean(correct[in_bin]))
                )

        plt.plot(
            bin_confidence,
            bin_accuracy,
            marker="o",
            label=name,
        )

    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Actual accuracy")
    plt.title("Reliability Diagram")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_coverage_error_curve(
    probabilities,
    labels,
    output_path,
):
    """Plot automatic-acceptance coverage against accepted error."""

    labels = np.asarray(labels)
    confidence = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = predictions == labels

    thresholds = np.unique(
        np.concatenate(([0.0], confidence, [1.0]))
    )

    coverages = []
    accepted_errors = []

    for threshold in thresholds:
        accepted = confidence >= threshold

        if not np.any(accepted):
            continue

        coverage = float(np.mean(accepted))
        accepted_error = float(np.mean(~correct[accepted]))

        coverages.append(coverage)
        accepted_errors.append(accepted_error)

    # Sort points by coverage so the curve is easy to read.
    order = np.argsort(coverages)

    coverages = np.asarray(coverages)[order]
    accepted_errors = np.asarray(accepted_errors)[order]

    plt.figure(figsize=(7, 6))
    plt.plot(
        coverages,
        accepted_errors,
        marker=".",
        markersize=4,
        label="Calibrated predictions",
    )

    plt.xlabel("Automatic-acceptance coverage")
    plt.ylabel("Error among accepted predictions")
    plt.title("Coverage-Error Curve")
    plt.xlim(0, 1.02)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def select_review_threshold(probabilities, labels):
    """Choose the lowest-error threshold with at least 80% coverage."""

    confidence = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = predictions == labels

    threshold_results = []

    # Include 1.00 as a candidate as well.
    thresholds = np.round(np.arange(0.50, 1.001, 0.01), 2)

    for threshold in thresholds:
        accepted = confidence >= threshold
        reviewed = ~accepted

        accepted_count = int(np.sum(accepted))
        review_count = int(np.sum(reviewed))
        coverage = float(np.mean(accepted))

        if accepted_count > 0:
            accepted_error = float(np.mean(~correct[accepted]))
        else:
            accepted_error = None

        threshold_results.append({
            "threshold": float(threshold),
            "coverage": coverage,
            "accepted_count": accepted_count,
            "review_count": review_count,
            "accepted_error": accepted_error,
        })

    eligible = [
        result
        for result in threshold_results
        if result["coverage"] >= MINIMUM_COVERAGE
        and result["accepted_error"] is not None
    ]

    if not eligible:
        raise ValueError(
            "No threshold achieved the required "
            f"{MINIMUM_COVERAGE:.0%} validation coverage."
        )

    # Prefer lower accepted error. If tied, prefer higher coverage.
    selected = min(
        eligible,
        key=lambda result: (
            result["accepted_error"],
            -result["coverage"],
        ),
    )

    threshold_path = OUTPUT_DIR / "threshold_candidates.csv"
    pd.DataFrame(threshold_results).to_csv(
        threshold_path, index=False
    )

    return selected, threshold_results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT}"
        )

    if not RUN_CONFIG.is_file():
        raise FileNotFoundError(
            f"Run configuration not found: {RUN_CONFIG}"
        )

    with open(RUN_CONFIG, "r", encoding="utf-8") as file:
        run_config = json.load(file)

    class_names = run_config.get("class_names")

    if class_names != ["1", "2", "3", "4", "5", "6"]:
        raise ValueError(
            "Unexpected class mapping in the run configuration."
        )

    print("Loading the existing model...")
    model = tf.keras.models.load_model(CHECKPOINT)

    final_layer = model.layers[-1]

    if (
        not isinstance(final_layer, tf.keras.layers.Dense)
        or final_layer.activation != tf.keras.activations.softmax
    ):
        raise ValueError(
            "Expected the model's final layer to be a "
            "Dense layer with softmax activation."
        )

    print("Loading validation images...")
    images, labels = get_validation_data()

    if len(labels) != len(images):
        raise ValueError("The number of images and labels does not match.")

    print(f"Validation samples: {len(labels)}")

    # Get the input to the final Dense layer.
    # The Dense weights and bias let us recover pre-softmax logits.
    feature_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=final_layer.input,
    )

    print("Extracting validation features...")
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
        raise ValueError(
            f"Unexpected logits shape: {logits.shape}"
        )

    # Find the temperature with the lowest validation NLL.
    print("Searching for the best temperature...")

    candidates = np.exp(
        np.linspace(
            np.log(TEMPERATURE_MIN),
            np.log(TEMPERATURE_MAX),
            TEMPERATURE_CANDIDATES,
        )
    )

    candidate_losses = []

    for temperature in candidates:
        probabilities = softmax(logits, temperature=temperature)
        loss = negative_log_likelihood(probabilities, labels)
        candidate_losses.append(loss)

    best_index = int(np.argmin(candidate_losses))
    best_temperature = float(candidates[best_index])

    raw_probabilities = softmax(logits, temperature=1.0)
    calibrated_probabilities = softmax(
        logits, temperature=best_temperature
    )
    reliability_path = OUTPUT_DIR / "reliability_diagram.png"
    coverage_error_path = OUTPUT_DIR / "coverage_error_curve.png"

    save_reliability_diagram(
        raw_probabilities,
        calibrated_probabilities,
        labels,
        reliability_path,
    )

    save_coverage_error_curve(
        calibrated_probabilities,
        labels,
        coverage_error_path,
    )    

    raw_nll = negative_log_likelihood(raw_probabilities, labels)
    calibrated_nll = negative_log_likelihood(
        calibrated_probabilities, labels
    )

    raw_ece = expected_calibration_error(raw_probabilities, labels)
    calibrated_ece = expected_calibration_error(
        calibrated_probabilities, labels
    )

    # Select the human-review threshold using validation data only.
    selected, threshold_results = select_review_threshold(
        calibrated_probabilities, labels
    )

    # Save temperature and threshold together for later inference.
    results = {
        "validation_samples": int(len(labels)),
        "temperature": best_temperature,
        "temperature_search_range": [
            TEMPERATURE_MIN,
            TEMPERATURE_MAX,
        ],
        "temperature_candidate_count": TEMPERATURE_CANDIDATES,
        "raw_nll": raw_nll,
        "calibrated_nll": calibrated_nll,
        "raw_ece_15_bins": raw_ece,
        "calibrated_ece_15_bins": calibrated_ece,
        "class_names": class_names,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "validation_split": "dataset/processed_original/val",
        "review_threshold": selected["threshold"],
        "minimum_validation_coverage": MINIMUM_COVERAGE,
        "validation_coverage": selected["coverage"],
        "validation_accepted_count": selected["accepted_count"],
        "validation_review_count": selected["review_count"],
        "validation_accepted_error": selected["accepted_error"],
        "threshold_selection_method": (
            "Lowest accepted-prediction error among thresholds "
            "with at least 80% validation coverage; ties prefer "
            "higher coverage."
        ),
    }

    config_path = OUTPUT_DIR / "calibration_config.json"
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    metrics = [
        {"metric": "raw_nll", "value": raw_nll},
        {"metric": "calibrated_nll", "value": calibrated_nll},
        {"metric": "raw_ece_15_bins", "value": raw_ece},
        {"metric": "calibrated_ece_15_bins", "value": calibrated_ece},
        {
            "metric": "selected_temperature",
            "value": best_temperature,
        },
        {
            "metric": "review_threshold",
            "value": selected["threshold"],
        },
        {
            "metric": "validation_coverage",
            "value": selected["coverage"],
        },
        {
            "metric": "validation_accepted_error",
            "value": selected["accepted_error"],
        },
        {
            "metric": "validation_accepted_count",
            "value": selected["accepted_count"],
        },
        {
            "metric": "validation_review_count",
            "value": selected["review_count"],
        },
    ]

    metrics_path = OUTPUT_DIR / "calibration_metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    print("\n--- Stage 3 Calibration Results ---")
    print(f"Validation samples: {len(labels)}")
    print(f"Selected temperature: {best_temperature:.6f}")
    print(f"Raw NLL: {raw_nll:.6f}")
    print(f"Calibrated NLL: {calibrated_nll:.6f}")
    print(f"Raw ECE: {raw_ece:.6f}")
    print(f"Calibrated ECE: {calibrated_ece:.6f}")

    print("\n--- Human-Review Routing ---")
    print(f"Selected threshold: {selected['threshold']:.2f}")
    print(f"Validation coverage: {selected['coverage']:.4f}")
    print(f"Accepted predictions: {selected['accepted_count']}")
    print(f"Sent for review: {selected['review_count']}")
    print(
        "Accepted-prediction error: "
        f"{selected['accepted_error']:.6f}"
    )

    print("\nSaved files:")
    print(config_path)
    print(metrics_path)
    print(OUTPUT_DIR / "threshold_candidates.csv")
    print(reliability_path)
    print(coverage_error_path)

if __name__ == "__main__":
    main()
