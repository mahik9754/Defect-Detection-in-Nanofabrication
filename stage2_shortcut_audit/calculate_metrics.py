
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parent
PREDICTIONS_FILE = ROOT / "predictions.csv"
METRICS_FILE = ROOT / "shortcut_metrics.csv"

CONDITIONS = ["original", "defect_only", "background_only"]
CLASS_LABELS = [1, 2, 3, 4, 5, 6]


def calculate_overall_metrics(df):
    rows = []

    for condition in CONDITIONS:
        part = df[df["condition"] == condition].copy()

        y_true = part["true_label"].astype(int)
        y_pred = part["predicted_label"].astype(int)

        rows.append({
            "metric_scope": "overall",
            "condition": condition,
            "class": "all",
            "n_images": len(part),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(
                y_true, y_pred
            ),
            "macro_f1": f1_score(
                y_true, y_pred,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            ),
            "macro_precision": precision_score(
                y_true, y_pred,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            ),
            "macro_recall": recall_score(
                y_true, y_pred,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            ),
            "mean_confidence": part["confidence"].mean(),
            "prediction_change_vs_original": None,
        })

    return rows


def calculate_class_metrics(df):
    rows = []

    for condition in CONDITIONS:
        part = df[df["condition"] == condition].copy()

        report = classification_report(
            part["true_label"].astype(int),
            part["predicted_label"].astype(int),
            labels=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        )

        for class_id in CLASS_LABELS:
            item = report[str(class_id)]

            rows.append({
                "metric_scope": "per_class",
                "condition": condition,
                "class": class_id,
                "n_images": int(
                    (part["true_label"].astype(int) == class_id).sum()
                ),
                "accuracy": None,
                "balanced_accuracy": None,
                "macro_f1": None,
                "macro_precision": None,
                "macro_recall": None,
                "class_precision": item["precision"],
                "class_recall": item["recall"],
                "class_f1": item["f1-score"],
                "mean_confidence": None,
                "prediction_change_vs_original": None,
            })

    return rows


def calculate_prediction_changes(df):
    wide = df.pivot(
        index="filename",
        columns="condition",
        values="predicted_label",
    )

    original = wide["original"]
    rows = []

    for condition in ["defect_only", "background_only"]:
        changed = wide[condition] != original

        rows.append({
            "metric_scope": "prediction_change",
            "condition": condition,
            "class": "all",
            "n_images": len(wide),
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "macro_precision": None,
            "macro_recall": None,
            "mean_confidence": None,
            "prediction_change_vs_original": changed.mean(),
            "changed_prediction_count": int(changed.sum()),
        })

    return rows


def main():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required = {
        "filename",
        "true_label",
        "predicted_label",
        "confidence",
        "condition",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"predictions.csv is missing columns: {sorted(missing)}"
        )

    counts = df.groupby("condition").size().to_dict()
    expected = set(CONDITIONS)

    if set(counts) != expected:
        raise ValueError(
            f"Expected conditions {expected}, got {set(counts)}"
        )

    if any(counts[c] != 688 for c in CONDITIONS):
        raise ValueError(
            f"Expected 688 predictions per condition, got {counts}"
        )

    if df.duplicated(["filename", "condition"]).any():
        raise ValueError(
            "Duplicate filename-condition predictions found."
        )

    # Verify each image has the same true label in all conditions.
    label_counts = df.groupby("filename")["true_label"].nunique()
    if (label_counts != 1).any():
        raise ValueError(
            "Some images have inconsistent true labels across conditions."
        )

    # Ensure all conditions contain the same image set.
    image_sets = {
        c: set(df.loc[df["condition"] == c, "filename"])
        for c in CONDITIONS
    }
    if not (
        image_sets["original"] == image_sets["defect_only"]
        == image_sets["background_only"]
    ):
        raise ValueError(
            "The three conditions do not contain identical test images."
        )

    rows = []
    rows.extend(calculate_overall_metrics(df))
    rows.extend(calculate_class_metrics(df))
    rows.extend(calculate_prediction_changes(df))

    # Add original-vs-variant correctness changes.
    correctness = df.pivot(
        index="filename",
        columns="condition",
        values="correct",
    )

    for condition in ["defect_only", "background_only"]:
        original_correct = correctness["original"].astype(bool)
        variant_correct = correctness[condition].astype(bool)

        rows.append({
            "metric_scope": "correctness_change",
            "condition": condition,
            "class": "all",
            "n_images": len(correctness),
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "macro_precision": None,
            "macro_recall": None,
            "mean_confidence": None,
            "prediction_change_vs_original": None,
            "changed_prediction_count": None,
            "correct_to_wrong": int(
                (original_correct & ~variant_correct).sum()
            ),
            "wrong_to_correct": int(
                (~original_correct & variant_correct).sum()
            ),
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_FILE, index=False)

    print("\nMETRICS CALCULATION COMPLETE")
    print(f"Input predictions: {len(df)}")
    print(f"Output: {METRICS_FILE}")

    overall = metrics[metrics["metric_scope"] == "overall"]
    print("\nOverall results:")
    print(
        overall[
            [
                "condition",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "mean_confidence",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    changes = metrics[
        metrics["metric_scope"] == "prediction_change"
    ]
    print("\nPrediction changes relative to original:")
    print(
        changes[
            [
                "condition",
                "prediction_change_vs_original",
                "changed_prediction_count",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    main()
