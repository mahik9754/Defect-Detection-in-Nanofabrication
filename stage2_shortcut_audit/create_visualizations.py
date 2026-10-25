
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Project paths
ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MANIFEST_FILE = DATASET_DIR / "manifest.csv"
CARINTHIA_CSV = DATASET_DIR / "carinthia_s" / "raw" / "carinthia-s.csv"

# Import the existing region extraction functions
STAGE1_DIR = ROOT / "stage1_data_pipeline"
sys.path.insert(0, str(STAGE1_DIR))

from region_extraction import extract_regions

OUTPUT_DIR = AUDIT_DIR / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = AUDIT_DIR / "predictions.csv"

# Load existing test predictions
predictions = pd.read_csv(PREDICTIONS_FILE)
original = predictions[predictions["condition"] == "original"].copy()

# Prefer correctly classified images with nonempty masks.
# This helps us inspect meaningful examples of the intended regions.
original["true_label"] = original["true_label"].astype(int)
original["predicted_label"] = original["predicted_label"].astype(int)

# Load dataset metadata
manifest = pd.read_csv(MANIFEST_FILE)
carinthia = pd.read_csv(CARINTHIA_CSV, sep=";")

# Join records by filename stem
manifest["stem"] = manifest["filepath"].apply(
    lambda p: Path(str(p)).stem
)
carinthia["stem"] = carinthia["filename"].apply(
    lambda p: Path(str(p)).stem
)

test_manifest = manifest[manifest["split"].astype(str).str.lower() == "test"]

records = test_manifest.merge(
    carinthia,
    on="stem",
    how="inner",
    suffixes=("_manifest", "_csv"),
)

# Keep only records correctly classified in the original condition.
records = records.merge(
    original[["filename", "true_label", "predicted_label"]],
    left_on="stem",
    right_on="filename",
    how="inner",
)

records = records[
    records["true_label"] == records["predicted_label"]
].copy()

if records.empty:
    raise RuntimeError(
        "No matching correctly classified test images were found. "
        "Check manifest and CSV filename columns."
    )

# Resolve image and mask paths, supporting paths stored relative to project root.
def resolve_path(value):
    path = Path(str(value))
    if path.is_absolute() and path.exists():
        return path

    candidates = [
        ROOT / path,
        DATASET_DIR / "carinthia_s" / "raw" / "images" / path.name,
        DATASET_DIR / "carinthia_s" / "raw" / "masks" / path.name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not resolve dataset path: {value}")


# Use the CSV's explicit image and mask paths.
records["resolved_image"] = records["image_path"].apply(resolve_path)
records["resolved_mask"] = records["mask_path"].apply(resolve_path)

# Select examples with nonempty masks.
def mask_is_nonempty(mask_path):
    with Image.open(mask_path) as im:
        return im.convert("L").getbbox() is not None

records["mask_nonempty_actual"] = records["resolved_mask"].apply(
    mask_is_nonempty
)

examples = records[records["mask_nonempty_actual"]].head(6)

if examples.empty:
    raise RuntimeError(
        "No nonempty masks found among matched test records. "
        "Inspect the CSV paths and mask conventions."
    )

# Save individual three-panel comparisons.
for _, row in examples.iterrows():
    record_id = row["stem"]

    regions = extract_regions(
        image_path=row["resolved_image"],
        defect_mask_path=row["resolved_mask"],
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, condition in zip(
        axes, ["original", "defect_only", "background_only"]
    ):
        ax.imshow(regions[condition])
        ax.set_title(condition.replace("_", " ").title())
        ax.axis("off")

    fig.suptitle(
        f"Record {record_id} | True class {row['true_label']}",
        fontsize=13,
    )
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{record_id}_comparison.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

# Create one combined figure for the report.
n = len(examples)
fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

if n == 1:
    axes = axes.reshape(1, 3)

for i, (_, row) in enumerate(examples.iterrows()):
    regions = extract_regions(
        image_path=row["resolved_image"],
        defect_mask_path=row["resolved_mask"],
    )

    for j, condition in enumerate(
        ["original", "defect_only", "background_only"]
    ):
        axes[i, j].imshow(regions[condition])
        axes[i, j].axis("off")

        if i == 0:
            axes[i, j].set_title(
                condition.replace("_", " ").title()
            )

    axes[i, 0].text(
        0.02, 0.98,
        f"Class {row['true_label']}",
        transform=axes[i, 0].transAxes,
        color="yellow",
        va="top",
        bbox={"facecolor": "black", "alpha": 0.65},
    )

fig.suptitle(
    "Visual Audit: Original, Defect-Only, and Background-Only SEM Images",
    fontsize=14,
)
fig.tight_layout(rect=[0, 0, 1, 0.98])

combined_path = OUTPUT_DIR / "region_comparison.png"
fig.savefig(combined_path, dpi=180, bbox_inches="tight")
plt.close(fig)

print("VISUALIZATIONS CREATED")
print(f"Combined comparison: {combined_path}")
print(f"Individual examples: {OUTPUT_DIR}")
print(f"Examples visualized: {n}")
