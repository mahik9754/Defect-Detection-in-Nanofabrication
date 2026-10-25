
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"

sys.path.insert(0, str(ROOT / "stage1_data_pipeline"))
from region_extraction import extract_regions

predictions = pd.read_csv(AUDIT_DIR / "predictions.csv")
manifest = pd.read_csv(DATASET_DIR / "manifest.csv")
carinthia = pd.read_csv(
    DATASET_DIR / "carinthia_s" / "raw" / "carinthia-s.csv",
    sep=";",
)

manifest["stem"] = manifest["filepath"].apply(
    lambda p: Path(str(p)).stem
)
carinthia["stem"] = carinthia["filename"].apply(
    lambda p: Path(str(p)).stem
)

test_manifest = manifest[
    manifest["split"].astype(str).str.lower() == "test"
]

records = test_manifest.merge(
    carinthia, on="stem", how="inner", suffixes=("_manifest", "_csv")
)

original = predictions[predictions["condition"] == "original"].copy()
original["true_label"] = original["true_label"].astype(int)
original["predicted_label"] = original["predicted_label"].astype(int)

records = records.merge(
    original[["filename", "true_label", "predicted_label"]],
    left_on="stem",
    right_on="filename",
    how="inner",
)

# Choose examples where the original model was correct,
# including examples where a masked prediction changes.
wide = predictions.pivot(
    index="filename",
    columns="condition",
    values="predicted_label",
)

wide.columns = [f"pred_{c}" for c in wide.columns]
# Use the dataset record's stem to match prediction filenames.
wide = wide.reset_index()

records = records.merge(
    wide,
    left_on="stem",
    right_on="filename",
    how="left",
    suffixes=("", "_prediction"),
)

records = records[
    records["true_label"] == records["predicted_label"]
].copy()

records["masked_change"] = (
    (records["pred_defect_only"] != records["pred_original"])
    | (records["pred_background_only"] != records["pred_original"])
)

records = records.sort_values("masked_change", ascending=False)
examples = records.head(6)

if examples.empty:
    raise RuntimeError("No matching test examples found.")

def resolve_path(value, folder):
    path = Path(str(value))
    candidates = [
        path,
        ROOT / path,
        DATASET_DIR / "carinthia_s" / "raw" / folder / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find {value}")

conditions = ["original", "defect_only", "background_only"]
fig, axes = plt.subplots(len(examples), 3, figsize=(15, 4 * len(examples)))

if len(examples) == 1:
    axes = axes.reshape(1, 3)

for i, (_, row) in enumerate(examples.iterrows()):
    image_path = resolve_path(row["image_path"], "images")
    mask_path = resolve_path(row["mask_path"], "masks")

    regions = extract_regions(image_path, mask_path)

    for j, condition in enumerate(conditions):
        ax = axes[i, j]
        ax.imshow(regions[condition])
        ax.axis("off")

        pred = int(row[f"pred_{condition}"])
        pred_rows = predictions[
            (predictions["filename"] == row["stem"])
            & (predictions["condition"] == condition)
        ]

        confidence = float(pred_rows.iloc[0]["confidence"])
        correct = pred == int(row["true_label"])

        ax.set_title(
            f"{condition.replace('_', ' ').title()}\n"
            f"True: {int(row['true_label'])} | "
            f"Predicted: {pred} | "
            f"Confidence: {confidence:.3f}\n"
            f"{'Correct' if correct else 'Incorrect'}"
        )

fig.suptitle(
    "Model Predictions on Original and Masked SEM Images",
    fontsize=15,
)
fig.tight_layout(rect=[0, 0, 1, 0.98])

output = AUDIT_DIR / "visualizations" / "example_predictions.png"
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=180, bbox_inches="tight")
plt.close(fig)

print("Example prediction visualization created:")
print(output)
