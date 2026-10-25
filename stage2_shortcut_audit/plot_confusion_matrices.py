
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Paths
ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "predictions.csv"
OUTPUT_FILE = ROOT / "confusion_matrices.png"

CLASSES = [1, 2, 3, 4, 5, 6]
CONDITIONS = ["original", "defect_only", "background_only"]

# Load predictions
df = pd.read_csv(INPUT_FILE)

required_columns = {
    "filename", "true_label", "condition", "predicted_label"
}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Validate data before plotting
for condition in CONDITIONS:
    subset = df[df["condition"] == condition]

    if len(subset) != 688:
        raise ValueError(
            f"{condition}: expected 688 rows, found {len(subset)}"
        )

    if subset["filename"].duplicated().any():
        raise ValueError(
            f"Duplicate filenames found in {condition}"
        )

# Confirm the same test images and true labels are used
label_check = df.pivot(
    index="filename",
    columns="condition",
    values="true_label"
)

if label_check[CONDITIONS].isna().any().any():
    raise ValueError("Some filenames are missing a condition.")

if not (
    label_check["original"].equals(label_check["defect_only"])
    and label_check["original"].equals(label_check["background_only"])
):
    raise ValueError("True labels differ across conditions.")

# Plot row-normalized confusion matrices.
# Rows = true classes; columns = predicted classes.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, condition in zip(axes, CONDITIONS):
    subset = df[df["condition"] == condition]

    y_true = subset["true_label"].astype(int)
    y_pred = subset["predicted_label"].astype(int)

    counts = confusion_matrix(
        y_true, y_pred, labels=CLASSES
    )

    row_totals = counts.sum(axis=1, keepdims=True)
    percentages = counts / row_totals * 100

    image = ax.imshow(
        percentages,
        interpolation="nearest",
        vmin=0,
        vmax=100,
        cmap="Blues"
    )

    ax.set_title(condition.replace("_", " ").title())
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)

    # Show percentage and raw image count in every cell
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            pct = percentages[i, j]
            count = counts[i, j]

            text_color = "white" if pct >= 55 else "black"

            ax.text(
                j, i,
                f"{pct:.1f}%\n(n={count})",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color
            )

fig.colorbar(
    image,
    ax=axes.tolist(),
    label="Percentage of each true class (%)",
    shrink=0.8
)

fig.suptitle(
    "Confusion Matrices: Original vs Masked Inputs",
    fontsize=15
)

fig.text(
    0.5, 0.01,
    "Rows are true classes; columns are predicted classes. "
    "Each row is normalized separately.",
    ha="center",
    fontsize=10
)

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("CONFUSION MATRICES CREATED")
print(f"Saved to: {OUTPUT_FILE}")
