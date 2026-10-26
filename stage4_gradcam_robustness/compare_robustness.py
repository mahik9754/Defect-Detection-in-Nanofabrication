
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("stage4_gradcam_robustness/results")

filtered_path = ROOT / "robustness" / "robustness_summary.csv"
original_path = ROOT / "robustness_original" / "robustness_summary.csv"

if not filtered_path.exists() or not original_path.exists():
    raise FileNotFoundError(
        "Could not find both robustness_summary.csv files. "
        "Check the results folders."
    )

filtered = pd.read_csv(filtered_path)
original = pd.read_csv(original_path)

comparison = original.merge(
    filtered,
    on="condition",
    suffixes=("_original", "_filtered")
)

comparison["accuracy_difference_percentage_points"] = (
    comparison["accuracy_original"] - comparison["accuracy_filtered"]
) * 100

out_dir = ROOT / "comparison"
out_dir.mkdir(parents=True, exist_ok=True)

comparison.to_csv(out_dir / "robustness_comparison.csv", index=False)

conditions = comparison["condition"]
x = range(len(conditions))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(
    [i - width / 2 for i in x],
    comparison["accuracy_original"] * 100,
    width,
    label="Original model"
)
plt.bar(
    [i + width / 2 for i in x],
    comparison["accuracy_filtered"] * 100,
    width,
    label="Filtered model"
)

plt.xticks(list(x), conditions, rotation=20, ha="right")
plt.ylabel("Accuracy (%)")
plt.xlabel("Test condition")
plt.title("Original vs Filtered Model Robustness")
plt.ylim(0, 105)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "robustness_comparison.png", dpi=300)
plt.close()

print(comparison[
    [
        "condition",
        "accuracy_original",
        "accuracy_filtered",
        "accuracy_difference_percentage_points"
    ]
].to_string(index=False))

print("\nSaved comparison CSV and graph to:", out_dir)
