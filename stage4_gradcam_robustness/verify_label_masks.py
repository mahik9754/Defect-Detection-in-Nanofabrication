
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
csv_path = ROOT / "dataset/carinthia_s/raw/data/carinthia-s.csv"
data_root = csv_path.parent

df = pd.read_csv(csv_path, sep=";")

mask_counts = Counter()
empty_by_label = Counter()
nonempty_by_label = Counter()
missing = []
mismatched = []
examples = {}

for _, row in df.iterrows():
    label = str(row["label"])
    image_path = data_root / str(row["image_path"])
    mask_path = data_root / str(row["mask_path"])

    if not image_path.exists() or not mask_path.exists():
        missing.append((str(row["image_path"]), str(row["mask_path"])))
        continue

    if image_path.stem != mask_path.stem:
        mismatched.append((str(row["image_path"]), str(row["mask_path"])))

    with Image.open(mask_path) as im:
        arr = np.array(im.convert("L"))

    nonzero = int(np.count_nonzero(arr))

    if nonzero == 0:
        mask_counts["empty"] += 1
        empty_by_label[label] += 1
    else:
        mask_counts["nonempty"] += 1
        nonempty_by_label[label] += 1

    if label not in examples:
        values = np.unique(arr)
        examples[label] = {
            "file": mask_path.name,
            "nonzero_pixels": nonzero,
            "unique_values": values[:20].tolist(),
            "unique_value_count": len(values),
        }

print("\n=== MASK SUMMARY ===")
print("Total CSV rows:", len(df))
print("Mask totals:", dict(mask_counts))
print("Empty masks by label:", dict(empty_by_label))
print("Non-empty masks by label:", dict(nonempty_by_label))
print("Missing image/mask pairs:", len(missing))
print("Filename mismatches:", len(mismatched))

print("\n=== ONE MASK EXAMPLE PER LABEL ===")
for label, info in sorted(examples.items()):
    print(f"Label {label}: {info}")

if missing:
    print("\nFirst missing pairs:", missing[:5])

if mismatched:
    print("\nFirst filename mismatches:", mismatched[:5])

print("\n=== CHECK COMPLETE ===")
