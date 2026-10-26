
import os
import pandas as pd
from PIL import Image

DATA_DIR = "dataset/carinthia_s/raw/data"
CSV_PATH = os.path.join(DATA_DIR, "carinthia-s.csv")

df = pd.read_csv(CSV_PATH, sep=";")

print("Columns:", df.columns.tolist())
print("Total pairs:", len(df))

valid_pairs = 0
missing_pairs = 0

for _, row in df.iterrows():
    image_path = os.path.join(DATA_DIR, row["image_path"])
    mask_path = os.path.join(DATA_DIR, row["mask_path"])

    if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
        missing_pairs += 1
        continue

    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        if image.size != mask.size:
            print("Size mismatch:", row["filename"], image.size, mask.size)
            continue

    valid_pairs += 1

print("Valid image-mask pairs:", valid_pairs)
print("Missing image or mask:", missing_pairs)
print("Check complete.")
