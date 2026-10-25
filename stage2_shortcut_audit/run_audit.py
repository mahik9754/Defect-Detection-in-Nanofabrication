
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SEED = 42

MODEL_PATH = ROOT / "models" / "original" / f"original_seed{SEED}.keras"
CONFIG_PATH = ROOT / "results" / "original" / f"original_seed{SEED}_run_config.json"
MANIFEST_PATH = ROOT / "dataset" / "manifest.csv"
DATA_CSV = ROOT / "dataset" / "carinthia_s" / "raw" / "carinthia-s.csv"
IMAGE_DIR = ROOT / "dataset" / "carinthia_s" / "raw" / "images"
MASK_DIR = ROOT / "dataset" / "carinthia_s" / "raw" / "masks"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = OUTPUT_DIR / "predictions.csv"

BATCH_SIZE = 32
CONDITIONS = ("original", "defect_only", "background_only")


def find_file(folder, stem):
    """Find a file with the requested stem, regardless of extension."""
    matches = list(folder.glob(stem + ".*"))
    if not matches:
        raise FileNotFoundError(f"No file with stem {stem!r} in {folder}")
    return matches[0]


def main():
    # Let inference.py import the existing Stage 1 preprocessing code.
    ROOT = Path(__file__).resolve().parent.parent
    stage1_dir = ROOT / "stage1_data_pipeline"

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(stage1_dir))

    import inference
    from inference import load_inference_model
    from preprocessing import process_image

    # Check required inputs before starting.
    required = [MODEL_PATH, CONFIG_PATH, MANIFEST_PATH, DATA_CSV]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    manifest = pd.read_csv(MANIFEST_PATH, dtype=str)
    dataset = pd.read_csv(DATA_CSV, sep=";", dtype=str).fillna("")

    # Preserve the saved split. Do not generate a new one.
    test = manifest[
        manifest["split"].str.lower() == "test"
    ].copy()

    test["stem"] = test["filepath"].map(lambda x: Path(x).stem)
    dataset["stem"] = dataset["filename"].map(
        lambda x: Path(x).stem
    )

    joined = test.merge(
        dataset[["stem", "label"]],
        on="stem",
        how="left",
        validate="one_to_one",
    )

    if joined["label"].isna().any():
        missing = int(joined["label"].isna().sum())
        raise RuntimeError(
            f"{missing} saved test images did not match the dataset CSV."
        )

    joined["true_label"] = joined["class"].astype(str)
    joined["csv_label"] = joined["label"].astype(str)

    mismatches = joined[
        joined["true_label"] != joined["csv_label"]
    ]
    if not mismatches.empty:
        raise RuntimeError(
            f"{len(mismatches)} labels differ between the saved manifest "
            "and Carinthia-S CSV. Stopping to protect evaluation validity."
        )

    print(f"Saved test images matched: {len(joined)}")

    model, class_names = load_inference_model(
        MODEL_PATH, CONFIG_PATH
    )
    class_names = [str(name) for name in class_names]
    class_index = {name: i for i, name in enumerate(class_names)}

    unknown = sorted(set(joined["true_label"]) - set(class_index))
    if unknown:
        raise RuntimeError(
            f"Test labels missing from model class mapping: {unknown}"
        )

    records = []

    # Temporary images are used only for preprocessing.
    # They are deleted automatically when the script finishes.
    with tempfile.TemporaryDirectory(prefix="stage2_sem_") as temp:
        temp_dir = Path(temp)

        for condition in CONDITIONS:
            print(f"\nRunning condition: {condition}")
            batch_images = []
            batch_metadata = []

            def predict_batch():
                if not batch_images:
                    return

                x = np.stack(batch_images).astype(np.float32)
                predictions = model.predict(
                    x, batch_size=BATCH_SIZE, verbose=0
                )

                for meta, probabilities in zip(
                    batch_metadata, predictions
                ):
                    predicted_index = int(np.argmax(probabilities))
                    predicted_label = class_names[predicted_index]

                    record = {
                        **meta,
                        "condition": condition,
                        "predicted_label": predicted_label,
                        "confidence": float(
                            probabilities[predicted_index]
                        ),
                        "true_class_probability": float(
                            probabilities[class_index[meta["true_label"]]]
                        ),
                        "correct": (
                            predicted_label == meta["true_label"]
                        ),
                    }

                    for j, name in enumerate(class_names):
                        record[f"prob_{name}"] = float(probabilities[j])

                    records.append(record)

                batch_images.clear()
                batch_metadata.clear()

            for number, row in enumerate(
                joined.itertuples(index=False), start=1
            ):
                stem = row.stem
                image_path = find_file(IMAGE_DIR, stem)
                mask_path = find_file(MASK_DIR, stem)

                with Image.open(image_path) as im:
                    image = im.convert("RGB")

                with Image.open(mask_path) as mask_image:
                    mask_image = mask_image.convert("L").resize(
                        image.size, Image.Resampling.NEAREST
                    )
                    # Provisional rule: any nonzero mask pixel is foreground.
                    mask = np.asarray(mask_image) > 0

                pixels = np.asarray(image).copy()

                if condition == "defect_only":
                    pixels = np.where(
                        mask[..., None], pixels, 0
                    ).astype(np.uint8)
                elif condition == "background_only":
                    pixels = np.where(
                        mask[..., None], 0, pixels
                    ).astype(np.uint8)

                mask_area = float(mask.mean())
                temp_image = temp_dir / f"{stem}_{condition}.png"
                Image.fromarray(pixels).save(temp_image)

                # Reuse the project's existing image preprocessing.
                processed = process_image(
                    str(temp_image), filtered=False
                )
                batch_images.append(
                    np.asarray(processed, dtype=np.float32)
                )
                batch_metadata.append({
                    "filename": stem,
                    "true_label": row.true_label,
                    "mask_nonempty": bool(mask.any()),
                    "mask_area_fraction": mask_area,
                })

                if len(batch_images) >= BATCH_SIZE:
                    predict_batch()

                if number % 50 == 0 or number == len(joined):
                    print(
                        f"  {condition}: {number}/{len(joined)} images"
                    )

            predict_batch()

    # Save only after all conditions complete successfully.
    output = pd.DataFrame(records)
    expected_rows = len(joined) * len(CONDITIONS)

    if len(output) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} predictions, got {len(output)}."
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)

    print("\nSTAGE 2 PREDICTION RUN COMPLETE")
    print(f"Test images: {len(joined)}")
    print(f"Predictions saved: {len(output)}")
    print(f"Output: {OUTPUT_CSV}")
    print("\nPrediction counts by condition:")
    print(output.groupby("condition").size().to_string())
    print("\nMean confidence by condition:")
    print(output.groupby("condition")["confidence"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
