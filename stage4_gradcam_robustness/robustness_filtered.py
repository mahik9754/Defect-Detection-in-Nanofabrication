
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageFilter, ImageEnhance

MODEL_PATH = "models/filtered/filtered_seed42.keras"
IMAGE_DIR = "dataset/processed_filtered/test"
OUTPUT_DIR = "stage4_gradcam_robustness/results/robustness"
IMG_SIZE = (224, 224)
MAX_IMAGES = 500
BATCH_SIZE = 16
SEED = 42


def load_image(path):
    image = Image.open(path).convert("RGB").resize(IMG_SIZE)
    return np.asarray(image, dtype=np.float32)


def make_variants(array, rng):
    image = Image.fromarray(np.uint8(np.clip(array, 0, 255)))

    noisy = np.clip(
        array + rng.normal(0, 15, array.shape), 0, 255
    ).astype(np.uint8)

    blurred = image.filter(ImageFilter.GaussianBlur(radius=2))
    low_contrast = ImageEnhance.Contrast(image).enhance(0.5)
    high_contrast = ImageEnhance.Contrast(image).enhance(1.5)

    return {
        "clean": array,
        "noise_std15": noisy.astype(np.float32),
        "blur_radius2": np.asarray(blurred, dtype=np.float32),
        "contrast_0.5": np.asarray(low_contrast, dtype=np.float32),
        "contrast_1.5": np.asarray(high_contrast, dtype=np.float32),
    }



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading filtered model...", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded.", flush=True)

    # Find all test images
    image_paths = []
    for root, _, files in os.walk(IMAGE_DIR):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, filename))

    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

    # Group images by their class folder
    class_groups = {}
    for path in image_paths:
        label = os.path.basename(os.path.dirname(path))
        class_groups.setdefault(label, []).append(path)

    classes = sorted(class_groups)

    if len(classes) != 6:
        raise ValueError(f"Expected 6 classes, found: {classes}")

    # Select a reproducible, balanced sample
    rng = np.random.default_rng(SEED)
    per_class = MAX_IMAGES // len(classes)
    remainder = MAX_IMAGES % len(classes)
    selected = []

    for i, label in enumerate(classes):
        group = class_groups[label].copy()
        rng.shuffle(group)

        count = per_class + (1 if i < remainder else 0)
        selected.extend(group[:count])

    image_paths = sorted(selected)

    print("\nSelected images per class:", flush=True)
    for label in classes:
        count = sum(
            os.path.basename(os.path.dirname(path)) == label
            for path in image_paths
        )
        print(f"Class {label}: {count}", flush=True)

    print(f"Total images selected: {len(image_paths)}", flush=True)

    if len(image_paths) < MAX_IMAGES:
        print(
            f"WARNING: Only {len(image_paths)} images available "
            f"for a target of {MAX_IMAGES}.",
            flush=True
        )

    # Generate and evaluate the five conditions
    rng = np.random.default_rng(SEED + 1)
    conditions = [
        "clean",
        "noise_std15",
        "blur_radius2",
        "contrast_0.5",
        "contrast_1.5"
    ]

    records = []
    total = len(image_paths)

    print(f"\nStarting robustness test on {total} images.", flush=True)

    for start in range(0, total, BATCH_SIZE):
        batch_paths = image_paths[start:start + BATCH_SIZE]
        batch_variants = {condition: [] for condition in conditions}
        batch_labels = []
        batch_filenames = []

        for path in batch_paths:
            true_label = os.path.basename(os.path.dirname(path))
            array = load_image(path)
            variants = make_variants(array, rng)

            batch_labels.append(true_label)
            batch_filenames.append(os.path.basename(path))

            for condition in conditions:
                batch_variants[condition].append(variants[condition])

        for condition in conditions:
            inputs = np.stack(batch_variants[condition], axis=0)
            predictions = model.predict(inputs, verbose=0)

            for i, probabilities in enumerate(predictions):
                predicted_class = str(int(np.argmax(probabilities)) + 1)
                confidence = float(np.max(probabilities))

                records.append({
                    "filename": batch_filenames[i],
                    "true_class": batch_labels[i],
                    "condition": condition,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "correct": predicted_class == batch_labels[i]
                })

        done = min(start + len(batch_paths), total)
        print(f"Processed {done}/{total} images.", flush=True)

        # Save progress after every batch
        results_path = os.path.join(
            OUTPUT_DIR, "robustness_results.csv"
        )
        pd.DataFrame(records).to_csv(results_path, index=False)

    # Save final summary
    df = pd.DataFrame(records)

    summary = df.groupby("condition").agg(
        accuracy=("correct", "mean"),
        mean_confidence=("confidence", "mean"),
        images=("filename", "nunique")
    ).reset_index()

    summary_path = os.path.join(
        OUTPUT_DIR, "robustness_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    print("\n--- Robustness summary ---")
    print(summary.to_string(index=False))
    print(f"\nResults saved in: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
