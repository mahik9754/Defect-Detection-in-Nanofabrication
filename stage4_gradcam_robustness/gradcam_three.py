
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------
# Paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
STAGE2_DIR = ROOT / "stage2_shortcut_audit"
DATASET_DIR = ROOT / "dataset"

MODEL_PATH = ROOT / "models" / "original" / "original_seed42.keras"
PREDICTIONS_FILE = STAGE2_DIR / "predictions.csv"
MANIFEST_FILE = DATASET_DIR / "manifest.csv"
CARINTHIA_CSV = (
    DATASET_DIR / "carinthia_s" / "raw" / "carinthia-s.csv"
)

OUTPUT_DIR = (
    ROOT / "stage4_gradcam_robustness"
    / "results" / "gradcam_three_conditions"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
NUM_EXAMPLES = 6

CONDITIONS = ["original", "defect_only", "background_only"]

STAGE1_DIR = ROOT / "stage1_data_pipeline"
sys.path.insert(0, str(STAGE1_DIR))
from region_extraction import extract_regions


# --------------------------------------------------
# Find MobileNetV2 and construct Grad-CAM model
# --------------------------------------------------

def find_backbone(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                layer.get_layer("Conv_1")
                return layer
            except ValueError:
                pass

    raise ValueError("Could not find MobileNetV2 Conv_1 layer.")


def build_gradcam_model(model):
    backbone = find_backbone(model)
    conv_layer = backbone.get_layer("Conv_1")

    feature_extractor = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=conv_layer.output,
    )

    image_input = tf.keras.Input(shape=(224, 224, 3))

    # The saved model contains its own Rescaling layer.
    x = model.layers[1](image_input)
    conv_features = feature_extractor(x, training=False)

    x = conv_features
    found_conv = False

    for layer in backbone.layers:
        if layer.name == "Conv_1":
            found_conv = True
            continue

        if found_conv:
            x = layer(x)

    x = model.layers[-3](x)  # GlobalAveragePooling2D
    x = model.layers[-2](x)  # Dropout
    predictions = model.layers[-1](x)  # Dense

    return tf.keras.Model(
        inputs=image_input,
        outputs=[conv_features, predictions],
    )


def generate_heatmap(grad_model, image_tensor, target_class):
    with tf.GradientTape() as tape:
        conv_features, predictions = grad_model(
            image_tensor, training=False
        )
        score = predictions[:, target_class]

    gradients = tape.gradient(score, conv_features)

    if gradients is None:
        raise RuntimeError("Gradients could not be calculated.")

    weights = tf.reduce_mean(gradients, axis=(1, 2))

    heatmap = tf.reduce_sum(
        conv_features[0] * weights[0][tf.newaxis, tf.newaxis, :],
        axis=-1,
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / tf.maximum(
        tf.reduce_max(heatmap),
        tf.keras.backend.epsilon(),
    )

    return heatmap.numpy()


# --------------------------------------------------
# Load images and metadata
# --------------------------------------------------

def resolve_path(value, preferred_folder=None):
    path = Path(str(value))

    candidates = []

    if path.is_absolute():
        candidates.append(path)

    candidates.extend([
        ROOT / path,
        DATASET_DIR / path,
    ])

    if preferred_folder:
        candidates.append(preferred_folder / path.name)

    raw_data_dir = DATASET_DIR / "carinthia_s" / "raw" / "data"

    # Search the actual image and mask folders.
    candidates.extend([
        raw_data_dir / "images" / path.name,
        raw_data_dir / "masks" / path.name,
    ])

    # Masks may be PNG even when the CSV path has another extension.
    if path.suffix.lower() != ".png":
        candidates.append(
            raw_data_dir / "masks" / f"{path.stem}.png"
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find: {value}\n"
        "Checked:\n" + "\n".join(str(p) for p in candidates)
    )

def load_resized_rgb(image):
    """Resize a PIL image to model input size."""
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def overlay_heatmap(image_array, heatmap):
    """Create a Grad-CAM overlay for display."""
    heatmap_image = Image.fromarray(
        np.uint8(np.clip(heatmap, 0, 1) * 255)
    )
    heatmap_image = heatmap_image.resize(
        IMG_SIZE, Image.Resampling.BILINEAR
    )

    heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0

    # Matplotlib's colormap is used only for visualization.
    colored = plt.get_cmap("jet")(heatmap_array)[..., :3] * 255.0

    overlay = 0.55 * image_array + 0.45 * colored
    return np.uint8(np.clip(overlay, 0, 255))


predictions_df = pd.read_csv(PREDICTIONS_FILE)
manifest = pd.read_csv(MANIFEST_FILE)
carinthia = pd.read_csv(CARINTHIA_CSV, sep=";")

required_prediction_columns = {
    "filename", "true_label", "predicted_label",
    "condition", "mask_nonempty",
}
missing = required_prediction_columns - set(predictions_df.columns)

if missing:
    raise ValueError(f"Missing predictions.csv columns: {sorted(missing)}")

original_predictions = predictions_df[
    predictions_df["condition"].astype(str).str.lower() == "original"
].copy()

original_predictions["true_label"] = pd.to_numeric(
    original_predictions["true_label"], errors="coerce"
)
original_predictions["predicted_label"] = pd.to_numeric(
    original_predictions["predicted_label"], errors="coerce"
)

# Select images correctly classified in the original condition.
original_predictions = original_predictions[
    original_predictions["true_label"]
    == original_predictions["predicted_label"]
].copy()

# Build filename stems for joining the two metadata files.
manifest["stem"] = manifest["filepath"].apply(
    lambda value: Path(str(value)).stem
)
carinthia["stem"] = carinthia["filename"].apply(
    lambda value: Path(str(value)).stem
)

original_predictions["stem"] = original_predictions["filename"].apply(
    lambda value: Path(str(value)).stem
)

test_manifest = manifest[
    manifest["split"].astype(str).str.lower() == "test"
].copy()

records = test_manifest.merge(
    carinthia,
    on="stem",
    how="inner",
    suffixes=("_manifest", "_csv"),
)

records = records.merge(
    original_predictions[
        ["stem", "true_label", "predicted_label"]
    ],
    on="stem",
    how="inner",
)

# Keep nonempty masks and select six examples deterministically.
def mask_is_nonempty(mask_path):
    with Image.open(mask_path) as mask_image:
        return mask_image.convert("L").getbbox() is not None


records["resolved_image"] = records["image_path"].apply(
    lambda value: resolve_path(
        value, DATASET_DIR / "carinthia_s" / "raw" / "images"
    )
)
records["resolved_mask"] = records["mask_path"].apply(
    lambda value: resolve_path(
        value, DATASET_DIR / "carinthia_s" / "raw" / "masks"
    )
)

records["mask_nonempty_actual"] = records["resolved_mask"].apply(
    mask_is_nonempty
)

records = records[records["mask_nonempty_actual"]].copy()
records = records.sort_values("stem").head(NUM_EXAMPLES)

if records.empty:
    raise RuntimeError(
        "No correctly classified test images with nonempty masks matched."
    )

print(f"Selected {len(records)} images.")


# --------------------------------------------------
# Load model and check Grad-CAM prediction consistency
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
grad_model = build_gradcam_model(model)

test_input = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
_, grad_test_predictions = grad_model(test_input, training=False)
normal_test_predictions = model(test_input, training=False)

max_difference = float(
    tf.reduce_max(
        tf.abs(grad_test_predictions - normal_test_predictions)
    ).numpy()
)

print(f"Grad-CAM/model prediction difference: {max_difference:.8f}")

if max_difference > 1e-4:
    raise RuntimeError(
        "Grad-CAM model predictions differ from the saved model. "
        "Stop rather than generating potentially incorrect heatmaps."
    )


# --------------------------------------------------
# Generate heatmaps for all three conditions
# --------------------------------------------------

summary_rows = []

for _, row in records.iterrows():
    record_id = row["stem"]

    regions = extract_regions(
        image_path=row["resolved_image"],
        defect_mask_path=row["resolved_mask"],
    )

    fig, axes = plt.subplots(3, 3, figsize=(14, 13))

    for row_index, condition in enumerate(CONDITIONS):
        region_image = regions[condition]
        image_array = load_resized_rgb(region_image)
        image_tensor = tf.convert_to_tensor(image_array[None, ...])

        # Use the TRUE class as the Grad-CAM target for every condition.
        # Labels in the CSV are 1..6; model output indices are 0..5.
        true_class_index = int(row["true_label"]) - 1

        heatmap = generate_heatmap(
            grad_model,
            image_tensor,
            target_class=true_class_index,
        )

        _, probabilities = grad_model(image_tensor, training=False)
        probabilities = probabilities.numpy()[0]

        predicted_index = int(np.argmax(probabilities))
        predicted_label = predicted_index + 1
        confidence = float(probabilities[predicted_index])

        overlay = overlay_heatmap(image_array, heatmap)

        axes[row_index, 0].imshow(image_array.astype(np.uint8))
        axes[row_index, 0].set_title(f"{condition}: input")

        axes[row_index, 1].imshow(
            heatmap, cmap="jet", vmin=0, vmax=1
        )
        axes[row_index, 1].set_title("Grad-CAM")

        axes[row_index, 2].imshow(overlay)
        axes[row_index, 2].set_title(
            f"Overlay | Predicted: {predicted_label} "
            f"| Confidence: {confidence:.3f}"
        )

        for ax in axes[row_index]:
            ax.axis("off")

        summary_rows.append({
            "filename": record_id,
            "true_label": int(row["true_label"]),
            "condition": condition,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "true_class_probability": float(
                probabilities[true_class_index]
            ),
        })

    fig.suptitle(
        f"Grad-CAM Comparison | {record_id} "
        f"| True class {int(row['true_label'])}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = OUTPUT_DIR / f"{record_id}_three_conditions.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path.name}")

summary_path = OUTPUT_DIR / "gradcam_three_conditions_summary.csv"
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

print("\nDONE")
print(f"Images visualized: {len(records)}")
print(f"Condition-specific heatmaps: {len(records) * len(CONDITIONS)}")
print(f"Results folder: {OUTPUT_DIR}")
print(f"Summary CSV: {summary_path}")
