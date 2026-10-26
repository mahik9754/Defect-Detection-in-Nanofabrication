import os
import csv
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/original/original_seed42.keras"
CSV_PATH = "dataset/carinthia_s/raw/data/carinthia-s.csv"
IMAGE_DIR = "dataset/carinthia_s/raw/data/images"
MASK_DIR = "dataset/carinthia_s/raw/data/masks"
OUTPUT_DIR = "stage4_gradcam_robustness/results/mask_evaluation_corrected"

IMG_SIZE = (224, 224)
MAX_IMAGES_PER_CLASS = 100
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6]


def find_backbone(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                layer.get_layer("Conv_1")
                return layer
            except ValueError:
                continue
    raise ValueError("Could not find MobileNetV2 backbone.")


def build_gradcam_model(model):
    backbone = find_backbone(model)
    conv_layer = backbone.get_layer("Conv_1")

    feature_extractor = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=conv_layer.output,
        name="gradcam_feature_extractor"
    )

    image_input = tf.keras.Input(shape=(224, 224, 3))
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

    x = model.layers[-3](x)
    x = model.layers[-2](x)
    predictions = model.layers[-1](x)

    grad_model = tf.keras.Model(
        inputs=image_input,
        outputs=[conv_features, predictions]
    )
    return grad_model


def generate_heatmap(grad_model, image_tensor):
    with tf.GradientTape() as tape:
        conv_features, predictions = grad_model(
            image_tensor, training=False
        )
        class_id = tf.argmax(predictions[0])
        score = predictions[:, class_id]

    gradients = tape.gradient(score, conv_features)
    if gradients is None:
        raise RuntimeError("Gradients disconnected from Conv_1.")

    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(
        conv_features[0] * weights[0][tf.newaxis, tf.newaxis, :],
        axis=-1
    )
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.maximum(
        tf.reduce_max(heatmap), tf.keras.backend.epsilon()
    )

    heatmap = tf.image.resize(
        heatmap[..., np.newaxis], IMG_SIZE
    ).numpy().squeeze()

    return heatmap, int(class_id.numpy())


def read_rows():
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def resolve_file(folder, name):
    path = os.path.join(folder, name)
    if os.path.isfile(path):
        return path

    stem = os.path.splitext(name)[0]
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            if os.path.splitext(filename)[0] == stem:
                candidate = os.path.join(folder, filename)
                if os.path.isfile(candidate):
                    return candidate
    return None


def calculate_scores(heatmap, mask):
    results = []

    for threshold in THRESHOLDS:
        predicted = heatmap >= threshold
        actual = mask > 0

        intersection = np.logical_and(predicted, actual).sum()
        union = np.logical_or(predicted, actual).sum()

        dice_denominator = predicted.sum() + actual.sum()
        iou = intersection / union if union else 1.0
        dice = (
            2 * intersection / dice_denominator
            if dice_denominator else 1.0
        )

        results.append((threshold, iou, dice))

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Loaded:", MODEL_PATH, model.input_shape, "->", model.output_shape)

    grad_model = build_gradcam_model(model)

    # Confirm that the Grad-CAM graph reproduces the saved model.
    test_input = tf.random.uniform(
        (2, 224, 224, 3), minval=0, maxval=256, dtype=tf.float32
    )
    _, grad_predictions = grad_model(test_input, training=False)
    saved_predictions = model(test_input, training=False).numpy()

    difference = np.max(
        np.abs(saved_predictions - grad_predictions.numpy())
    )
    print("Maximum prediction difference:", difference)

    if difference > 1e-4:
        raise RuntimeError(
            "Prediction consistency check failed. "
            "Do not trust evaluation results."
        )

    print("Prediction consistency check passed.")

    rows = read_rows()
    selected = {str(i): [] for i in range(1, 6)}
    skipped = 0

    for row in rows:
        label = str(row["label"]).strip()

        # Class 6 represents no-defect and is excluded from localization.
        if label not in selected:
            continue

        image_name = os.path.basename(row["image_path"].strip())
        mask_name = os.path.basename(row["mask_path"].strip())

        image_path = resolve_file(IMAGE_DIR, image_name)
        mask_path = resolve_file(MASK_DIR, mask_name)

        if not image_path or not mask_path:
            skipped += 1
            continue

        if len(selected[label]) < MAX_IMAGES_PER_CLASS:
            selected[label].append((image_path, mask_path))

    print("Missing image/mask pairs skipped:", skipped)
    print("Selected images per class:",
          {k: len(v) for k, v in selected.items()})

    detailed_rows = []

    for label, pairs in selected.items():
        for image_path, mask_path in pairs:
            image = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
            image_array = np.asarray(image, dtype=np.float32)
            image_tensor = image_array[np.newaxis, ...]

            heatmap, predicted_class = generate_heatmap(
                grad_model, image_tensor
            )

            mask_image = Image.open(mask_path).convert("L").resize(
                IMG_SIZE, Image.Resampling.NEAREST
            )
            mask = np.asarray(mask_image)

            # Empty expert masks are not meaningful for defect localization.
            if not np.any(mask > 0):
                continue

            scores = calculate_scores(heatmap, mask)

            for threshold, iou, dice in scores:
                detailed_rows.append({
                    "true_label": label,
                    "predicted_label": predicted_class + 1,
                    "threshold": threshold,
                    "iou": iou,
                    "dice": dice,
                    "image": os.path.basename(image_path)
                })

    detailed_path = os.path.join(OUTPUT_DIR, "mask_scores.csv")

    with open(detailed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "true_label", "predicted_label", "threshold",
            "iou", "dice", "image"
        ])
        writer.writeheader()
        writer.writerows(detailed_rows)

    summary_path = os.path.join(OUTPUT_DIR, "mask_summary.csv")

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "threshold", "images", "mean_iou", "mean_dice"
        ])

        for threshold in THRESHOLDS:
            subset = [
                r for r in detailed_rows
                if r["threshold"] == threshold
            ]
            if subset:
                writer.writerow([
                    threshold,
                    len(subset),
                    np.mean([r["iou"] for r in subset]),
                    np.mean([r["dice"] for r in subset])
                ])

    print("Non-empty-mask evaluations:", len(detailed_rows) // len(THRESHOLDS))
    print("Saved:", detailed_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()