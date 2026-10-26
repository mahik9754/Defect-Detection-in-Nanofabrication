
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

MODEL_PATH = "models/filtered/filtered_seed42.keras"
IMAGE_DIR = "dataset/processed_filtered/test"
OUTPUT_DIR = "stage4_gradcam_robustness/results/gradcam"
IMG_SIZE = (224, 224)


def find_backbone(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                layer.get_layer("Conv_1")
                return layer
            except ValueError:
                continue
    raise ValueError("Could not find MobileNetV2 backbone or Conv_1.")


def build_gradcam_model(model):
    backbone = find_backbone(model)
    conv_layer = backbone.get_layer("Conv_1")

    # Model that returns Conv_1 activations from the original input.
    feature_extractor = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=conv_layer.output,
        name="gradcam_feature_extractor"
    )

    image_input = tf.keras.Input(shape=(224, 224, 3))
    x = model.layers[1](image_input)  # Saved Rescaling layer
    conv_features = feature_extractor(x, training=False)

    # Continue through the original backbone from Conv_1 onward.
    x = conv_features
    found_conv = False
    for layer in backbone.layers:
        if layer.name == "Conv_1":
            found_conv = True
            continue
        if found_conv:
            x = layer(x)

    # Existing classification head.
    x = model.layers[-3](x)  # GlobalAveragePooling2D
    x = model.layers[-2](x)  # Dropout
    predictions = model.layers[-1](x)  # Dense

    return tf.keras.Model(
        inputs=image_input,
        outputs=[conv_features, predictions],
        name="gradcam_model"
    )


def generate_heatmap(grad_model, image_tensor):
    with tf.GradientTape() as tape:
        conv_features, predictions = grad_model(
            image_tensor, training=False
        )
        class_id = tf.argmax(predictions[0])
        score = predictions[:, class_id]

    gradients = tape.gradient(score, conv_features)
    if gradients is None:
        raise RuntimeError("Gradients are disconnected from Conv_1.")

    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(
        conv_features[0] * weights[0][tf.newaxis, tf.newaxis, :],
        axis=-1
    )
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.maximum(
        tf.reduce_max(heatmap), tf.keras.backend.epsilon()
    )

    return heatmap.numpy(), int(class_id.numpy()), predictions.numpy()[0]



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Loaded model:", model.input_shape, "->", model.output_shape)

    grad_model = build_gradcam_model(model)
    print("Grad-CAM graph built.")

    # Check whether both models produce matching predictions.
    test_input = tf.random.uniform(
        (2, 224, 224, 3),
        minval=0,
        maxval=256,
        dtype=tf.float32
    )

    _, grad_predictions = grad_model(test_input, training=False)
    saved_predictions = model(test_input, training=False).numpy()

    difference = np.max(
        np.abs(saved_predictions - grad_predictions.numpy())
    )

    print("\nMaximum prediction difference:", difference)

    if difference > 1e-4:
        raise ValueError(
            "Prediction check failed. Do not trust the heatmaps yet."
        )

    print("Prediction check passed.")

    # Generate Grad-CAM images only after the check passes.
    image_paths = []

    for root, _, files in os.walk(IMAGE_DIR):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(root, filename))

    image_paths = sorted(image_paths)[:20]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

    for path in image_paths:
        image = Image.open(path).convert("RGB").resize(IMG_SIZE)
        array = np.asarray(image, dtype=np.float32)
        tensor = array[np.newaxis, ...]

        heatmap, class_id, predictions = generate_heatmap(
            grad_model, tensor
        )

        heatmap = tf.image.resize(
            heatmap[..., np.newaxis], IMG_SIZE
        ).numpy().squeeze()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(array.astype(np.uint8))
        axes[0].set_title("Original image")

        axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM heatmap")

        axes[2].imshow(array.astype(np.uint8))
        axes[2].imshow(
            heatmap, cmap="jet", alpha=0.4, vmin=0, vmax=1
        )
        axes[2].set_title(
            f"Predicted class {class_id + 1}\n"
            f"Confidence {predictions[class_id]:.3f}"
        )

        for axis in axes:
            axis.axis("off")

        fig.tight_layout()

        name = os.path.splitext(os.path.basename(path))[0]
        output_path = os.path.join(
            OUTPUT_DIR, name + "_gradcam.png"
        )

        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print("Saved:", output_path)

    print("Grad-CAM run complete.")


if __name__ == "__main__":
    main()
