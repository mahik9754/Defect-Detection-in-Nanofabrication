
"""Shared inference interface for the existing MobileNetV2 classifier."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from preprocessing import process_image


def load_inference_model(checkpoint_path, run_config_path):
    """
    Load a trained checkpoint and its recorded class mapping.

    The checkpoint and run configuration must belong to the same run.
    """
    checkpoint_path = Path(checkpoint_path)
    run_config_path = Path(run_config_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    if not run_config_path.is_file():
        raise FileNotFoundError(
            f"Run configuration not found: {run_config_path}"
        )

    with run_config_path.open("r", encoding="utf-8") as file:
        run_config = json.load(file)

    class_names = run_config.get("class_names")

    if not isinstance(class_names, list) or len(class_names) != 6:
        raise ValueError(
            "Run configuration must contain the six recorded class names."
        )

    model = tf.keras.models.load_model(checkpoint_path)

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            "Model output count does not match the recorded class mapping."
        )

    return model, class_names


def predict_image(image_path, model, class_names, filtered=False):
    """
    Predict one SEM image using the project's existing preprocessing.

    Returns the predicted class, confidence, and all six probabilities.
    """
    if len(class_names) != 6:
        raise ValueError("Exactly six class names are required.")

    # Reuse the existing grayscale, optional median-filter, resize and RGB
    # preprocessing. The model itself performs MobileNetV2 rescaling.
    processed_image = process_image(image_path, filtered=filtered)

    image_array = np.asarray(
        processed_image,
        dtype=np.float32
    )

    batch = np.expand_dims(image_array, axis=0)
    probabilities = model.predict(batch, verbose=0)[0]

    if probabilities.shape != (6,):
        raise ValueError(
            f"Expected six probabilities, got {probabilities.shape}"
        )

    predicted_index = int(np.argmax(probabilities))

    return {
        "image_path": str(image_path),
        "predicted_class_index": predicted_index,
        "predicted_class": str(class_names[predicted_index]),
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            str(class_names[index]): float(probabilities[index])
            for index in range(6)
        },
        "preprocessing": (
            "median_filtered" if filtered else "original"
        ),
    }
