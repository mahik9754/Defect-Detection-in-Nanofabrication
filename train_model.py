
import argparse
import json
import random
from collections import Counter

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import (
    IMG_SIZE,
    BATCH_SIZE,
    NUM_CLASSES,
    PROCESSED,
    MODELS_DIR,
    RESULTS_DIR
)
from model_builder import build_model, compile_model, unfreeze_top
from augmentation import augment_train_dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_datasets(data_dir,seed):
    """Load train and validation images from split/class folders."""

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    for folder in (train_dir, val_dir):
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing folder: {folder}\n"
                "Expected structure: dataset/version/train/class_name/"
                " and dataset/version/val/class_name/"
            )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        color_mode="rgb",
        shuffle=True,
        seed=seed
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        color_mode="rgb",
        shuffle=False
    )

    if len(train_ds.class_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"found {len(train_ds.class_names)}: {train_ds.class_names}"
        )

    if train_ds.class_names != val_ds.class_names:
        raise ValueError("Train and validation class folders do not match.")

    return train_ds, val_ds, train_ds.class_names


def calculate_class_weights(train_ds):
    """Calculate balanced class weights, capped at 5."""

    counts = Counter()

    for _, labels in train_ds:
        class_ids = np.argmax(labels.numpy(), axis=1)
        counts.update(class_ids.tolist())

    total = sum(counts.values())

    if len(counts) != NUM_CLASSES:
        raise ValueError(
            "Not all classes have training images. Check the dataset split."
        )

    weights = {}

    for class_id in range(NUM_CLASSES):
        weight = total / (NUM_CLASSES * counts[class_id])
        weights[class_id] = min(float(weight), 5.0)

    print("Training image counts:", dict(counts))
    print("Class weights:", weights)

    return weights


def combine_histories(*histories):
    combined = {}

    for history in histories:
        for metric, values in history.history.items():
            combined.setdefault(metric, []).extend(
                float(value) for value in values
            )

    return combined


def plot_history(history, output_path):
    epochs = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["accuracy"], label="Train")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["loss"], label="Train")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        choices=["original", "filtered"],
        required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = PROCESSED[args.version]

    run_name = f"{args.version}_seed{args.seed}"

    model_dir = MODELS_DIR / args.version
    results_dir = RESULTS_DIR / args.version

    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset version: {args.version}")
    print(f"Seed: {args.seed}")
    print(f"Dataset location: {data_dir}")

    train_ds, val_ds, class_names = load_datasets(data_dir, args.seed)

    print("Classes:", class_names)

    # Apply augmentation only to training images.
    train_ds = augment_train_dataset(train_ds)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Calculate weights before augmentation changes the dataset.
    # The dataset remains iterable after this calculation.
    class_weights = calculate_class_weights(
        tf.keras.utils.image_dataset_from_directory(
            data_dir / "train",
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            color_mode="rgb",
            shuffle=False
        )
    )

    model = build_model(
        num_classes=NUM_CLASSES,
        input_shape=(*IMG_SIZE, 3)
    )
    compile_model(model)

    checkpoint_path = model_dir / f"{run_name}.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("\nPhase 1: Training classification head")

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        class_weight=class_weights,
        callbacks=callbacks
    )

    print("\nPhase 2: Fine-tuning MobileNetV2")

    unfreeze_top(model, num_layers=30)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Restore the best model saved across both phases.
    best_model = tf.keras.models.load_model(checkpoint_path)

    history = combine_histories(history1, history2)

    history_path = results_dir / f"{run_name}_training_history.json"
    curves_path = results_dir / f"{run_name}_training_curves.png"
    config_path = results_dir / f"{run_name}_run_config.json"

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    plot_history(history, curves_path)

    run_config = {
        "version": args.version,
        "seed": args.seed,
        "image_size": list(IMG_SIZE),
        "batch_size": BATCH_SIZE,
        "num_classes": NUM_CLASSES,
        "class_names": class_names,
        "model": "MobileNetV2",
        "fine_tuning_layers": 30,
        "best_model": str(checkpoint_path)
    }

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)

    print("\nTraining completed.")
    print(f"Best model: {checkpoint_path}")
    print(f"History: {history_path}")
    print(f"Curves: {curves_path}")
    print(f"Config: {config_path}")


if __name__ == "__main__":
    main()
