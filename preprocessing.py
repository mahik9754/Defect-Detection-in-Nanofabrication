
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter
import config as C


def to_uint8_gray(img):
    """Convert an image to 8-bit grayscale without clipping 16-bit data."""
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(img, dtype=np.float64)

        if not np.isfinite(arr).all():
            raise ValueError("Image contains non-finite pixel values")

        low = arr.min()
        high = arr.max()

        if high > low:
            arr = (arr - low) / (high - low) * 255
        else:
            arr = np.zeros_like(arr)

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    return img.convert("L")


def process_image(path, filtered=False):
    """Convert to grayscale, optionally filter, resize and convert to RGB."""
    with Image.open(path) as img:
        gray = to_uint8_gray(img)

    if filtered:
        gray = gray.filter(ImageFilter.MedianFilter(size=3))

    resized = gray.resize(
        C.IMG_SIZE,
        Image.Resampling.LANCZOS
    )

    return resized.convert("RGB")


def save_preprocessing_samples(df):
    """Save original-versus-filtered examples for visual inspection."""
    sample = df.head(4)

    if sample.empty:
        print("No images available for sample visualization.")
        return

    fig, axes = plt.subplots(
        len(sample), 3,
        figsize=(10, 3.2 * len(sample)),
        squeeze=False
    )

    for row, (_, item) in enumerate(sample.iterrows()):
        path = C.RAW_DIR / item["filepath"]

        original = process_image(path, filtered=False)
        filtered = process_image(path, filtered=True)

        axes[row, 0].imshow(original)
        axes[row, 0].set_title("Resized original")

        axes[row, 1].imshow(filtered)
        axes[row, 1].set_title("Median filtered")

        # Visualize the difference between the two versions.
        difference = np.abs(
            np.asarray(original).astype(np.int16)
            - np.asarray(filtered).astype(np.int16)
        )
        axes[row, 2].imshow(difference, cmap="gray")
        axes[row, 2].set_title("Absolute difference")

        for ax in axes[row]:
            ax.axis("off")

    fig.tight_layout()
    output = C.RESULTS_DIR / "preprocessing_samples.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output)


def build(version, df):
    """Create one processed dataset version using the existing manifest."""
    output_root = C.PROCESSED[version]
    output_root.mkdir(parents=True, exist_ok=True)

    use_filter = version == "filtered"

    for index, row in df.iterrows():
        relative_path = str(row["filepath"])
        class_name = str(row["class"])
        split = str(row["split"])

        source = C.RAW_DIR / relative_path

        if not source.is_file():
            raise FileNotFoundError(
                f"Image listed in manifest was not found: {source}"
            )

        if split not in C.SPLITS:
            raise ValueError(f"Unexpected split: {split}")

        destination_dir = output_root / split / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        output_path = destination_dir / f"{index:05d}.png"

        processed = process_image(source, filtered=use_filter)
        processed.save(output_path, format="PNG")

        if (index + 1) % 500 == 0:
            print(f"{version}: processed {index + 1}/{len(df)} images")

    # Verify that every expected class folder exists in every split.
    classes = set(df["class"].astype(str))

    for split in C.SPLITS:
        split_dir = output_root / split
        found_classes = {
            p.name for p in split_dir.iterdir() if p.is_dir()
        } if split_dir.exists() else set()

        missing = classes - found_classes

        if missing:
            raise RuntimeError(
                f"{version}/{split} is missing class folders: {missing}"
            )

    print(f"{version}: completed successfully.")


def main():
    if not C.MANIFEST.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {C.MANIFEST}"
        )

    df = pd.read_csv(C.MANIFEST, dtype={"class": str})

    required = {"filepath", "class", "split"}
    missing_columns = required - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest is missing columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("The manifest contains no images.")

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Images in manifest:", len(df))
    print("Classes:", sorted(df["class"].unique()))
    print("Images per split:")
    print(df["split"].value_counts())

    # Create both versions without changing the split assignments.
    build("original", df)
    build("filtered", df)

    save_preprocessing_samples(df)

    print("\nPreprocessing completed.")
    print("Original folder:", C.PROCESSED["original"])
    print("Filtered folder:", C.PROCESSED["filtered"])


if __name__ == "__main__":
    main()
