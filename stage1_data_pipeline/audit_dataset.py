
"""Member 1: audit Carinthia-S images, expert masks, and dataset integrity."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "dataset" / "carinthia_s" / "raw"
CSV_PATH = DATA_ROOT / "carinthia-s.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = OUTPUT_DIR / "examples"
BASELINE_MANIFEST = ROOT / "dataset" / "manifest.csv"

REQUIRED_COLUMNS = {"image_path", "mask_path", "filename", "label"}


def resolve_dataset_path(value):
    """Resolve a metadata path relative to the raw dataset directory."""
    if pd.isna(value) or not str(value).strip():
        raise ValueError("Dataset path is empty.")

    relative_path = Path(str(value).strip())

    if relative_path.is_absolute():
        raise ValueError(f"Absolute dataset paths are not allowed: {value}")

    # Metadata normally uses images/... and masks/...
    candidate = (DATA_ROOT / relative_path).resolve()
    data_root_resolved = DATA_ROOT.resolve()

    if not candidate.is_relative_to(data_root_resolved):
        raise ValueError(f"Path escapes dataset directory: {value}")

    if not candidate.is_file():
        raise FileNotFoundError(f"File does not exist: {candidate}")

    return candidate


def md5_file(path):
    """Calculate an MD5 checksum for exact duplicate detection."""
    digest = hashlib.md5()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_mask(mask_path):
    """
    Read a mask without changing its pixel values.

    Provisional interpretation:
    nonzero pixels are foreground. Verify this convention against the
    dataset documentation before scientific region-based experiments.
    """
    mask_path = Path(mask_path)

    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    with Image.open(mask_path) as mask_image:
        mask_array = np.asarray(mask_image)

    if mask_array.ndim == 3:
        # RGB/RGBA masks are accepted only if RGB channels agree.
        rgb = mask_array[..., :3]

        if not np.all(rgb == rgb[..., :1]):
            raise ValueError(
                f"Mask has different RGB channel values: {mask_path}"
            )

        mask_array = rgb[..., 0]

    if mask_array.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask; got shape {mask_array.shape}: {mask_path}"
        )

    unique_values = np.unique(mask_array)
    foreground = mask_array > 0

    return mask_array, foreground, unique_values


def validate_image_mask(image_path, mask_path):
    """Check readability, dimensions, and basic mask characteristics."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Verify the image file can be decoded.
    with Image.open(image_path) as image:
        image.verify()

    # Reopen after verify() to obtain dimensions.
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    mask_array, foreground, unique_values = read_mask(mask_path)
    mask_height, mask_width = mask_array.shape

    return {
        "image_width": image_width,
        "image_height": image_height,
        "mask_width": mask_width,
        "mask_height": mask_height,
        "aligned": (
            image_width == mask_width
            and image_height == mask_height
        ),
        "mask_values": ",".join(
            str(int(value)) for value in unique_values
        ),
        "mask_unique_value_count": int(len(unique_values)),
        "mask_is_empty": bool(not foreground.any()),
        "mask_is_full": bool(foreground.all()),
        "mask_has_intermediate_values": bool(
            np.any((mask_array > 0) & (mask_array < 255))
        ),
        "foreground_pixels": int(foreground.sum()),
        "foreground_fraction": float(foreground.mean()),
    }


def make_overlay(image_path, mask_path, output_path):
    """Save original image, binary visualization, and red overlay side by side."""
    with Image.open(image_path) as source:
        original = source.convert("RGB")

    _, foreground, _ = read_mask(mask_path)

    if original.size != (
        foreground.shape[1],
        foreground.shape[0],
    ):
        raise ValueError(
            f"Image-mask dimensions differ for {image_path.name}"
        )

    mask_visual = Image.fromarray(
        (foreground.astype(np.uint8) * 255)
    ).convert("RGB")

    image_array = np.asarray(original).copy()
    overlay_array = image_array.copy()

    # Highlight provisionally identified foreground pixels in red.
    overlay_array[foreground] = (
        0.5 * overlay_array[foreground]
        + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    overlay = Image.fromarray(overlay_array)

    width, height = original.size
    label_height = 24

    canvas = Image.new(
        "RGB",
        (width * 3, height + label_height),
        "white",
    )

    draw = ImageDraw.Draw(canvas)
    panels = [
        (original, "Original"),
        (mask_visual, "Mask (nonzero pixels)"),
        (overlay, "Red overlay"),
    ]

    for index, (panel, label) in enumerate(panels):
        x = index * width
        canvas.paste(panel, (x, label_height))
        draw.text((x + 6, 5), label, fill="black")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def load_baseline_splits():
    """Map original image IDs to the existing baseline train/val/test splits."""
    if not BASELINE_MANIFEST.is_file():
        print(
            "WARNING: Baseline manifest not found. "
            "Baseline splits will be blank."
        )
        return {}

    baseline = pd.read_csv(BASELINE_MANIFEST)

    required = {"filepath", "split"}
    if not required.issubset(baseline.columns):
        print(
            "WARNING: Baseline manifest lacks filepath/split columns. "
            "Baseline splits will be blank."
        )
        return {}

    split_map = {}

    for _, row in baseline.iterrows():
        image_id = Path(str(row["filepath"])).stem
        split_map[image_id] = str(row["split"]).strip().lower()

    return split_map


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found: {CSV_PATH}"
        )

    # The supplied Carinthia-S metadata CSV uses semicolons.
    df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8-sig")

    # Remove accidental whitespace from column names.
    df.columns = [str(column).strip() for column in df.columns]

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing_columns)}. "
            f"Found columns: {df.columns.tolist()}"
        )

    if df.empty:
        raise ValueError("Metadata CSV contains no records.")

    if df["filename"].isna().any():
        raise ValueError("Metadata contains empty filename values.")

    if df["filename"].astype(str).duplicated().any():
        duplicates = df.loc[
            df["filename"].astype(str).duplicated(keep=False),
            "filename",
        ].astype(str).tolist()

        raise ValueError(
            "Metadata contains duplicate filenames. Examples: "
            + ", ".join(duplicates[:10])
        )

    split_map = load_baseline_splits()

    audit_rows = []
    manifest_rows = []
    seen_hashes = {}
    overlay_candidates = {}

    for row_number, row in df.iterrows():
        filename = str(row["filename"]).strip()
        record_id = Path(filename).stem
        label = row["label"]

        image_path_text = str(row["image_path"]).strip()
        mask_path_text = str(row["mask_path"]).strip()

        audit_record = {
            "row_number": int(row_number) + 2,
            "record_id": record_id,
            "label": label,
            "image_path": image_path_text,
            "mask_path": mask_path_text,
            "image_md5": "",
            "status": "error",
            "error": "",
        }

        try:
            image_path = resolve_dataset_path(image_path_text)
            mask_path = resolve_dataset_path(mask_path_text)

            # This is the validation-result variable used below.
            info = validate_image_mask(image_path, mask_path)

            image_hash = md5_file(image_path)
            duplicate_of = seen_hashes.get(image_hash)

            if duplicate_of is None:
                seen_hashes[image_hash] = record_id

            error_messages = []

            if not info["aligned"]:
                error_messages.append(
                    "Image-mask dimension mismatch"
                )

            audit_record["image_md5"] = image_hash
            audit_record["status"] = (
                "ok" if not error_messages else "error"
            )
            audit_record["error"] = "; ".join(error_messages)

            manifest_record = {
                "record_id": record_id,
                "filename": filename,
                "label": label,
                "image_path": image_path_text,
                "mask_path": mask_path_text,
                "baseline_split": split_map.get(record_id, ""),
                "image_width": info["image_width"],
                "image_height": info["image_height"],
                "mask_width": info["mask_width"],
                "mask_height": info["mask_height"],
                "aligned": info["aligned"],
                "mask_values": info["mask_values"],
                "mask_unique_value_count": (
                    info["mask_unique_value_count"]
                ),
                "mask_is_empty": info["mask_is_empty"],
                "mask_is_full": info["mask_is_full"],
                "mask_has_intermediate_values": (
                    info["mask_has_intermediate_values"]
                ),
                "foreground_pixels": info["foreground_pixels"],
                "foreground_fraction": info["foreground_fraction"],
                "image_md5": image_hash,
                "exact_duplicate_of": duplicate_of or "",
            }

            manifest_rows.append(manifest_record)

            # Retain aligned examples for visual inspection.
            if info["aligned"] and duplicate_of is None:
                label_key = str(label)
                if label_key not in overlay_candidates:
                    overlay_candidates[label_key] = (
                        image_path,
                        mask_path,
                        record_id,
                    )

        except Exception as exc:
            audit_record["status"] = "error"
            audit_record["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        audit_rows.append(audit_record)

    audit_df = pd.DataFrame(audit_rows)
    manifest_df = pd.DataFrame(manifest_rows)

    audit_path = OUTPUT_DIR / "data_audit.csv"
    manifest_path = OUTPUT_DIR / "mask_manifest.csv"
    conflicts_path = OUTPUT_DIR / "duplicate_split_conflicts.csv"

    audit_df.to_csv(audit_path, index=False)
    manifest_df.to_csv(manifest_path, index=False)

    # Find byte-identical images assigned to multiple baseline splits.
    conflict_rows = []

    if not manifest_df.empty:
        known_splits = manifest_df[
            manifest_df["baseline_split"].isin(
                ["train", "val", "test"]
            )
        ]

        for image_hash, group in known_splits.groupby("image_md5"):
            splits = sorted(group["baseline_split"].unique())

            if len(splits) > 1:
                conflict_rows.append({
                    "image_md5": image_hash,
                    "record_ids": ",".join(
                        group["record_id"].astype(str)
                    ),
                    "baseline_splits": ",".join(splits),
                })

    pd.DataFrame(
        conflict_rows,
        columns=[
            "image_md5",
            "record_ids",
            "baseline_splits",
        ],
    ).to_csv(conflicts_path, index=False)

    # Save one representative aligned example per observed class.
    for label_key, (image_path, mask_path, record_id) in (
        overlay_candidates.items()
    ):
        safe_label = "".join(
            char for char in label_key
            if char.isalnum() or char in ("-", "_")
        )

        try:
            make_overlay(
                image_path,
                mask_path,
                EXAMPLES_DIR / f"class_{safe_label}_{record_id}.png",
            )
        except Exception as exc:
            print(
                f"WARNING: Could not create overlay for {record_id}: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n===== DATASET AUDIT SUMMARY =====")
    print(f"CSV records: {len(df)}")
    print("\nAudit status counts:")
    print(audit_df["status"].value_counts().to_string())

    if not manifest_df.empty:
        print("\nImage class distribution:")
        print(
            manifest_df["label"].value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

        print("\nImage-mask dimension checks:")
        print(
            "Aligned image-mask pairs: "
            f"{int(manifest_df['aligned'].sum())}/"
            f"{len(manifest_df)}"
        )

        print("\nMask characteristics:")
        print(
            "Empty masks: "
            f"{int(manifest_df['mask_is_empty'].sum())}"
        )
        print(
            "Full masks: "
            f"{int(manifest_df['mask_is_full'].sum())}"
        )
        print(
            "Masks with intermediate grayscale values: "
            f"{int(manifest_df['mask_has_intermediate_values'].sum())}"
        )

        print("\nBaseline split availability:")
        print(
            manifest_df["baseline_split"]
            .replace("", "(unmatched)")
            .value_counts(dropna=False)
            .to_string()
        )

        print("\nExact duplicate image entries:")
        print(
            int(
                manifest_df["exact_duplicate_of"]
                .astype(str)
                .ne("")
                .sum()
            )
        )
    else:
        print("\nNo readable image-mask pairs were added to the manifest.")

    print(
        "\nDuplicate split conflicts: "
        f"{len(conflict_rows)}"
    )

    print("\nSaved:")
    print(audit_path.resolve())
    print(manifest_path.resolve())
    print(conflicts_path.resolve())
    print(EXAMPLES_DIR.resolve())


if __name__ == "__main__":
    main()
