
"""Member 1: Extract controlled image regions for shortcut experiments.

Conditions:
1. original        - unchanged source image
2. defect_only     - pixels inside the expert defect mask
3. background_only - pixels outside the expert defect mask
4. artifact_focused - pixels inside a separately supplied artifact mask

Masked-out pixels are black. Region extraction is performed before the
project's normal preprocessing and resizing.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(mask_path):
    """Load a mask as a grayscale NumPy array."""
    mask_path = Path(mask_path)

    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    with Image.open(mask_path) as mask_image:
        return np.asarray(mask_image.convert("L"))


def load_rgb_image(image_path):
    """Load an image as RGB without resizing it."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        return image.convert("RGB")


def validate_region_mask(mask, image_size, mask_name="region"):
    """Validate mask dimensions and return a binary foreground mask."""
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(
            f"{mask_name} must be a 2D mask; got shape {mask.shape}"
        )

    expected_width, expected_height = image_size

    if mask.shape != (expected_height, expected_width):
        raise ValueError(
            f"{mask_name} dimensions {mask.shape} do not match "
            f"image dimensions {(expected_height, expected_width)}"
        )

    # Provisional convention: nonzero pixels belong to the region.
    return mask > 0


def keep_region(image_array, region_mask):
    """Keep pixels inside a region and set everything else to black."""
    if image_array.shape[:2] != region_mask.shape:
        raise ValueError("Image and region mask dimensions differ.")

    output = np.zeros_like(image_array)
    output[region_mask] = image_array[region_mask]

    return output


def extract_regions(
    image_path,
    defect_mask_path,
    artifact_mask_path=None,
):
    """
    Extract original, defect-only, and background-only images.

    If artifact_mask_path is supplied, also extract an artifact-focused
    image using that separate mask.

    Returns a dictionary of PIL RGB images.
    """
    original = load_rgb_image(image_path)
    image_array = np.asarray(original)

    defect_mask = load_mask(defect_mask_path)
    defect_region = validate_region_mask(
        defect_mask,
        original.size,
        mask_name="Defect mask",
    )

    regions = {
        "original": original,
        "defect_only": Image.fromarray(
            keep_region(image_array, defect_region)
        ),
        "background_only": Image.fromarray(
            keep_region(image_array, ~defect_region)
        ),
    }

    if artifact_mask_path is not None:
        artifact_mask = load_mask(artifact_mask_path)
        artifact_region = validate_region_mask(
            artifact_mask,
            original.size,
            mask_name="Artifact mask",
        )

        regions["artifact_focused"] = Image.fromarray(
            keep_region(image_array, artifact_region)
        )

    return regions


def save_regions(regions, output_dir, record_id):
    """Save extracted conditions as PNG files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    for condition, image in regions.items():
        output_path = output_dir / f"{record_id}_{condition}.png"
        image.save(output_path, format="PNG")
        saved_paths[condition] = output_path

    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Extract SEM image regions using expert masks."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Source SEM image",
    )
    parser.add_argument(
        "--mask",
        required=True,
        help="Expert defect mask",
    )
    parser.add_argument(
        "--artifact-mask",
        default=None,
        help="Optional separate artifact annotation mask",
    )
    parser.add_argument(
        "--output-dir",
        default="stage1_data_pipeline/region_examples",
        help="Directory for extracted PNG files",
    )
    parser.add_argument(
        "--record-id",
        default=None,
        help="Output filename prefix; defaults to image filename",
    )

    args = parser.parse_args()

    record_id = args.record_id or Path(args.image).stem

    regions = extract_regions(
        image_path=args.image,
        defect_mask_path=args.mask,
        artifact_mask_path=args.artifact_mask,
    )

    saved_paths = save_regions(
        regions,
        output_dir=args.output_dir,
        record_id=record_id,
    )

    print("Region extraction completed.")
    for condition, path in saved_paths.items():
        print(f"{condition}: {path}")

    if "artifact_focused" not in regions:
        print(
            "Artifact-focused image not generated: "
            "no separate artifact mask was supplied."
        )


if __name__ == "__main__":
    main()
