
from pathlib import Path

import numpy as np
from PIL import Image


def read_mask(mask_path):
    """Read a grayscale or single-channel mask without altering its values."""
    mask_path = Path(mask_path)

    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    with Image.open(mask_path) as mask_image:
        mask_array = np.asarray(mask_image)

    if mask_array.ndim == 3:
        # Accept RGB/RGBA only if all colour channels are identical.
        colour_channels = mask_array[..., :3]

        if not np.all(
            colour_channels == colour_channels[..., :1]
        ):
            raise ValueError(
                f"Mask has different colour-channel values: {mask_path}"
            )

        mask_array = colour_channels[..., 0]

    if mask_array.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask, got shape {mask_array.shape}: {mask_path}"
        )

    unique_values = np.unique(mask_array)

    # Provisional interpretation: every nonzero pixel is foreground.
    # Preserve the original grayscale values for auditing.
    foreground = mask_array > 0

    return mask_array, foreground, unique_values


def validate_image_mask(image_path, mask_path):
    """Check image/mask readability, dimensions, and mask characteristics."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as image:
        image.verify()

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
        "mask_unique_value_count": len(unique_values),
        "mask_is_empty": not foreground.any(),
        "mask_is_full": foreground.all(),
        "mask_has_intermediate_values": np.any(
            (mask_array > 0) & (mask_array < 255)
        ),
        "foreground_pixels": int(foreground.sum()),
        "foreground_fraction": float(foreground.mean()),
    }


def keep_region(image, region_mask):
    """Keep pixels inside a Boolean region and zero all pixels outside it."""
    image = np.asarray(image)
    region_mask = np.asarray(region_mask, dtype=bool)

    if image.ndim not in (2, 3):
        raise ValueError(
            f"Expected a 2D or 3D image, got shape {image.shape}"
        )

    if image.shape[:2] != region_mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch: {image.shape[:2]} "
            f"vs {region_mask.shape}"
        )

    output = np.zeros_like(image)
    output[region_mask] = image[region_mask]
    return output


def defect_only(image, defect_mask):
    """Keep the pixels currently interpreted as foreground."""
    return keep_region(image, defect_mask)


def background_only(image, defect_mask):
    """Keep pixels currently interpreted as background."""
    return keep_region(image, ~np.asarray(defect_mask, dtype=bool))
