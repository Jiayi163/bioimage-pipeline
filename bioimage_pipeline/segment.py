"""Object segmentation helpers."""

import numpy as np
from skimage import measure, morphology


def remove_small_objects_from_mask(mask: np.ndarray, min_size: int = 20) -> np.ndarray:
    """Remove small connected components from a binary mask.

    Args:
        mask: Binary mask array.
        min_size: Minimum object size in pixels to keep.

    Returns:
        Cleaned boolean mask with the same shape as the input.
    """
    mask_arr = np.asarray(mask).astype(bool)
    max_size = max(0, min_size - 1)
    cleaned = morphology.remove_small_objects(mask_arr, max_size=max_size)
    return cleaned.astype(bool)


def label_objects(mask: np.ndarray) -> np.ndarray:
    """Label connected objects in a binary mask.

    Args:
        mask: Binary mask array (2D).

    Returns:
        Integer label image where background is 0.

    Raises:
        ValueError: If the mask is not 2D.
    """
    mask_arr = np.asarray(mask).astype(bool)
    if mask_arr.ndim != 2:
        raise ValueError("Only 2D masks are supported")

    return measure.label(mask_arr).astype(np.int32)
