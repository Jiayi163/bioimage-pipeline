"""Object segmentation helpers."""

import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from skimage.feature import peak_local_max
from skimage.segmentation import clear_border, watershed


def _as_2d_bool_mask(mask: np.ndarray) -> np.ndarray:
    mask_arr = np.asarray(mask).astype(bool)
    if mask_arr.ndim != 2:
        raise ValueError("Only 2D masks are supported")
    return mask_arr


def remove_small_objects_from_mask(mask: np.ndarray, min_size: int = 20) -> np.ndarray:
    """Remove small connected components from a binary mask.

    Args:
        mask: Binary mask array.
        min_size: Minimum object size in pixels to keep.

    Returns:
        Cleaned boolean mask with the same shape as the input.
    """
    mask_arr = _as_2d_bool_mask(mask)
    cleaned = morphology.remove_small_objects(mask_arr, min_size=max(1, min_size))
    return cleaned.astype(bool)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill internal holes in binary mask objects.

    Args:
        mask: Binary mask array (2D).

    Returns:
        Boolean mask with enclosed background regions filled.
    """
    mask_arr = _as_2d_bool_mask(mask)
    filled = ndimage.binary_fill_holes(mask_arr)
    return filled.astype(bool)


def clear_border_objects(mask: np.ndarray) -> np.ndarray:
    """Remove objects that touch the image border.

    Args:
        mask: Binary mask array (2D).

    Returns:
        Boolean mask with border-connected objects removed.
    """
    mask_arr = _as_2d_bool_mask(mask)
    labels = measure.label(mask_arr)
    cleared = clear_border(labels)
    return (cleared > 0).astype(bool)


def clean_mask(
    mask: np.ndarray,
    *,
    min_size: int = 20,
    fill_holes_in_mask: bool = True,
    clear_border: bool = True,
) -> np.ndarray:
    """Apply standard morphological cleanup to a binary mask.

    Steps (when enabled): fill holes, remove small objects, clear border objects.
    """
    cleaned = _as_2d_bool_mask(mask)

    if fill_holes_in_mask:
        cleaned = fill_holes(cleaned)
    cleaned = remove_small_objects_from_mask(cleaned, min_size=min_size)
    if clear_border:
        cleaned = clear_border_objects(cleaned)

    return cleaned.astype(bool)


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """Compute the Euclidean distance transform of a binary mask.

    For each foreground pixel, returns the distance to the nearest background
    pixel. Background pixels are zero. Peaks occur near object centers and are
    used as seeds for watershed splitting (Phase 11.3).

    Args:
        mask: Binary mask array (2D).

    Returns:
        Float array with the same shape as the mask.
    """
    mask_arr = _as_2d_bool_mask(mask)
    return ndimage.distance_transform_edt(mask_arr)


def split_touching_objects(
    mask: np.ndarray,
    *,
    min_distance: int = 8,
    min_peak_ratio: float = 0.5,
) -> np.ndarray:
    """Label objects in a binary mask, splitting touching components.

    Uses a distance transform and watershed segmentation with seeds at
    distance peaks. Well-separated objects receive one label each; touching
    objects are split when multiple peaks are detected.

    Args:
        mask: Binary mask array (2D).
        min_distance: Minimum spacing between watershed seed peaks.
        min_peak_ratio: Seed threshold as a fraction of the maximum distance.

    Returns:
        Integer label image where background is 0.
    """
    mask_arr = _as_2d_bool_mask(mask)
    if not mask_arr.any():
        return np.zeros(mask_arr.shape, dtype=np.int32)

    distances = distance_transform(mask_arr)
    max_distance = float(distances.max())
    threshold_abs = max_distance * min_peak_ratio if max_distance > 0 else None

    coordinates = peak_local_max(
        distances,
        labels=mask_arr,
        min_distance=max(1, min_distance),
        threshold_abs=threshold_abs,
        exclude_border=False,
    )

    if coordinates.size == 0:
        return label_objects(mask_arr)

    markers = np.zeros(mask_arr.shape, dtype=np.int32)
    for index, (row, col) in enumerate(coordinates, start=1):
        markers[int(row), int(col)] = index

    labels = watershed(-distances, markers, mask=mask_arr)
    return labels.astype(np.int32)


def label_objects(mask: np.ndarray) -> np.ndarray:
    """Label connected objects in a binary mask.

    Args:
        mask: Binary mask array (2D).

    Returns:
        Integer label image where background is 0.

    Raises:
        ValueError: If the mask is not 2D.
    """
    mask_arr = _as_2d_bool_mask(mask)
    return measure.label(mask_arr).astype(np.int32)
