"""Thresholding algorithms."""

import numpy as np
from skimage import filters


def manual_threshold(image: np.ndarray, value: float) -> np.ndarray:
    """Create a binary mask using a fixed intensity threshold.

    Args:
        image: Input image array.
        value: Pixels at or above this value are True in the mask.

    Returns:
        A boolean mask with the same shape as the input.
    """
    image_arr = np.asarray(image)
    return (image_arr >= value).astype(bool)


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Create a binary mask using Otsu's automatic threshold.

    Args:
        image: Input image array.

    Returns:
        A boolean mask with the same shape as the input.
    """
    image_arr = np.asarray(image)
    threshold = filters.threshold_otsu(image_arr)
    return (image_arr > threshold).astype(bool)


def adaptive_threshold(
    image: np.ndarray,
    block_size: int = 51,
    offset: float = 0,
) -> np.ndarray:
    """Create a binary mask using local adaptive thresholding.

    Args:
        image: Input image array.
        block_size: Odd window size for the local neighborhood.
        offset: Constant subtracted from the local threshold.

    Returns:
        A boolean mask with the same shape as the input.

    Raises:
        ValueError: If block_size is not a positive odd integer.
    """
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    if block_size % 2 == 0:
        raise ValueError("block_size must be odd")

    image_arr = np.asarray(image)
    local_thresh = filters.threshold_local(
        image_arr,
        block_size=block_size,
        offset=offset,
    )
    return (image_arr > local_thresh).astype(bool)


def sauvola_threshold(
    image: np.ndarray,
    *,
    block_size: int = 51,
    k: float = 0.2,
    r: float | None = None,
) -> np.ndarray:
    """Create a binary mask using Sauvola local thresholding.

    Args:
        image: Input image array.
        block_size: Odd window size for the local neighborhood.
        k: Sauvola sensitivity parameter.
        r: Dynamic range of the standard deviation. Defaults to the image max
            when ``None``.

    Returns:
        A boolean mask with the same shape as the input.

    Raises:
        ValueError: If block_size is not a positive odd integer.
    """
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    if block_size % 2 == 0:
        raise ValueError("block_size must be odd")

    image_arr = np.asarray(image)
    dynamic_range = float(image_arr.max()) if r is None else r
    if dynamic_range <= 0:
        dynamic_range = 1.0

    local_thresh = filters.threshold_sauvola(
        image_arr,
        window_size=block_size,
        k=k,
        r=dynamic_range,
    )
    return (image_arr > local_thresh).astype(bool)
