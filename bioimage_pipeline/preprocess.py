"""Image preprocessing operations."""

import numpy as np
from scipy import ndimage


def gaussian_blur(image: np.ndarray, sigma: float = 1) -> np.ndarray:
    """Apply a Gaussian blur to an image.

    Args:
        image: Input image array.
        sigma: Standard deviation for the Gaussian kernel.

    Returns:
        A blurred image with the same shape as the input.

    Raises:
        ValueError: If sigma is negative.
    """
    if sigma < 0:
        raise ValueError("sigma must be greater than or equal to 0")

    return ndimage.gaussian_filter(np.asarray(image), sigma=sigma)


def median_filter_image(image: np.ndarray, radius: int = 1) -> np.ndarray:
    """Apply a median filter to an image.

    Args:
        image: Input image array.
        radius: Radius of the median filter window.

    Returns:
        A median-filtered image with the same shape as the input.

    Raises:
        ValueError: If radius is negative.
    """
    if radius < 0:
        raise ValueError("radius must be greater than or equal to 0")

    size = 2 * radius + 1
    return ndimage.median_filter(np.asarray(image), size=size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Scale image intensities to the range [0, 1].

    Args:
        image: Input image array.

    Returns:
        A floating-point image with values between 0 and 1.
    """
    image_float = np.asarray(image, dtype=float)
    min_value = image_float.min()
    max_value = image_float.max()

    if min_value == max_value:
        return np.zeros_like(image_float)

    return (image_float - min_value) / (max_value - min_value)
