"""Tests for image preprocessing operations."""

import numpy as np
import pytest

from bioimage_pipeline.preprocess import (
    gaussian_blur,
    median_filter_image,
    normalize_image,
    rolling_ball_subtract,
)


def test_gaussian_blur_preserves_shape_and_input() -> None:
    image = np.zeros((5, 5), dtype=np.float32)
    image[2, 2] = 1
    original = image.copy()

    blurred = gaussian_blur(image, sigma=1)

    assert blurred.shape == image.shape
    np.testing.assert_array_equal(image, original)


def test_gaussian_blur_works_on_simple_array() -> None:
    image = np.zeros((5, 5), dtype=np.float32)
    image[2, 2] = 1

    blurred = gaussian_blur(image, sigma=1)

    assert blurred[2, 2] < image[2, 2]
    assert blurred[2, 2] > 0
    assert blurred[2, 1] > 0


def test_gaussian_blur_invalid_sigma_raises_value_error() -> None:
    image = np.zeros((3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="sigma"):
        gaussian_blur(image, sigma=-1)


def test_median_filter_preserves_shape_and_input() -> None:
    image = np.zeros((5, 5), dtype=np.uint16)
    image[2, 2] = 255
    original = image.copy()

    filtered = median_filter_image(image, radius=1)

    assert filtered.shape == image.shape
    np.testing.assert_array_equal(image, original)


def test_median_filter_removes_isolated_bright_noise_pixel() -> None:
    image = np.zeros((5, 5), dtype=np.uint16)
    image[2, 2] = 255

    filtered = median_filter_image(image, radius=1)

    assert filtered[2, 2] == 0


def test_median_filter_invalid_radius_raises_value_error() -> None:
    image = np.zeros((3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="radius"):
        median_filter_image(image, radius=-1)


def test_normalize_image_preserves_shape_and_input() -> None:
    image = np.array([[0, 5], [10, 15]], dtype=np.uint16)
    original = image.copy()

    normalized = normalize_image(image)

    assert normalized.shape == image.shape
    np.testing.assert_array_equal(image, original)


def test_normalize_image_returns_values_between_zero_and_one() -> None:
    image = np.array([[10, 20], [30, 40]], dtype=np.uint16)

    normalized = normalize_image(image)

    assert normalized.dtype.kind == "f"
    assert normalized.min() == 0
    assert normalized.max() == 1


def test_normalize_image_handles_constant_images() -> None:
    image = np.full((4, 4), 7, dtype=np.uint8)

    normalized = normalize_image(image)

    np.testing.assert_array_equal(normalized, np.zeros((4, 4), dtype=float))


def test_rolling_ball_subtract_reduces_vignette() -> None:
    image = np.full((128, 128), 200, dtype=np.uint16)
    image[:30, :] = 80
    corrected = rolling_ball_subtract(image, radius=12)
    assert corrected.dtype == np.float32
    assert float(corrected[10, 64]) > float(corrected[64, 64]) - 50
