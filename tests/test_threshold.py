"""Tests for thresholding operations."""

import numpy as np
import pytest

from bioimage_pipeline.threshold import (
    adaptive_threshold,
    manual_threshold,
    otsu_threshold,
)


def test_manual_threshold_returns_bool_mask_with_same_shape() -> None:
    image = np.array([[0, 50], [100, 150]], dtype=np.uint8)

    mask = manual_threshold(image, value=75)

    assert mask.dtype == bool
    assert mask.shape == image.shape
    np.testing.assert_array_equal(mask, [[False, False], [True, True]])


def test_manual_threshold_does_not_modify_input() -> None:
    image = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    original = image.copy()

    manual_threshold(image, value=25)

    np.testing.assert_array_equal(image, original)


def test_otsu_threshold_returns_bool_mask_with_same_shape() -> None:
    image = np.zeros((20, 40), dtype=np.uint8)
    image[:, 20:] = 200

    mask = otsu_threshold(image)

    assert mask.dtype == bool
    assert mask.shape == image.shape
    assert mask[:, 20:].all()
    assert not mask[:, :20].any()


def test_otsu_threshold_on_simple_bimodal_array() -> None:
    image = np.array([10, 10, 10, 10, 200, 200, 200, 200], dtype=np.uint8)

    mask = otsu_threshold(image)

    np.testing.assert_array_equal(mask, [False, False, False, False, True, True, True, True])


def test_adaptive_threshold_returns_bool_mask_with_same_shape() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)
    image[5:10, 5:10] = 120
    image[12:17, 12:17] = 40

    mask = adaptive_threshold(image, block_size=11, offset=0)

    assert mask.dtype == bool
    assert mask.shape == image.shape


def test_adaptive_threshold_even_block_size_raises_value_error() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="odd"):
        adaptive_threshold(image, block_size=50)


def test_adaptive_threshold_invalid_block_size_raises_value_error() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="at least 1"):
        adaptive_threshold(image, block_size=0)
