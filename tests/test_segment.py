"""Tests for object segmentation."""

import numpy as np
import pytest

from bioimage_pipeline.segment import label_objects, remove_small_objects_from_mask


def test_remove_small_objects_removes_noise() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2, 2] = True
    mask[5:8, 5:8] = True

    cleaned = remove_small_objects_from_mask(mask, min_size=5)

    assert cleaned.shape == mask.shape
    assert not cleaned[2, 2]
    assert cleaned[5:8, 5:8].all()


def test_remove_small_objects_does_not_modify_input() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    original = mask.copy()

    remove_small_objects_from_mask(mask, min_size=2)

    np.testing.assert_array_equal(mask, original)


def test_label_objects_background_is_zero() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:4, 2:4] = True
    mask[5:7, 5:7] = True

    labels = label_objects(mask)

    assert labels.dtype == np.int32
    assert labels.shape == mask.shape
    assert labels.min() == 0
    assert labels.max() == 2
    assert set(np.unique(labels)) == {0, 1, 2}


def test_label_objects_assigns_unique_labels() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[6:9, 6:9] = True

    labels = label_objects(mask)

    object_labels = labels[labels > 0]
    assert len(np.unique(object_labels)) == 2


def test_label_objects_non_2d_raises_value_error() -> None:
    mask = np.zeros((3, 3, 3), dtype=bool)

    with pytest.raises(ValueError, match="2D"):
        label_objects(mask)
