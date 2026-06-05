"""Tests for object segmentation."""

import numpy as np
import pytest

from bioimage_pipeline.segment import (
    clean_mask,
    clear_border_objects,
    distance_transform,
    fill_holes,
    label_objects,
    remove_small_objects_from_mask,
    split_touching_objects,
)


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


def test_fill_holes_fills_enclosed_background() -> None:
    mask = np.zeros((12, 12), dtype=bool)
    mask[2:10, 2:10] = True
    mask[4:8, 4:8] = False

    filled = fill_holes(mask)

    assert filled[2:10, 2:10].all()
    assert filled[5, 5]


def test_fill_holes_does_not_modify_input() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:7, 1:7] = True
    mask[3:5, 3:5] = False
    original = mask.copy()

    fill_holes(mask)

    np.testing.assert_array_equal(mask, original)


def test_fill_holes_non_2d_raises_value_error() -> None:
    mask = np.zeros((3, 3, 3), dtype=bool)

    with pytest.raises(ValueError, match="2D"):
        fill_holes(mask)


def test_clear_border_objects_removes_edge_connected_objects() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:3, 0:3] = True
    mask[4:7, 4:7] = True

    cleared = clear_border_objects(mask)

    assert not cleared[0:3, 0:3].any()
    assert cleared[4:7, 4:7].all()


def test_clear_border_objects_does_not_modify_input() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[0, :] = True
    mask[3:6, 3:6] = True
    original = mask.copy()

    clear_border_objects(mask)

    np.testing.assert_array_equal(mask, original)


def test_clear_border_objects_non_2d_raises_value_error() -> None:
    mask = np.zeros((3, 3, 3), dtype=bool)

    with pytest.raises(ValueError, match="2D"):
        clear_border_objects(mask)


def test_clean_mask_applies_fill_remove_small_and_clear_border() -> None:
    mask = np.zeros((14, 14), dtype=bool)
    mask[0:3, 0:3] = True
    mask[5:11, 5:11] = True
    mask[6:10, 6:10] = False

    cleaned = clean_mask(mask, min_size=4)

    assert not cleaned[0:3, 0:3].any()
    assert cleaned[5:11, 5:11].all()
    assert cleaned[6:10, 6:10].all()


def test_clean_mask_preserves_well_separated_interior_objects() -> None:
    mask = np.zeros((12, 12), dtype=bool)
    mask[3:6, 3:6] = True
    mask[7:10, 7:10] = True

    cleaned = clean_mask(mask, min_size=4)

    np.testing.assert_array_equal(cleaned, mask)


def _circular_mask(
    shape: tuple[int, int],
    center_y: int,
    center_x: int,
    radius: int,
) -> np.ndarray:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    circle = (rows - center_y) ** 2 + (cols - center_x) ** 2 <= radius**2
    return circle.astype(bool)


def test_distance_transform_matches_mask_shape_and_is_nonnegative() -> None:
    mask = _circular_mask((25, 25), 12, 12, 6)
    distances = distance_transform(mask)

    assert distances.shape == mask.shape
    assert distances.dtype == np.float64
    assert distances.min() >= 0.0
    assert (distances[~mask] == 0).all()


def test_distance_transform_peak_is_near_disk_center() -> None:
    mask = _circular_mask((31, 31), 15, 15, 8)
    distances = distance_transform(mask)

    peak_y, peak_x = np.unravel_index(np.argmax(distances), distances.shape)
    assert peak_y == 15
    assert peak_x == 15
    assert distances[15, 15] == pytest.approx(distances.max())


def test_distance_transform_touching_disks_have_two_peaks() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask |= _circular_mask((40, 40), 20, 12, 8)
    mask |= _circular_mask((40, 40), 20, 28, 8)

    distances = distance_transform(mask)
    foreground = distances[mask]
    threshold = foreground.max() * 0.8
    peaks = (distances >= threshold) & mask

    assert label_objects(peaks).max() == 2


def test_distance_transform_does_not_modify_input() -> None:
    mask = _circular_mask((20, 20), 10, 10, 5)
    original = mask.copy()

    distance_transform(mask)

    np.testing.assert_array_equal(mask, original)


def test_distance_transform_non_2d_raises_value_error() -> None:
    mask = np.zeros((3, 3, 3), dtype=bool)

    with pytest.raises(ValueError, match="2D"):
        distance_transform(mask)


def _touching_disks_mask() -> np.ndarray:
    mask = np.zeros((40, 40), dtype=bool)
    mask |= _circular_mask((40, 40), 20, 12, 8)
    mask |= _circular_mask((40, 40), 20, 28, 8)
    return mask


def test_split_touching_objects_separates_touching_disks() -> None:
    mask = _touching_disks_mask()

    connected = label_objects(mask)
    split = split_touching_objects(mask)

    assert connected.max() == 1
    assert split.max() == 2


def test_split_touching_objects_preserves_separated_disks() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask |= _circular_mask((40, 40), 12, 12, 7)
    mask |= _circular_mask((40, 40), 28, 28, 7)

    labels = split_touching_objects(mask)

    assert labels.max() == 2
    assert (labels > 0).sum() == mask.sum()


def test_split_touching_objects_single_disk_returns_one_label() -> None:
    mask = _circular_mask((25, 25), 12, 12, 6)

    labels = split_touching_objects(mask)

    assert labels.max() == 1
    assert (labels > 0).sum() == mask.sum()


def test_split_touching_objects_empty_mask_returns_background() -> None:
    mask = np.zeros((10, 10), dtype=bool)

    labels = split_touching_objects(mask)

    assert labels.max() == 0


def test_split_touching_objects_does_not_modify_input() -> None:
    mask = _touching_disks_mask()
    original = mask.copy()

    split_touching_objects(mask)

    np.testing.assert_array_equal(mask, original)


def test_split_touching_objects_non_2d_raises_value_error() -> None:
    mask = np.zeros((3, 3, 3), dtype=bool)

    with pytest.raises(ValueError, match="2D"):
        split_touching_objects(mask)
