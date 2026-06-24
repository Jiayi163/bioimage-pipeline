"""Tests for object-level segmentation validation metrics."""

from __future__ import annotations

import numpy as np

from bioimage_pipeline.validation import compare_objects, compare_segmentation


def _disk_mask(center: tuple[int, int], radius: int, shape: tuple[int, int]) -> np.ndarray:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    row, col = center
    return ((rows - row) ** 2 + (cols - col) ** 2) <= radius**2


def test_compare_objects_perfect_match() -> None:
    mask = _disk_mask((20, 20), 5, (64, 64))
    result = compare_objects(mask, mask, match_iou_threshold=0.3)

    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_compare_objects_detects_false_positive() -> None:
    reference = _disk_mask((20, 20), 5, (64, 64))
    predicted = reference.copy()
    predicted |= _disk_mask((40, 40), 4, (64, 64))

    result = compare_objects(predicted, reference, match_iou_threshold=0.3)

    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 0
    assert result.precision == 0.5
    assert result.recall == 1.0


def test_compare_objects_detects_false_negative() -> None:
    reference = _disk_mask((20, 20), 5, (64, 64)) | _disk_mask((40, 40), 4, (64, 64))
    predicted = _disk_mask((20, 20), 5, (64, 64))

    result = compare_objects(predicted, reference, match_iou_threshold=0.3)

    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 1
    assert result.recall == 0.5


def test_compare_objects_over_detection_increases_false_positives() -> None:
    reference = _disk_mask((32, 32), 6, (64, 64))
    predicted = reference | _disk_mask((10, 10), 3, (64, 64)) | _disk_mask((50, 50), 3, (64, 64))

    result = compare_objects(predicted, reference, match_iou_threshold=0.3)

    assert result.reference_object_count == 1
    assert result.predicted_object_count == 3
    assert result.false_positives >= 2


def test_compare_segmentation_includes_pixel_and_object_metrics() -> None:
    reference = _disk_mask((24, 24), 6, (64, 64))
    predicted = reference.copy()

    comparison = compare_segmentation(predicted, reference)

    assert comparison.pixel.iou == 1.0
    assert comparison.object_level.f1 == 1.0
