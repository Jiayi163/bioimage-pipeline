"""Tests for classifier segmentation QC metrics."""

from __future__ import annotations

import numpy as np

from bioimage_pipeline.segmentation_qc import summarize_predicted_mask


def test_summarize_predicted_mask_negative_control() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:8, 5:8] = True
    summary = summarize_predicted_mask(
        mask,
        image_name="noEV_001_zmax_ev_mask.tif",
        is_negative_control=True,
    )
    assert summary.object_count == 1
    assert summary.negative_control_false_positive_count == 1
    assert summary.warnings is not None
