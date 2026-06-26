"""Tests for classifier pipeline adapter."""

from __future__ import annotations

from pathlib import Path

from bioimage_pipeline.classifier_pipeline_adapter import materialize_classifier_pipeline


def test_materialize_classifier_pipeline_patches_threshold(tmp_path: Path) -> None:
    output = tmp_path / "working.cppipe"
    path = materialize_classifier_pipeline(
        output,
        measurement_mode="probability_map",
        probability_threshold=0.6,
        object_diameter_min=4,
        object_diameter_max=10,
    )
    text = path.read_text(encoding="utf-8")
    assert "Threshold correction factor:0.60" in text
    assert "Typical diameter of objects, in pixel units (Min,Max):4,10" in text
