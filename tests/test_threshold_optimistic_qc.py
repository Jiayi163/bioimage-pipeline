"""Tests for fast optimistic threshold QC assessment."""

from __future__ import annotations

from bioimage_pipeline.threshold_optimistic_qc import (
    OptimisticQcConfig,
    assess_optimistic_qc,
    collect_biological_suspicion_warnings,
)
from bioimage_pipeline.threshold_variant_comparison import (
    ThresholdVariantMeasurementSummary,
)


def _summary(**overrides: object) -> ThresholdVariantMeasurementSummary:
    payload = {
        "variant_id": "001_optimistic_otsu_adaptive",
        "display_name": "Optimistic Otsu Adaptive",
        "success": True,
        "object_count": 120,
        "median_area": 18.0,
        "mean_area": 20.0,
        "tiny_frac": 0.05,
        "huge_frac": 0.02,
        "normal_frac": 0.93,
        "median_intensity": 180.0,
    }
    payload.update(overrides)
    return ThresholdVariantMeasurementSummary(**payload)


def test_assess_optimistic_qc_passes_reasonable_candidate() -> None:
    assessment = assess_optimistic_qc(_summary())

    assert assessment.passed is True
    assert assessment.score is not None
    assert assessment.score.score >= 0.5
    assert not assessment.reasons


def test_assess_optimistic_qc_fails_on_zero_objects() -> None:
    assessment = assess_optimistic_qc(_summary(object_count=0))

    assert assessment.passed is False
    assert any("No objects" in reason for reason in assessment.reasons)


def test_assess_optimistic_qc_fails_on_high_tiny_fraction() -> None:
    assessment = assess_optimistic_qc(_summary(tiny_frac=0.35))

    assert assessment.passed is False
    assert any("tiny_frac" in reason for reason in assessment.reasons)


def test_collect_biological_suspicion_warnings_flags_noise_and_merges() -> None:
    warnings = collect_biological_suspicion_warnings(
        _summary(object_count=1, tiny_frac=0.2, huge_frac=0.08, normal_frac=0.4),
        config=OptimisticQcConfig(),
    )

    assert any("low object count" in warning.lower() for warning in warnings)
    assert any("tiny_frac" in warning for warning in warnings)
    assert any("huge_frac" in warning for warning in warnings)
    assert any("normal_frac" in warning for warning in warnings)
