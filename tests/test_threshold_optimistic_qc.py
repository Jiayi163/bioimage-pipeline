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


def _baseline(**overrides: object) -> ThresholdVariantMeasurementSummary:
    payload = {
        "variant_id": "001_baseline",
        "display_name": "Baseline (original)",
        "success": True,
        "object_count": 701,
        "tiny_frac": 0.03,
        "huge_frac": 0.01,
        "normal_frac": 0.96,
    }
    payload.update(overrides)
    return ThresholdVariantMeasurementSummary(**payload)


def test_assess_optimistic_qc_passes_reasonable_candidate() -> None:
    baseline = _baseline(object_count=100)
    assessment = assess_optimistic_qc(_summary(object_count=120), baseline=baseline)

    assert assessment.passed is True
    assert assessment.score is not None
    assert assessment.score.score >= 0.5
    assert assessment.object_count_ratio_vs_baseline == 1.2
    assert not assessment.reasons


def test_assess_optimistic_qc_fails_on_zero_objects() -> None:
    assessment = assess_optimistic_qc(_summary(object_count=0), baseline=_baseline())

    assert assessment.passed is False
    assert any("No objects" in reason for reason in assessment.reasons)


def test_assess_optimistic_qc_fails_on_high_tiny_fraction() -> None:
    assessment = assess_optimistic_qc(_summary(tiny_frac=0.35), baseline=_baseline())

    assert assessment.passed is False
    assert any("tiny_frac" in reason for reason in assessment.reasons)


def test_assess_optimistic_qc_fails_on_extreme_object_count_over_baseline() -> None:
    baseline = _baseline(object_count=701)
    assessment = assess_optimistic_qc(
        _summary(object_count=39_792, normal_frac=1.0, tiny_frac=0.0, huge_frac=0.0),
        baseline=baseline,
    )

    assert assessment.passed is False
    assert assessment.object_count_ratio_vs_baseline is not None
    assert assessment.object_count_ratio_vs_baseline > 5.0
    assert any("baseline" in reason for reason in assessment.reasons)


def test_assess_optimistic_qc_fails_on_extreme_object_count_under_baseline() -> None:
    baseline = _baseline(object_count=1000)
    assessment = assess_optimistic_qc(
        _summary(object_count=50, normal_frac=0.95, tiny_frac=0.02, huge_frac=0.01),
        baseline=baseline,
    )

    assert assessment.passed is False
    assert any("baseline" in reason for reason in assessment.reasons)


def test_collect_biological_suspicion_warnings_flags_count_with_good_sizes() -> None:
    baseline = _baseline(object_count=701)
    warnings = collect_biological_suspicion_warnings(
        _summary(object_count=39_792, normal_frac=1.0, tiny_frac=0.0, huge_frac=0.0),
        config=OptimisticQcConfig(),
        baseline=baseline,
    )

    assert any("Possible over-detection" in warning for warning in warnings)


def test_collect_biological_suspicion_warnings_flags_noise_and_merges() -> None:
    warnings = collect_biological_suspicion_warnings(
        _summary(object_count=1, tiny_frac=0.2, huge_frac=0.08, normal_frac=0.4),
        config=OptimisticQcConfig(),
    )

    assert any("low object count" in warning.lower() for warning in warnings)
    assert any("tiny_frac" in warning for warning in warnings)
    assert any("huge_frac" in warning for warning in warnings)
    assert any("normal_frac" in warning for warning in warnings)
