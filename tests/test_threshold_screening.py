"""Tests for threshold parameter assistant screening labels."""

from __future__ import annotations

from bioimage_pipeline.threshold_variant_scoring import (
    HEURISTIC_SCREENING_PREFIX,
    ThresholdVariantScore,
    derive_screening_label,
    format_screening_reason,
)


def test_derive_screening_label_flags_extreme_count_ratio() -> None:
    score = ThresholdVariantScore(
        rank=1,
        variant_id="001_optimistic",
        display_name="Optimistic",
        score=1.0,
        reason="normal_frac = 1.00",
        success=True,
        object_count_ratio_vs_baseline=57.0,
    )

    assert derive_screening_label(score) == "flagged"


def test_format_screening_reason_adds_prefix() -> None:
    assert format_screening_reason("normal_frac = 1.00").startswith(
        HEURISTIC_SCREENING_PREFIX
    )
