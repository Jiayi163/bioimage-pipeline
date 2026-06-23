"""Tests for threshold variant heuristic scoring and ranking."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bioimage_pipeline.threshold_variant_comparison import (
    ThresholdVariantMeasurementSummary,
)
from bioimage_pipeline.threshold_variant_scoring import (
    ThresholdVariantScoreConfig,
    rank_threshold_variant_summaries,
    save_threshold_variant_ranking,
    score_threshold_variant_summary,
)


def _summary(
    variant_id: str,
    display_name: str,
    *,
    success: bool = True,
    object_count: int | None = 100,
    normal_frac: float | None = 0.9,
    tiny_frac: float | None = 0.05,
    huge_frac: float | None = 0.02,
    median_intensity: float | None = 180.0,
    error_message: str | None = None,
) -> ThresholdVariantMeasurementSummary:
    return ThresholdVariantMeasurementSummary(
        variant_id=variant_id,
        display_name=display_name,
        success=success,
        object_count=object_count,
        normal_frac=normal_frac,
        tiny_frac=tiny_frac,
        huge_frac=huge_frac,
        median_intensity=median_intensity,
        error_message=error_message,
    )


def test_score_prefers_good_size_distribution_over_noisy_candidate() -> None:
    baseline = _summary("001_baseline", "Baseline (original)")
    good = _summary(
        "002_otsu_global",
        "Otsu Global",
        normal_frac=0.91,
        tiny_frac=0.02,
        huge_frac=0.02,
    )
    noisy = _summary(
        "003_otsu_adaptive_cf_0_9",
        "Otsu Adaptive (CF 0.9)",
        normal_frac=0.52,
        tiny_frac=0.48,
        huge_frac=0.0,
        object_count=400,
    )

    good_score = score_threshold_variant_summary(
        good,
        ThresholdVariantScoreConfig(),
        baseline=baseline,
    )
    noisy_score = score_threshold_variant_summary(
        noisy,
        ThresholdVariantScoreConfig(),
        baseline=baseline,
    )

    assert good_score.score > noisy_score.score
    assert any("tiny_frac = 0.48" in line for line in noisy_score.explanations)
    assert any("normal_frac = 0.91" in line for line in good_score.explanations)


def test_score_marks_failed_variant_invalid() -> None:
    failed = _summary(
        "004_failed",
        "Failed",
        success=False,
        object_count=None,
        normal_frac=None,
        tiny_frac=None,
        huge_frac=None,
        median_intensity=None,
        error_message="CellProfiler failed",
    )

    score = score_threshold_variant_summary(failed, ThresholdVariantScoreConfig())

    assert score.score == 0.0
    assert score.success is False
    assert "failed" in score.reason.lower()
    assert any("Rejected because" in line for line in score.explanations)


def test_rank_threshold_variant_summaries_orders_best_first() -> None:
    summaries = [
        _summary("001_baseline", "Baseline (original)", object_count=100),
        _summary(
            "002_otsu_global",
            "Otsu Global",
            normal_frac=0.91,
            tiny_frac=0.02,
            huge_frac=0.02,
        ),
        _summary(
            "003_otsu_adaptive_cf_0_9",
            "Otsu Adaptive (CF 0.9)",
            normal_frac=0.52,
            tiny_frac=0.48,
            huge_frac=0.0,
            object_count=400,
        ),
        _summary(
            "004_failed",
            "Failed",
            success=False,
            object_count=None,
            normal_frac=None,
            tiny_frac=None,
            huge_frac=None,
            median_intensity=None,
        ),
    ]

    ranked = rank_threshold_variant_summaries(summaries)

    assert [item.variant_id for item in ranked] == [
        "002_otsu_global",
        "001_baseline",
        "003_otsu_adaptive_cf_0_9",
        "004_failed",
    ]
    assert ranked[0].rank == 1
    assert ranked[-1].success is False


def test_rank_flags_extreme_object_count_relative_to_baseline() -> None:
    summaries = [
        _summary("001_baseline", "Baseline (original)", object_count=100),
        _summary(
            "005_mce_adaptive_cf_0_9",
            "Minimum Cross-Entropy Adaptive (CF 0.9)",
            object_count=600,
            normal_frac=0.85,
            tiny_frac=0.05,
            huge_frac=0.05,
        ),
    ]

    ranked = rank_threshold_variant_summaries(summaries)
    candidate = next(
        item for item in ranked if item.variant_id.endswith("mce_adaptive_cf_0_9")
    )

    assert candidate.object_count_ratio_vs_baseline == 6.0
    assert any("6.0x higher than baseline" in line for line in candidate.explanations)


def test_score_penalizes_extreme_object_count_despite_perfect_size_metrics() -> None:
    baseline = _summary("001_baseline", "Baseline (original)", object_count=701)
    over_detected = _summary(
        "001_optimistic_otsu_adaptive",
        "Optimistic Otsu Adaptive",
        object_count=39_792,
        normal_frac=1.0,
        tiny_frac=0.0,
        huge_frac=0.0,
    )

    score = score_threshold_variant_summary(
        over_detected,
        ThresholdVariantScoreConfig(),
        baseline=baseline,
    )

    assert score.object_count_ratio_vs_baseline is not None
    assert score.object_count_ratio_vs_baseline > 50.0
    assert score.score < 0.5


def test_save_threshold_variant_ranking_writes_csv_and_json(tmp_path: Path) -> None:
    ranked = rank_threshold_variant_summaries(
        [
            _summary("001_baseline", "Baseline (original)"),
            _summary(
                "002_otsu_global",
                "Otsu Global",
                normal_frac=0.91,
                tiny_frac=0.02,
                huge_frac=0.02,
            ),
        ]
    )

    paths = save_threshold_variant_ranking(ranked, tmp_path / "ranking")

    assert paths["csv"].exists()
    assert paths["json"].exists()

    dataframe = pd.read_csv(paths["csv"])
    assert list(dataframe.columns) == [
        "rank",
        "variant_id",
        "name",
        "score",
        "reason",
        "success",
        "object_count",
        "object_count_ratio_vs_baseline",
        "normal_frac",
        "tiny_frac",
        "huge_frac",
        "median_intensity",
        "explanations",
    ]
    assert dataframe.loc[0, "rank"] == 1

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload[0]["rank"] == 1
    assert "explanations" in payload[0]
