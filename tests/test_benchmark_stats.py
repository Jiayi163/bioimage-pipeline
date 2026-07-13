"""Tests for benchmark statistics helpers."""

from __future__ import annotations

from bioimage_pipeline.puncta.benchmark_stats import (
    aggregate_benchmark_group,
    summarize_binary_outcomes,
    wilson_score_interval,
)
import pandas as pd


def test_wilson_interval_bounds() -> None:
    lo, hi = wilson_score_interval(8, 10)
    assert lo is not None and hi is not None
    assert 0.0 <= lo <= hi <= 1.0


def test_summarize_binary_outcomes() -> None:
    stats = summarize_binary_outcomes([True, True, False, True])
    assert stats["completed_runs"] == 4
    assert stats["successes"] == 3
    assert stats["rate"] == 0.75


def test_aggregate_benchmark_group_includes_ci() -> None:
    frame = pd.DataFrame(
        [
            {
                "separation_px": 2,
                "sigma": 2.2,
                "exact_count_correct": True,
                "pass_criterion": True,
                "predicted_accepted_count": 2,
                "under_split": False,
                "over_split": False,
                "mean_localization_error_px": 0.5,
                "median_localization_error_px": 0.5,
                "runtime_s": 10.0,
            },
            {
                "separation_px": 2,
                "sigma": 2.2,
                "exact_count_correct": False,
                "pass_criterion": False,
                "predicted_accepted_count": 1,
                "under_split": True,
                "over_split": False,
                "mean_localization_error_px": None,
                "median_localization_error_px": None,
                "runtime_s": 12.0,
            },
        ]
    )
    summary = aggregate_benchmark_group(frame, ["separation_px", "sigma"])
    assert len(summary) == 1
    assert summary.iloc[0]["completed_runs"] == 2
    assert summary.iloc[0]["full_pass_rate"] == 0.5
