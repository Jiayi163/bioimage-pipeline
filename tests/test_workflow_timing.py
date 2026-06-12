"""Tests for workflow timing helpers."""

import pytest

from bioimage_pipeline.workflow_timing import (
    finalize_workflow_timing,
    format_timing_breakdown,
    init_workflow_timing,
)


def test_finalize_workflow_timing_computes_aggregates_and_unaccounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter([115.0])
    monkeypatch.setattr(
        "bioimage_pipeline.workflow_timing.time.perf_counter",
        lambda: next(times),
    )
    timing = init_workflow_timing()
    timing["config_validation_seconds"] = 0.5
    timing["cellprofiler_startup_seconds"] = 1.0
    timing["cellprofiler_subprocess_seconds"] = 9.0
    timing["fiji_startup_seconds"] = 2.0
    timing["fiji_subprocess_seconds"] = 3.0
    timing["fiji_postprocess_seconds"] = 0.5
    timing["qc_seconds"] = 1.0
    timing["final_cleanup_seconds"] = 0.2

    finalize_workflow_timing(timing, total_started=100.0)

    assert timing["cellprofiler_seconds"] == 10.0
    assert timing["fiji_export_seconds"] == 5.5
    assert timing["total_seconds"] == 15.0
    assert timing["unaccounted_seconds"] == 0.0


def test_format_timing_breakdown_lists_checkpoints() -> None:
    timing = init_workflow_timing()
    timing["config_validation_seconds"] = 0.1
    timing["total_seconds"] = 1.0
    timing["unaccounted_seconds"] = 0.9

    text = format_timing_breakdown(timing)

    assert "Workflow timing breakdown:" in text
    assert "config validation: 0.10s" in text
    assert "total: 1.00s" in text
    assert "unaccounted: 0.90s" in text
