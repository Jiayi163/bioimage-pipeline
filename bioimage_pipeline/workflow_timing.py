"""Workflow stage timing helpers for profiling overhead."""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_META_KEYS = frozenset({"total_seconds", "unaccounted_seconds"})
_LEGACY_AGGREGATE_KEYS = frozenset({"cellprofiler_seconds", "fiji_export_seconds"})

TIMING_DISPLAY_ORDER: tuple[str, ...] = (
    "config_validation_seconds",
    "pipeline_materialization_seconds",
    "setup_directories_seconds",
    "prepare_input_seconds",
    "classifier_prediction_seconds",
    "classifier_staging_seconds",
    "classifier_qc_seconds",
    "adaptive_threshold_seconds",
    "cellprofiler_startup_seconds",
    "cellprofiler_subprocess_seconds",
    "cellprofiler_seconds",
    "copy_measurements_seconds",
    "inspect_cp_logs_seconds",
    "load_measurements_seconds",
    "csv_merge_export_seconds",
    "fiji_startup_seconds",
    "fiji_subprocess_seconds",
    "fiji_postprocess_seconds",
    "fiji_export_seconds",
    "qc_seconds",
    "final_cleanup_seconds",
    "unaccounted_seconds",
    "total_seconds",
)


def init_workflow_timing() -> dict[str, float]:
    """Return a timing dict with all workflow checkpoint keys zeroed."""
    return {key: 0.0 for key in TIMING_DISPLAY_ORDER if key != "unaccounted_seconds"}


def elapsed_since(started: float) -> float:
    """Return seconds elapsed since *started* from :func:`time.perf_counter`."""
    return time.perf_counter() - started


def finalize_workflow_timing(
    timing: dict[str, float],
    total_started: float,
) -> dict[str, float]:
    """Set legacy aggregates, total wall time, and unaccounted overhead."""
    timing["cellprofiler_seconds"] = (
        timing.get("cellprofiler_startup_seconds", 0.0)
        + timing.get("cellprofiler_subprocess_seconds", 0.0)
    )
    timing["fiji_export_seconds"] = (
        timing.get("fiji_startup_seconds", 0.0)
        + timing.get("fiji_subprocess_seconds", 0.0)
        + timing.get("fiji_postprocess_seconds", 0.0)
    )
    timing["total_seconds"] = elapsed_since(total_started)
    accounted = sum(
        value
        for key, value in timing.items()
        if key not in _META_KEYS and key not in _LEGACY_AGGREGATE_KEYS
    )
    timing["unaccounted_seconds"] = max(0.0, timing["total_seconds"] - accounted)
    return timing


def format_timing_breakdown(timing: dict[str, float]) -> str:
    """Format a human-readable timing breakdown for logs and the GUI."""
    lines = ["Workflow timing breakdown:"]
    for key in TIMING_DISPLAY_ORDER:
        if key not in timing:
            continue
        label = key.replace("_seconds", "").replace("_", " ")
        lines.append(f"  {label}: {timing[key]:.2f}s")
    return "\n".join(lines)


def log_timing_breakdown(
    timing: dict[str, float],
    *,
    logs_dir: Path | None = None,
) -> None:
    """Log and optionally persist the workflow timing breakdown."""
    text = format_timing_breakdown(timing)
    logger.info("\n%s", text)
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "timing_breakdown.txt").write_text(text + "\n", encoding="utf-8")
