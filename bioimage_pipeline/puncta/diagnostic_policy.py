"""Select which mask objects receive expensive diagnostic PNG exports."""

from __future__ import annotations

from dataclasses import dataclass, field

from bioimage_pipeline.puncta.config import DiagnosticMode, PunctaDeclumpConfig
from bioimage_pipeline.puncta.object_processor import ObjectProcessResult
from bioimage_pipeline.puncta.types import ObjectInfo


@dataclass
class DiagnosticRecord:
    """One object eligible for diagnostic PNG export."""

    object_id: int
    obj: ObjectInfo
    result: ObjectProcessResult
    categories: set[str] = field(default_factory=set)
    one_gaussian_r_squared: float | None = None
    one_gaussian_residual_relative: float | None = None
    priority: int = 0


def _is_gmm_routed(result: ObjectProcessResult) -> bool:
    if result.path == "gmm":
        return True
    if result.mixture is not None and result.mixture.n_components > 1:
        return True
    return False


def _is_fallback(result: ObjectProcessResult) -> bool:
    if result.path == "fallback":
        return True
    return any(c.fit_status == "fit_failed_fallback" for c in result.candidates)


def _gmm_tried_rejected(result: ObjectProcessResult) -> bool:
    debug = result.debug
    if not debug.tried_gmm:
        return False
    if result.mixture is not None and result.mixture.n_components > 1:
        return False
    if debug.rejected_component_reason:
        return True
    if debug.model_selection_reason and "kept_single" in debug.model_selection_reason:
        return True
    return False


def _is_ordinary_single_fit_ok(result: ObjectProcessResult) -> bool:
    """True when this is a normal single-spot Gaussian fit with no suspicion flags."""
    if result.path != "single":
        return False
    if result.debug.under_split_suspect:
        return False
    if result.debug.tried_gmm:
        return False
    primary = result.candidates[0] if result.candidates else None
    if primary is None:
        return False
    if primary.fit_status != "fit_ok":
        return False
    r2 = result.debug.one_gaussian_r_squared
    resid = result.debug.one_gaussian_residual_relative
    if r2 is not None and r2 < 0.5:
        return False
    if resid is not None and resid > 0.20:
        return False
    return True


def classify_object_for_diagnostics(
    obj: ObjectInfo,
    result: ObjectProcessResult,
    config: PunctaDeclumpConfig,
) -> DiagnosticRecord | None:
    """Return a diagnostic record if this object qualifies; None if it should be skipped."""
    mode = config.diagnostic_mode
    if mode in ("off", "summary"):
        return None

    manual_ids = set(config.diagnostic_object_ids)
    categories: set[str] = set()
    debug = result.debug

    if obj.label in manual_ids:
        categories.add("manual")

    if mode == "selected_objects":
        if "manual" not in categories:
            return None
    elif mode == "all":
        if not _is_ordinary_single_fit_ok(result):
            categories.add("all_non_ordinary")
        else:
            categories.add("ordinary")
    elif mode in ("balanced", "suspicious_only"):
        if _is_gmm_routed(result):
            categories.add("gmm")
        if debug.under_split_suspect:
            categories.add("undersplit")
        if _is_fallback(result):
            categories.add("fallback")
        if _gmm_tried_rejected(result):
            categories.add("gmm_rejected")
        r2 = debug.one_gaussian_r_squared
        if r2 is not None and r2 < config.diagnostic_low_r_squared:
            categories.add("low_r2")
        resid = debug.one_gaussian_residual_relative
        if resid is not None and resid > config.diagnostic_high_residual_relative:
            categories.add("high_residual")
        if "manual" in categories:
            pass
        if not categories:
            return None
        if _is_ordinary_single_fit_ok(result) and categories <= {"manual"}:
            pass
        elif _is_ordinary_single_fit_ok(result):
            return None

    priority = 0
    if "manual" in categories:
        priority += 1000
    if "gmm" in categories:
        priority += 500
    if "undersplit" in categories:
        priority += 400
    if "fallback" in categories:
        priority += 300
    if "gmm_rejected" in categories:
        priority += 250
    if "low_r2" in categories:
        priority += 200
    if "high_residual" in categories:
        priority += 150

    return DiagnosticRecord(
        object_id=obj.label,
        obj=obj,
        result=result,
        categories=categories,
        one_gaussian_r_squared=debug.one_gaussian_r_squared,
        one_gaussian_residual_relative=debug.one_gaussian_residual_relative,
        priority=priority,
    )


def select_objects_for_diagnostics(
    records: list[DiagnosticRecord],
    config: PunctaDeclumpConfig,
) -> list[DiagnosticRecord]:
    """Apply max_diagnostic_objects cap with priority buckets."""
    if not records:
        return []
    if config.diagnostic_mode in ("off", "summary"):
        return []

    max_n = config.max_diagnostic_objects
    if len(records) <= max_n:
        return records

    manual_ids = set(config.diagnostic_object_ids)
    selected: dict[int, DiagnosticRecord] = {}

    def add(rec: DiagnosticRecord) -> None:
        selected[rec.object_id] = rec

    # Priority buckets when over cap.
    for rec in records:
        if rec.object_id in manual_ids or "manual" in rec.categories:
            add(rec)
    for rec in records:
        if "gmm" in rec.categories:
            add(rec)
    for rec in records:
        if "undersplit" in rec.categories:
            add(rec)

    # Worst 20 by R² (lowest).
    low_r2 = sorted(
        records,
        key=lambda r: r.one_gaussian_r_squared if r.one_gaussian_r_squared is not None else 1.0,
    )[:20]
    for rec in low_r2:
        add(rec)

    # Worst 20 by residual (highest).
    high_res = sorted(
        records,
        key=lambda r: r.one_gaussian_residual_relative
        if r.one_gaussian_residual_relative is not None
        else 0.0,
        reverse=True,
    )[:20]
    for rec in high_res:
        add(rec)

    # Fill remaining slots by priority if under cap.
    if len(selected) < max_n:
        remaining = [r for r in records if r.object_id not in selected]
        remaining.sort(key=lambda r: (-r.priority, r.object_id))
        for rec in remaining:
            if len(selected) >= max_n:
                break
            add(rec)

    # If over cap after union, trim by priority (keep highest priority).
    ordered = sorted(selected.values(), key=lambda r: (-r.priority, r.object_id))
    return ordered[:max_n]
