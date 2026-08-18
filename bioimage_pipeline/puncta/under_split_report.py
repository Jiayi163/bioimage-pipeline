"""Under-split object report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bioimage_pipeline.puncta.types import DeclumpResult, PunctumCandidate


def _object_rows(result: DeclumpResult) -> pd.DataFrame:
    """Collapse candidates to one row per object with under-split diagnostics."""
    rows: list[dict[str, object]] = []
    by_object: dict[int, list[PunctumCandidate]] = {}
    for candidate in result.candidates:
        by_object.setdefault(candidate.object_id, []).append(candidate)

    obj_lookup = {obj.label: obj for obj in result.objects}
    for object_id, candidates in by_object.items():
        accepted = [c for c in candidates if c.accepted and c.fit_status == "fit_ok"]
        primary = candidates[0]
        obj = obj_lookup.get(object_id)
        rows.append(
            {
                "object_id": object_id,
                "n_candidates": len(candidates),
                "n_accepted_fit_ok": len(accepted),
                "path": primary.path,
                "fit_status": primary.fit_status,
                "area": primary.object_area if primary.object_area is not None else (obj.area if obj else None),
                "equivalent_diameter": primary.object_equivalent_diameter
                if primary.object_equivalent_diameter is not None
                else (obj.equivalent_diameter if obj else None),
                "eccentricity": primary.object_eccentricity
                if primary.object_eccentricity is not None
                else (obj.eccentricity if obj else None),
                "solidity": primary.object_solidity
                if primary.object_solidity is not None
                else (obj.solidity if obj else None),
                "major_axis_length": primary.object_major_axis_length
                if primary.object_major_axis_length is not None
                else (obj.major_axis_length if obj else None),
                "minor_axis_length": primary.object_minor_axis_length
                if primary.object_minor_axis_length is not None
                else (obj.minor_axis_length if obj else None),
                "elongation": primary.object_elongation
                if primary.object_elongation is not None
                else (obj.elongation if obj else None),
                "n_raw_local_maxima": primary.n_raw_local_maxima,
                "n_filtered_local_maxima": primary.n_filtered_local_maxima,
                "tried_gmm": primary.tried_gmm,
                "gmm_trigger_reasons": primary.gmm_trigger_reasons,
                "gmm_candidate_components": primary.gmm_candidate_components,
                "one_gaussian_r_squared": primary.one_gaussian_r_squared,
                "one_gaussian_residual_relative": primary.one_gaussian_residual_relative,
                "best_gmm_r_squared": primary.best_gmm_r_squared,
                "best_gmm_residual_relative": primary.best_gmm_residual_relative,
                "model_selection_reason": primary.model_selection_reason,
                "rejected_component_reason": primary.rejected_component_reason,
                "under_split_suspect": primary.under_split_suspect,
                "under_split_reasons": primary.under_split_reasons,
                "sigma": primary.sigma,
                "sigma_x": primary.sigma_col,
                "sigma_y": primary.sigma_row,
                "r_squared": primary.r_squared,
                "residual_relative": primary.residual_relative,
                "failure_category": _classify_failure(primary, len(accepted)),
                "local_peak_recovery_attempted": primary.local_peak_recovery_attempted,
                "local_peak_recovery_success": primary.local_peak_recovery_success,
                "local_peak_recovery_raw_count": primary.local_peak_recovery_raw_count,
                "local_peak_recovery_filtered_count": primary.local_peak_recovery_filtered_count,
                "peak_source": primary.peak_source,
            }
        )
    return pd.DataFrame(rows)


def _classify_failure(candidate: PunctumCandidate, n_accepted: int) -> str:
    """Explain why a visually clumped object may have stayed as one spot."""
    if n_accepted >= 2:
        return "split_ok"
    reasons = candidate.under_split_reasons or ""
    triggers = candidate.gmm_trigger_reasons or ""
    rejected = candidate.rejected_component_reason or ""

    n_raw = candidate.n_raw_local_maxima or 0
    n_filt = candidate.n_filtered_local_maxima or 0

    if n_raw < 2 and n_filt < 2 and not candidate.tried_gmm:
        if "elongat" in triggers or "eccentric" in triggers or "large_" in triggers:
            return "shape_suggests_clump_but_maxima_not_detected"
        return "local_maxima_not_detected"
    if n_raw >= 2 and n_filt < 2 and not candidate.tried_gmm:
        return "second_peak_filtered_out_before_gmm"
    if not candidate.tried_gmm:
        return "gmm_not_triggered"
    if candidate.tried_gmm and n_accepted <= 1:
        if "too_close" in rejected or "merged_comp" in rejected:
            return "gmm_tried_second_component_merged_too_close"
        if "weak_comp" in rejected or "amplitude" in rejected.lower():
            return "gmm_tried_second_component_amplitude_too_low"
        if "bic_not_better" in rejected:
            return "gmm_tried_but_model_selection_kept_single"
        if "collapsed_to_one" in (candidate.model_selection_reason or ""):
            return "gmm_tried_but_components_collapsed"
        if rejected:
            return "gmm_tried_but_second_component_rejected"
        return "gmm_tried_but_only_one_accepted"
    if "mask" in reasons.lower():
        return "mask_or_threshold_problem"
    return "unknown_or_not_suspect"


def build_under_split_report(
    result: DeclumpResult,
    *,
    top_n: int = 50,
) -> list[dict[str, object]]:
    """Return ranked under-split suspect objects with failure categories."""
    dataframe = _object_rows(result)
    if dataframe.empty:
        return []

    suspects = dataframe[dataframe["under_split_suspect"] == True].copy()  # noqa: E712
    if suspects.empty:
        # Also include objects with low R2 / high residual / large sigma / elongated
        # even if not flagged, for the report.
        suspects = dataframe.copy()

    def score(row: pd.Series) -> float:
        value = 0.0
        if bool(row.get("under_split_suspect")):
            value += 10.0
        r2 = row.get("one_gaussian_r_squared")
        if r2 is not None and pd.notna(r2):
            value += max(0.0, 0.9 - float(r2)) * 5.0
        resid = row.get("one_gaussian_residual_relative")
        if resid is not None and pd.notna(resid):
            value += min(float(resid), 1.0) * 4.0
        elong = row.get("elongation")
        if elong is not None and pd.notna(elong):
            value += max(0.0, float(elong) - 1.0) * 2.0
        n_raw = row.get("n_raw_local_maxima") or 0
        n_acc = row.get("n_accepted_fit_ok") or 0
        if int(n_raw) >= 2 and int(n_acc) <= 1:
            value += 8.0
        sigma = row.get("sigma")
        if sigma is not None and pd.notna(sigma):
            value += max(0.0, float(sigma) - 1.5)
        return value

    suspects["rank_score"] = suspects.apply(score, axis=1)
    ranked = suspects.sort_values("rank_score", ascending=False).head(top_n)
    return ranked.to_dict(orient="records")


def export_under_split_report(
    output_dir: str | Path,
    result: DeclumpResult,
    *,
    stem: str = "puncta",
    top_n: int = 50,
) -> dict[str, Path]:
    """Write under-split CSV + JSON report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    object_df = _object_rows(result)
    report_rows = build_under_split_report(result, top_n=top_n)
    result.under_split_report = report_rows

    paths: dict[str, Path] = {}
    object_csv = output_path / f"{stem}_object_diagnostics.csv"
    object_df.to_csv(object_csv, index=False)
    paths["object_diagnostics"] = object_csv

    report_csv = output_path / f"{stem}_undersplit_report.csv"
    pd.DataFrame(report_rows).to_csv(report_csv, index=False)
    paths["undersplit_report"] = report_csv

    report_json = output_path / f"{stem}_undersplit_report.json"
    # Categorize counts
    categories: dict[str, int] = {}
    for row in report_rows:
        cat = str(row.get("failure_category", "unknown"))
        categories[cat] = categories.get(cat, 0) + 1

    payload = {
        "top_n": top_n,
        "n_suspects_listed": len(report_rows),
        "failure_category_counts": categories,
        "objects": report_rows,
        "legend": {
            "local_maxima_not_detected": "No 2nd peak found inside mask ROI",
            "second_peak_filtered_out_before_gmm": "Raw peaks >=2 but filtered peaks <2",
            "gmm_not_triggered": "Object never entered GMM path",
            "gmm_tried_but_model_selection_kept_single": "2-comp fit ran but BIC/R2 kept 1-comp",
            "gmm_tried_second_component_merged_too_close": "2nd component merged by distance",
            "gmm_tried_second_component_amplitude_too_low": "2nd component removed as weak",
            "gmm_tried_but_second_component_rejected": "GMM ran; 2nd component failed validation",
            "shape_suggests_clump_but_maxima_not_detected": "Elongated/large but only 1 peak",
            "split_ok": "Already split into 2+ accepted components",
        },
    }
    report_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    paths["undersplit_report_json"] = report_json
    return paths
