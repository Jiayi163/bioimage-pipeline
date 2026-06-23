"""Basic QC for the fast optimistic threshold recommendation candidate."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from bioimage_pipeline.threshold_variant_comparison import (
    ThresholdVariantMeasurementSummary,
)
from bioimage_pipeline.threshold_variant_runner import ThresholdVariantRunResult
from bioimage_pipeline.threshold_variant_scoring import (
    ThresholdVariantScore,
    ThresholdVariantScoreConfig,
    object_count_ratio_vs_baseline,
    score_threshold_variant_summary,
)

OPTIMISTIC_QC_FILENAME = "optimistic_qc.json"


@dataclass(frozen=True)
class OptimisticQcConfig:
    """Cutoffs for optimistic candidate pass/fail."""

    min_score: float = 0.5
    tiny_frac_fail_threshold: float = 0.2
    huge_frac_fail_threshold: float = 0.1
    low_object_count_threshold: int = 1
    object_count_ratio_max_fail: float = 5.0
    object_count_ratio_min_fail: float = 0.2
    suspicious_tiny_frac: float = 0.15
    suspicious_huge_frac: float = 0.05
    suspicious_normal_frac: float = 0.5


@dataclass
class OptimisticQcAssessment:
    """Outcome of basic QC on one optimistic subset trial."""

    passed: bool
    summary: ThresholdVariantMeasurementSummary
    score: ThresholdVariantScore | None = None
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    baseline_object_count: int | None = None
    object_count_ratio_vs_baseline: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "passed": self.passed,
            "warnings": list(self.warnings),
            "reasons": list(self.reasons),
            "baseline_object_count": self.baseline_object_count,
            "object_count_ratio_vs_baseline": self.object_count_ratio_vs_baseline,
            "summary": self.summary.to_dict(),
        }
        if self.score is not None:
            payload["score"] = self.score.to_dict()
        return payload


def _size_metrics_acceptable(
    summary: ThresholdVariantMeasurementSummary,
    *,
    config: OptimisticQcConfig,
) -> bool:
    tiny_frac = summary.tiny_frac
    huge_frac = summary.huge_frac
    tiny_ok = tiny_frac is None or tiny_frac < config.tiny_frac_fail_threshold
    huge_ok = huge_frac is None or huge_frac < config.huge_frac_fail_threshold
    return tiny_ok and huge_ok


def collect_biological_suspicion_warnings(
    summary: ThresholdVariantMeasurementSummary,
    *,
    config: OptimisticQcConfig | None = None,
    baseline: ThresholdVariantMeasurementSummary | None = None,
) -> list[str]:
    """Return warnings when optimistic results look biologically suspicious."""
    qc_config = config or OptimisticQcConfig()
    warnings = list(summary.warnings)

    if summary.object_count is not None and summary.object_count <= qc_config.low_object_count_threshold:
        warnings.append(
            "Very low object count may indicate an overly strict threshold."
        )

    tiny_frac = summary.tiny_frac
    if tiny_frac is not None and tiny_frac >= qc_config.suspicious_tiny_frac:
        warnings.append(
            f"tiny_frac = {tiny_frac:.2f} suggests noise over-detection."
        )

    huge_frac = summary.huge_frac
    if huge_frac is not None and huge_frac >= qc_config.suspicious_huge_frac:
        warnings.append(
            f"huge_frac = {huge_frac:.2f} suggests merged spots or background blobs."
        )

    normal_frac = summary.normal_frac
    if normal_frac is not None and normal_frac < qc_config.suspicious_normal_frac:
        warnings.append(
            f"normal_frac = {normal_frac:.2f} is low for expected compact spots."
        )

    ratio = object_count_ratio_vs_baseline(summary, baseline)
    if ratio is not None and _size_metrics_acceptable(summary, config=qc_config):
        if ratio > qc_config.object_count_ratio_max_fail:
            warnings.append(
                f"Possible over-detection: object_count is {ratio:.1f}x baseline "
                "despite acceptable size metrics."
            )
        elif ratio < qc_config.object_count_ratio_min_fail:
            warnings.append(
                f"Possible under-detection: object_count is {ratio:.1f}x baseline "
                "despite acceptable size metrics."
            )

    return warnings


def assess_optimistic_qc(
    summary: ThresholdVariantMeasurementSummary,
    *,
    baseline: ThresholdVariantMeasurementSummary | None = None,
    qc_config: OptimisticQcConfig | None = None,
    score_config: ThresholdVariantScoreConfig | None = None,
) -> OptimisticQcAssessment:
    """Decide whether the optimistic candidate passes basic subset QC."""
    config = qc_config or OptimisticQcConfig()
    baseline_object_count = baseline.object_count if baseline is not None else None
    count_ratio = object_count_ratio_vs_baseline(summary, baseline)
    warnings = collect_biological_suspicion_warnings(
        summary,
        config=config,
        baseline=baseline,
    )
    reasons: list[str] = []

    if not summary.success:
        reasons.append("CellProfiler run failed for the optimistic candidate.")
        if summary.error_message:
            reasons.append(summary.error_message)
        return OptimisticQcAssessment(
            passed=False,
            summary=summary,
            warnings=warnings,
            reasons=reasons,
            baseline_object_count=baseline_object_count,
            object_count_ratio_vs_baseline=count_ratio,
        )

    object_count = summary.object_count
    if object_count is None or object_count <= 0:
        reasons.append("No objects were detected on the subset.")

    tiny_frac = summary.tiny_frac
    if tiny_frac is not None and tiny_frac >= config.tiny_frac_fail_threshold:
        reasons.append(
            f"tiny_frac = {tiny_frac:.2f} exceeds QC limit "
            f"({config.tiny_frac_fail_threshold:.2f})."
        )

    huge_frac = summary.huge_frac
    if huge_frac is not None and huge_frac >= config.huge_frac_fail_threshold:
        reasons.append(
            f"huge_frac = {huge_frac:.2f} exceeds QC limit "
            f"({config.huge_frac_fail_threshold:.2f})."
        )

    if count_ratio is not None:
        if count_ratio > config.object_count_ratio_max_fail:
            reasons.append(
                f"object_count is {count_ratio:.1f}x baseline, exceeding QC limit "
                f"({config.object_count_ratio_max_fail:.1f}x)."
            )
        elif count_ratio < config.object_count_ratio_min_fail:
            reasons.append(
                f"object_count is {count_ratio:.2f}x baseline, below QC limit "
                f"({config.object_count_ratio_min_fail:.1f}x)."
            )

    score = score_threshold_variant_summary(
        summary,
        score_config or ThresholdVariantScoreConfig(),
        baseline=baseline,
    )
    if not score.success:
        reasons.append(score.reason)
    elif score.score < config.min_score:
        reasons.append(
            f"Heuristic score {score.score:.2f} is below QC minimum "
            f"{config.min_score:.2f}."
        )

    passed = not reasons
    return OptimisticQcAssessment(
        passed=passed,
        summary=summary,
        score=score,
        warnings=warnings,
        reasons=reasons,
        baseline_object_count=baseline_object_count,
        object_count_ratio_vs_baseline=count_ratio,
    )


def save_optimistic_qc_report(
    assessment: OptimisticQcAssessment,
    output_dir: str | Path,
    *,
    run_result: ThresholdVariantRunResult | None = None,
    preview_paths: Sequence[Path] | None = None,
) -> Path:
    """Write optimistic QC JSON with counts, size stats, warnings, and previews."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    payload = assessment.to_dict()
    payload["preview_image_paths"] = [
        str(path.resolve()) for path in (preview_paths or [])
    ]
    if run_result is not None:
        payload["variant_id"] = run_result.spec.variant_id
        payload["display_name"] = run_result.spec.display_name
        payload["variant_dir"] = str(run_result.variant_dir.resolve())
        payload["pipeline_path"] = str(run_result.pipeline_path.resolve())
        payload["measurements_dir"] = str(run_result.measurements_dir.resolve())
        payload["qc_dir"] = str(run_result.qc_dir.resolve())

    report_path = (destination / OPTIMISTIC_QC_FILENAME).resolve()
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path
