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

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "passed": self.passed,
            "warnings": list(self.warnings),
            "reasons": list(self.reasons),
            "summary": self.summary.to_dict(),
        }
        if self.score is not None:
            payload["score"] = self.score.to_dict()
        return payload


def collect_biological_suspicion_warnings(
    summary: ThresholdVariantMeasurementSummary,
    *,
    config: OptimisticQcConfig | None = None,
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

    return warnings


def assess_optimistic_qc(
    summary: ThresholdVariantMeasurementSummary,
    *,
    qc_config: OptimisticQcConfig | None = None,
    score_config: ThresholdVariantScoreConfig | None = None,
) -> OptimisticQcAssessment:
    """Decide whether the optimistic candidate passes basic subset QC."""
    config = qc_config or OptimisticQcConfig()
    warnings = collect_biological_suspicion_warnings(summary, config=config)
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

    score = score_threshold_variant_summary(
        summary,
        score_config or ThresholdVariantScoreConfig(),
        baseline=None,
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
