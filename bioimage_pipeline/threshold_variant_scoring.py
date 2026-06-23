"""Heuristic scoring and ranking for threshold variant measurement summaries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from bioimage_pipeline.threshold_variant_comparison import (
    ThresholdVariantMeasurementSummary,
)

_DEFAULT_BASELINE_MARKERS = ("baseline",)


@dataclass(frozen=True)
class ThresholdVariantScoreConfig:
    """Weights and cutoffs for heuristic threshold variant scoring."""

    normal_frac_weight: float = 1.0
    intensity_weight: float = 0.3
    intensity_scale_max: float = 255.0
    tiny_frac_penalty: float = 1.5
    huge_frac_penalty: float = 1.5
    tiny_frac_flag_threshold: float = 0.2
    huge_frac_flag_threshold: float = 0.1
    failure_penalty: float = 10.0
    object_count_ratio_fail_max: float = 5.0
    object_count_ratio_fail_min: float = 0.2
    object_count_extreme_penalty: float = 2.0
    object_count_deviation_threshold: float = 2.0
    object_count_deviation_penalty: float = 0.5
    low_object_count_threshold: int = 1
    low_object_count_penalty: float = 1.0
    good_normal_frac_threshold: float = 0.8
    good_huge_frac_threshold: float = 0.05


@dataclass
class ThresholdVariantScore:
    """Heuristic score and explanation for one variant summary."""

    rank: int
    variant_id: str
    display_name: str
    score: float
    reason: str
    explanations: list[str] = field(default_factory=list)
    component_scores: dict[str, float] = field(default_factory=dict)
    success: bool = True
    object_count: int | None = None
    normal_frac: float | None = None
    tiny_frac: float | None = None
    huge_frac: float | None = None
    median_intensity: float | None = None
    object_count_ratio_vs_baseline: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_baseline_summary(
    summaries: Sequence[ThresholdVariantMeasurementSummary],
) -> ThresholdVariantMeasurementSummary | None:
    for summary in summaries:
        variant_id = summary.variant_id.lower()
        display_name = summary.display_name.lower()
        if any(marker in variant_id for marker in _DEFAULT_BASELINE_MARKERS):
            return summary
        if any(marker in display_name for marker in _DEFAULT_BASELINE_MARKERS):
            return summary
    return summaries[0] if summaries else None


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))


def object_count_ratio_vs_baseline(
    summary: ThresholdVariantMeasurementSummary,
    baseline: ThresholdVariantMeasurementSummary | None,
) -> float | None:
    """Return candidate object_count divided by baseline object_count."""
    count = summary.object_count
    if count is None or baseline is None:
        return None
    reference_count = baseline.object_count
    if reference_count is None or reference_count <= 0:
        return None
    return count / reference_count


def _scale_intensity(
    median_intensity: float | None,
    *,
    intensity_scale_max: float,
) -> float:
    if median_intensity is None or intensity_scale_max <= 0:
        return 0.0
    return max(0.0, min(1.0, median_intensity / intensity_scale_max))


def _object_count_penalty(
    summary: ThresholdVariantMeasurementSummary,
    baseline: ThresholdVariantMeasurementSummary | None,
    config: ThresholdVariantScoreConfig,
) -> tuple[float, list[str], float | None]:
    explanations: list[str] = []
    count = summary.object_count
    ratio = object_count_ratio_vs_baseline(summary, baseline)
    if count is None:
        return 0.0, explanations, ratio

    if count <= config.low_object_count_threshold:
        explanations.append(
            f"Flagged because object_count = {count}, suggesting the threshold may "
            "be too strict."
        )
        return config.low_object_count_penalty, explanations, ratio

    if ratio is None:
        return 0.0, explanations, ratio

    if ratio > config.object_count_ratio_fail_max:
        penalty = config.object_count_extreme_penalty * (
            ratio / config.object_count_ratio_fail_max
        )
        explanations.append(
            f"Flagged because object_count is {ratio:.1f}x higher than baseline."
        )
        return penalty, explanations, ratio

    if ratio < config.object_count_ratio_fail_min:
        penalty = config.object_count_extreme_penalty * (
            config.object_count_ratio_fail_min / ratio
        )
        explanations.append(
            f"Flagged because object_count is {1.0 / ratio:.1f}x lower than baseline."
        )
        return penalty, explanations, ratio

    threshold = config.object_count_deviation_threshold
    if ratio > threshold:
        penalty = config.object_count_deviation_penalty * (ratio - threshold) / threshold
        explanations.append(
            f"Flagged because object_count is {ratio:.1f}x higher than baseline."
        )
        return penalty, explanations, ratio

    if ratio < (1.0 / threshold):
        penalty = config.object_count_deviation_penalty * ((1.0 / ratio) - threshold) / threshold
        explanations.append(
            f"Flagged because object_count is {1.0 / ratio:.1f}x lower than baseline."
        )
        return penalty, explanations, ratio

    return 0.0, explanations, ratio


def _primary_reason(
    *,
    success: bool,
    score: float,
    explanations: list[str],
) -> str:
    if not success:
        return "CellProfiler run failed; candidate invalid."

    flagged = [line for line in explanations if line.startswith("Flagged")]
    rejected = [line for line in explanations if line.startswith("Rejected")]
    positive = [line for line in explanations if line.startswith("Ranked high")]

    if rejected:
        return rejected[0].removeprefix("Rejected because ").removesuffix(".")
    if score >= 0.7 and positive:
        return positive[0].removeprefix("Ranked high because ").removesuffix(".")
    if flagged:
        return flagged[0].removeprefix("Flagged because ").removesuffix(".")
    if positive:
        return positive[0].removeprefix("Ranked high because ").removesuffix(".")
    return "Acceptable candidate metrics."


def score_threshold_variant_summary(
    summary: ThresholdVariantMeasurementSummary,
    config: ThresholdVariantScoreConfig,
    *,
    baseline: ThresholdVariantMeasurementSummary | None = None,
) -> ThresholdVariantScore:
    """Score one variant summary using configurable heuristic weights."""
    explanations: list[str] = []
    components: dict[str, float] = {}

    if not summary.success:
        explanations.append(
            "Rejected because the CellProfiler run failed for this candidate."
        )
        if summary.error_message:
            explanations.append(f"Failure detail: {summary.error_message}")
        return ThresholdVariantScore(
            rank=0,
            variant_id=summary.variant_id,
            display_name=summary.display_name,
            score=0.0,
            reason=_primary_reason(success=False, score=0.0, explanations=explanations),
            explanations=explanations,
            component_scores={"failure_penalty": -config.failure_penalty},
            success=False,
            object_count=summary.object_count,
            normal_frac=summary.normal_frac,
            tiny_frac=summary.tiny_frac,
            huge_frac=summary.huge_frac,
            median_intensity=summary.median_intensity,
            object_count_ratio_vs_baseline=object_count_ratio_vs_baseline(
                summary, baseline
            ),
        )

    raw_score = 0.0

    normal_frac = summary.normal_frac
    if normal_frac is not None:
        normal_term = config.normal_frac_weight * normal_frac
        components["normal_frac"] = normal_term
        raw_score += normal_term
        if normal_frac >= config.good_normal_frac_threshold:
            explanations.append(
                f"Ranked high because normal_frac = {normal_frac:.2f}."
            )

    intensity_term = config.intensity_weight * _scale_intensity(
        summary.median_intensity,
        intensity_scale_max=config.intensity_scale_max,
    )
    if intensity_term > 0:
        components["median_intensity_scaled"] = intensity_term
        raw_score += intensity_term
        if summary.median_intensity is not None:
            explanations.append(
                f"Reasonable intensity: median_intensity = {summary.median_intensity:.1f}."
            )

    tiny_frac = summary.tiny_frac
    if tiny_frac is not None:
        tiny_penalty = config.tiny_frac_penalty * tiny_frac
        components["tiny_frac_penalty"] = -tiny_penalty
        raw_score -= tiny_penalty
        if tiny_frac >= config.tiny_frac_flag_threshold:
            explanations.append(
                f"Flagged because tiny_frac = {tiny_frac:.2f}, suggesting noise "
                "over-detection."
            )

    huge_frac = summary.huge_frac
    if huge_frac is not None:
        huge_penalty = config.huge_frac_penalty * huge_frac
        components["huge_frac_penalty"] = -huge_penalty
        raw_score -= huge_penalty
        if huge_frac >= config.huge_frac_flag_threshold:
            explanations.append(
                f"Flagged because huge_frac = {huge_frac:.2f}, suggesting merged "
                "spots or background blobs."
            )
        elif huge_frac <= config.good_huge_frac_threshold and normal_frac is not None:
            explanations.append(
                f"Ranked high because huge_frac = {huge_frac:.2f}."
            )

    count_penalty, count_explanations, count_ratio = _object_count_penalty(
        summary, baseline, config
    )
    if count_penalty:
        components["object_count_penalty"] = -count_penalty
        raw_score -= count_penalty
    explanations.extend(count_explanations)

    if summary.object_count == 0:
        explanations.append(
            "Rejected because no objects were detected for this candidate."
        )
        raw_score = 0.0

    if not explanations:
        explanations.append("Limited metrics available for detailed scoring.")

    clamped_score = _clamp_score(raw_score)
    return ThresholdVariantScore(
        rank=0,
        variant_id=summary.variant_id,
        display_name=summary.display_name,
        score=clamped_score,
        reason=_primary_reason(
            success=True,
            score=clamped_score,
            explanations=explanations,
        ),
        explanations=explanations,
        component_scores=components,
        success=True,
        object_count=summary.object_count,
        normal_frac=summary.normal_frac,
        tiny_frac=summary.tiny_frac,
        huge_frac=summary.huge_frac,
        median_intensity=summary.median_intensity,
        object_count_ratio_vs_baseline=count_ratio,
    )


def rank_threshold_variant_summaries(
    summaries: Sequence[ThresholdVariantMeasurementSummary],
    config: ThresholdVariantScoreConfig | None = None,
) -> list[ThresholdVariantScore]:
    """Score and rank variant summaries from best to worst."""
    score_config = config or ThresholdVariantScoreConfig()
    baseline = _find_baseline_summary(summaries)

    scored = [
        score_threshold_variant_summary(summary, score_config, baseline=baseline)
        for summary in summaries
    ]
    scored.sort(
        key=lambda item: (item.success, item.score, item.variant_id),
        reverse=True,
    )

    ranked: list[ThresholdVariantScore] = []
    for index, item in enumerate(scored, start=1):
        ranked.append(
            ThresholdVariantScore(
                rank=index,
                variant_id=item.variant_id,
                display_name=item.display_name,
                score=item.score,
                reason=item.reason,
                explanations=list(item.explanations),
                component_scores=dict(item.component_scores),
                success=item.success,
                object_count=item.object_count,
                normal_frac=item.normal_frac,
                tiny_frac=item.tiny_frac,
                huge_frac=item.huge_frac,
                median_intensity=item.median_intensity,
                object_count_ratio_vs_baseline=item.object_count_ratio_vs_baseline,
            )
        )
    return ranked


def threshold_variant_ranking_to_dataframe(
    scores: Sequence[ThresholdVariantScore],
) -> pd.DataFrame:
    """Convert ranked scores to a tabular ranking view."""
    rows = [
        {
            "rank": score.rank,
            "variant_id": score.variant_id,
            "name": score.display_name,
            "score": score.score,
            "reason": score.reason,
            "success": score.success,
            "object_count": score.object_count,
            "object_count_ratio_vs_baseline": score.object_count_ratio_vs_baseline,
            "normal_frac": score.normal_frac,
            "tiny_frac": score.tiny_frac,
            "huge_frac": score.huge_frac,
            "median_intensity": score.median_intensity,
            "explanations": "; ".join(score.explanations),
        }
        for score in scores
    ]
    return pd.DataFrame(rows)


def save_threshold_variant_ranking(
    scores: Sequence[ThresholdVariantScore],
    output_dir: str | Path,
    *,
    basename: str = "threshold_variant_ranking",
) -> dict[str, Path]:
    """Write ranked scores with explanations to CSV and JSON."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    csv_path = (destination / f"{basename}.csv").resolve()
    json_path = (destination / f"{basename}.json").resolve()

    dataframe = threshold_variant_ranking_to_dataframe(scores)
    dataframe.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps([score.to_dict() for score in scores], indent=2),
        encoding="utf-8",
    )

    return {"csv": csv_path, "json": json_path}
