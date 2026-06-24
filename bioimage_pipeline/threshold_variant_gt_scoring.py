"""Ground-truth scoring for threshold variant CellProfiler runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from bioimage_pipeline.export import classify_tiff_for_fiji_export
from bioimage_pipeline.ground_truth import (
    GroundTruthManifest,
    load_reference_mask,
)
from bioimage_pipeline.io import read_tiff
from bioimage_pipeline.threshold_variant_runner import ThresholdVariantRunResult
from bioimage_pipeline.validation import compare_segmentation

GT_LABEL_BEST = "best_match"
GT_LABEL_ACCEPTABLE = "acceptable"
GT_LABEL_POOR = "poor"

PREDICTED_MASK_NOT_FOUND_MSG = (
    "Predicted mask not found. Ensure the pipeline exports "
    "segmentation masks via SaveImages."
)

SEGMENTATION_SUFFIXES: tuple[str, ...] = (
    "_mask",
    "_predicted_mask",
    "_objects",
    "_labels",
    "_segmented",
    "_seg",
)

SEGMENTATION_KEYWORDS: tuple[str, ...] = (
    "mask",
    "objects",
    "labels",
    "label",
    "segmented",
    "primaryobjects",
    "secondaryobjects",
    "tertiaryobjects",
)

FOREGROUND_NEAR_EMPTY = 0.001
FOREGROUND_NEAR_FULL = 0.99
REFERENCE_NEAR_FULL = 0.99


def is_segmentation_export_filename(image_stem: str, tiff_stem: str) -> bool:
    """Return whether a TIFF stem looks like a SaveImages segmentation export."""
    image_lower = image_stem.lower()
    tiff_lower = tiff_stem.lower()

    if tiff_lower == image_lower:
        return False
    if not tiff_lower.startswith(image_lower):
        return False

    remainder = tiff_lower[len(image_lower) :]
    if not remainder or remainder[0] not in "._-":
        return False

    remainder_body = remainder.lstrip("._-")
    for suffix in SEGMENTATION_SUFFIXES:
        token = suffix.lstrip("_")
        if remainder_body == token or remainder_body.startswith(f"{token}_"):
            return True
        if remainder_body.endswith(f"_{token}") or remainder_body.endswith(token):
            return True

    return any(keyword in remainder_body for keyword in SEGMENTATION_KEYWORDS)


def _foreground_fraction(array: np.ndarray) -> float:
    """Estimate foreground coverage for a mask or label image."""
    squeezed = np.squeeze(np.asarray(array))
    if squeezed.dtype == bool:
        return float(np.count_nonzero(squeezed)) / squeezed.size
    return float(np.count_nonzero(squeezed > 0)) / squeezed.size


def appears_grayscale_raw(array: np.ndarray, *, kind: str) -> bool:
    """Return whether an array looks like fluorescence rather than segmentation."""
    if kind == "mask":
        return False

    squeezed = np.squeeze(np.asarray(array))
    if squeezed.ndim != 2:
        return True

    unique = np.unique(squeezed)
    if len(unique) <= 3 and set(unique.tolist()).issubset({0, 1, 255}):
        return False

    if kind == "label" and np.issubdtype(squeezed.dtype, np.integer):
        positive = unique[unique > 0]
        if positive.size and int(positive.max()) <= len(positive) * 2:
            return False

    if int(unique.min()) > 0 and len(unique) >= 4:
        return True
    if len(unique) > 32:
        return True
    return False


def validate_reference_mask_array(array: np.ndarray) -> list[str]:
    """Return blocking warnings for an invalid reference mask."""
    warnings: list[str] = []
    squeezed = np.squeeze(np.asarray(array))
    if squeezed.ndim != 2:
        warnings.append(
            f"Reference mask must be 2D, got shape {getattr(squeezed, 'shape', ())}."
        )
        return warnings

    foreground = _foreground_fraction(squeezed)
    if foreground >= REFERENCE_NEAR_FULL:
        warnings.append(
            "Reference mask foreground fraction is near 100%; "
            "annotations may be invalid for comparison."
        )
    elif foreground <= FOREGROUND_NEAR_EMPTY:
        warnings.append(
            f"Reference mask foreground fraction is near 0% ({foreground:.1%})."
        )
    return warnings


def validate_predicted_segmentation_array(
    array: np.ndarray,
    *,
    filename: str = "",
) -> list[str]:
    """Return blocking warnings when a candidate is not a valid segmentation mask."""
    warnings: list[str] = []
    kind = classify_tiff_for_fiji_export(array, filename=filename)
    if kind == "intensity":
        warnings.append(
            "Predicted mask appears to be a raw/grayscale image, not a "
            "SaveImages segmentation export."
        )
        return warnings

    squeezed = np.squeeze(np.asarray(array))
    if squeezed.ndim != 2:
        warnings.append(
            f"Predicted mask must be 2D, got shape {getattr(squeezed, 'shape', ())}."
        )
        return warnings

    if appears_grayscale_raw(squeezed, kind=kind):
        warnings.append(
            "Predicted mask appears grayscale/raw-like rather than binary "
            "or label-like."
        )

    foreground = _foreground_fraction(squeezed)
    if foreground <= FOREGROUND_NEAR_EMPTY:
        warnings.append(
            f"Predicted mask foreground fraction is near 0% ({foreground:.1%})."
        )
    elif foreground >= FOREGROUND_NEAR_FULL:
        warnings.append(
            f"Predicted mask foreground fraction is near 100% ({foreground:.1%})."
        )
    return warnings


@dataclass
class GroundTruthImageComparison:
    """Ground-truth comparison for one variant and one image."""

    variant_id: str
    display_name: str
    image_name: str
    success: bool
    reference_available: bool = False
    predicted_mask_available: bool = False
    iou: float | None = None
    dice: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    true_positives: int | None = None
    false_positives: int | None = None
    false_negatives: int | None = None
    count_error: int | None = None
    count_ratio: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroundTruthVariantScore:
    """Aggregate ground-truth score for one threshold variant."""

    variant_id: str
    display_name: str
    gt_rank: int
    gt_score: float
    gt_label: str
    mean_f1: float | None = None
    mean_dice: float | None = None
    mean_iou: float | None = None
    mean_count_error: float | None = None
    annotated_image_count: int = 0
    success: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_predicted_mask_path(
    raw_output_dir: str | Path,
    image_name: str,
) -> Path | None:
    """Locate a CellProfiler-exported predicted mask for one input image."""
    from bioimage_pipeline.cellprofiler_runner import discover_cellprofiler_tiff_files

    raw_path = Path(raw_output_dir)
    if not raw_path.is_dir():
        return None

    image_stem = Path(image_name).stem
    mask_paths: list[Path] = []
    label_paths: list[Path] = []

    for tiff_path in discover_cellprofiler_tiff_files(raw_path):
        if not is_segmentation_export_filename(image_stem, tiff_path.stem):
            continue
        try:
            image = read_tiff(tiff_path)
            kind = classify_tiff_for_fiji_export(image, filename=tiff_path.name)
        except (OSError, ValueError):
            continue
        if kind == "intensity":
            continue
        if validate_predicted_segmentation_array(
            image,
            filename=tiff_path.name,
        ):
            continue

        resolved = tiff_path.resolve()
        if kind == "mask":
            mask_paths.append(resolved)
        elif kind == "label":
            label_paths.append(resolved)

    def _pick(candidates: list[Path]) -> Path | None:
        if not candidates:
            return None
        for candidate in candidates:
            if candidate.stem.lower().endswith("_mask"):
                return candidate
        for candidate in candidates:
            if image_stem in candidate.stem:
                return candidate
        return candidates[0]

    return _pick(mask_paths) or _pick(label_paths)


def load_predicted_mask(path: str | Path) -> np.ndarray:
    """Load a predicted segmentation mask as boolean 2D."""
    array = np.asarray(read_tiff(path))
    if array.ndim == 3 and array.shape[0] <= 4:
        array = array[0]
    elif array.ndim == 3:
        array = array[..., 0]
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Predicted mask must reduce to 2D, got shape {array.shape}")
    return array > 0


def compare_variant_run_to_ground_truth(
    run_result: ThresholdVariantRunResult,
    *,
    manifest: GroundTruthManifest,
    match_iou_threshold: float = 0.3,
) -> list[GroundTruthImageComparison]:
    """Compare one variant run against all annotated subset images."""
    rows: list[GroundTruthImageComparison] = []
    if not run_result.success:
        for entry in manifest.entries:
            rows.append(
                GroundTruthImageComparison(
                    variant_id=run_result.spec.variant_id,
                    display_name=run_result.spec.display_name,
                    image_name=entry.image_name,
                    success=False,
                    reference_available=True,
                    warnings=["CellProfiler variant run failed."],
                )
            )
        return rows

    for entry in manifest.entries:
        warnings: list[str] = []
        predicted_path = resolve_predicted_mask_path(
            run_result.raw_output_dir,
            entry.image_name,
        )
        if predicted_path is None:
            rows.append(
                GroundTruthImageComparison(
                    variant_id=run_result.spec.variant_id,
                    display_name=run_result.spec.display_name,
                    image_name=entry.image_name,
                    success=False,
                    reference_available=True,
                    predicted_mask_available=False,
                    warnings=[PREDICTED_MASK_NOT_FOUND_MSG],
                )
            )
            continue

        try:
            reference_mask = load_reference_mask(entry.reference_mask_path)
            predicted_array = np.asarray(read_tiff(predicted_path))
            reference_warnings = validate_reference_mask_array(reference_mask)
            predicted_warnings = validate_predicted_segmentation_array(
                predicted_array,
                filename=predicted_path.name,
            )
            blocking_warnings = [*reference_warnings, *predicted_warnings]
            if blocking_warnings:
                rows.append(
                    GroundTruthImageComparison(
                        variant_id=run_result.spec.variant_id,
                        display_name=run_result.spec.display_name,
                        image_name=entry.image_name,
                        success=False,
                        reference_available=True,
                        predicted_mask_available=True,
                        warnings=blocking_warnings,
                    )
                )
                continue

            predicted_mask = load_predicted_mask(predicted_path)
            comparison = compare_segmentation(
                predicted_mask,
                reference_mask,
                match_iou_threshold=match_iou_threshold,
            )
        except (OSError, ValueError) as exc:
            rows.append(
                GroundTruthImageComparison(
                    variant_id=run_result.spec.variant_id,
                    display_name=run_result.spec.display_name,
                    image_name=entry.image_name,
                    success=False,
                    reference_available=True,
                    predicted_mask_available=True,
                    warnings=[str(exc)],
                )
            )
            continue

        object_level = comparison.object_level
        pixel = comparison.pixel
        rows.append(
            GroundTruthImageComparison(
                variant_id=run_result.spec.variant_id,
                display_name=run_result.spec.display_name,
                image_name=entry.image_name,
                success=True,
                reference_available=True,
                predicted_mask_available=True,
                iou=pixel.iou,
                dice=pixel.dice,
                precision=object_level.precision,
                recall=object_level.recall,
                f1=object_level.f1,
                true_positives=object_level.true_positives,
                false_positives=object_level.false_positives,
                false_negatives=object_level.false_negatives,
                count_error=object_level.count_error,
                count_ratio=object_level.count_ratio,
                warnings=warnings,
            )
        )
    return rows


def compare_threshold_variants_to_ground_truth(
    run_results: Sequence[ThresholdVariantRunResult],
    *,
    manifest: GroundTruthManifest,
    match_iou_threshold: float = 0.3,
) -> list[GroundTruthImageComparison]:
    """Compare all variant runs against the ground-truth manifest."""
    rows: list[GroundTruthImageComparison] = []
    for run_result in run_results:
        rows.extend(
            compare_variant_run_to_ground_truth(
                run_result,
                manifest=manifest,
                match_iou_threshold=match_iou_threshold,
            )
        )
    return rows


def _mean(values: Sequence[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _derive_gt_label(
    mean_f1: float | None,
    *,
    success: bool,
) -> str:
    if not success or mean_f1 is None:
        return GT_LABEL_POOR
    if mean_f1 >= 0.75:
        return GT_LABEL_BEST
    if mean_f1 >= 0.5:
        return GT_LABEL_ACCEPTABLE
    return GT_LABEL_POOR


def rank_ground_truth_variant_scores(
    image_rows: Sequence[GroundTruthImageComparison],
) -> list[GroundTruthVariantScore]:
    """Aggregate per-image GT rows into ranked variant scores."""
    by_variant: dict[str, list[GroundTruthImageComparison]] = {}
    display_names: dict[str, str] = {}
    for row in image_rows:
        by_variant.setdefault(row.variant_id, []).append(row)
        display_names[row.variant_id] = row.display_name

    provisional: list[GroundTruthVariantScore] = []
    for variant_id, rows in by_variant.items():
        successful = [row for row in rows if row.success]
        warnings: list[str] = []
        if not successful:
            warnings.append("No successful ground-truth comparisons for this variant.")
        mean_f1 = _mean([row.f1 for row in successful])
        mean_dice = _mean([row.dice for row in successful])
        mean_iou = _mean([row.iou for row in successful])
        mean_count_error = _mean(
            [float(row.count_error) for row in successful if row.count_error is not None]
        )

        if mean_f1 is not None:
            gt_score = mean_f1
        elif mean_dice is not None:
            gt_score = mean_dice
        else:
            gt_score = 0.0

        if mean_count_error is not None and mean_count_error >= 5:
            warnings.append(
                f"High mean object count error ({mean_count_error:.1f}) vs reference."
            )

        provisional.append(
            GroundTruthVariantScore(
                variant_id=variant_id,
                display_name=display_names[variant_id],
                gt_rank=0,
                gt_score=gt_score,
                gt_label=_derive_gt_label(mean_f1, success=bool(successful)),
                mean_f1=mean_f1,
                mean_dice=mean_dice,
                mean_iou=mean_iou,
                mean_count_error=mean_count_error,
                annotated_image_count=len(successful),
                success=bool(successful),
                warnings=warnings,
            )
        )

    ranked = sorted(
        provisional,
        key=lambda item: (
            item.success,
            item.gt_score,
            item.mean_dice or 0.0,
            - (item.mean_count_error or 0.0),
        ),
        reverse=True,
    )
    return [
        GroundTruthVariantScore(
            variant_id=score.variant_id,
            display_name=score.display_name,
            gt_rank=index,
            gt_score=score.gt_score,
            gt_label=score.gt_label,
            mean_f1=score.mean_f1,
            mean_dice=score.mean_dice,
            mean_iou=score.mean_iou,
            mean_count_error=score.mean_count_error,
            annotated_image_count=score.annotated_image_count,
            success=score.success,
            warnings=score.warnings,
        )
        for index, score in enumerate(ranked, start=1)
    ]


def ground_truth_image_rows_to_dataframe(
    rows: Sequence[GroundTruthImageComparison],
) -> pd.DataFrame:
    """Convert per-image GT rows to a table."""
    payload = [
        {
            "variant_id": row.variant_id,
            "name": row.display_name,
            "image_name": row.image_name,
            "success": row.success,
            "iou": row.iou,
            "dice": row.dice,
            "precision": row.precision,
            "recall": row.recall,
            "f1": row.f1,
            "true_positives": row.true_positives,
            "false_positives": row.false_positives,
            "false_negatives": row.false_negatives,
            "count_error": row.count_error,
            "count_ratio": row.count_ratio,
            "warnings": "; ".join(row.warnings),
        }
        for row in rows
    ]
    return pd.DataFrame(payload)


def ground_truth_variant_scores_to_dataframe(
    scores: Sequence[GroundTruthVariantScore],
) -> pd.DataFrame:
    """Convert aggregate GT scores to a table."""
    payload = [
        {
            "gt_rank": score.gt_rank,
            "variant_id": score.variant_id,
            "name": score.display_name,
            "gt_score": score.gt_score,
            "gt_label": score.gt_label,
            "mean_f1": score.mean_f1,
            "mean_dice": score.mean_dice,
            "mean_iou": score.mean_iou,
            "mean_count_error": score.mean_count_error,
            "annotated_image_count": score.annotated_image_count,
            "warnings": "; ".join(score.warnings),
        }
        for score in scores
    ]
    return pd.DataFrame(payload)


def save_threshold_variant_gt_comparison(
    image_rows: Sequence[GroundTruthImageComparison],
    variant_scores: Sequence[GroundTruthVariantScore],
    output_dir: str | Path,
    *,
    basename: str = "threshold_variant_gt_comparison",
) -> dict[str, Path]:
    """Write GT comparison CSV/JSON and ranking CSV/JSON."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    image_csv = (destination / f"{basename}.csv").resolve()
    image_json = (destination / f"{basename}.json").resolve()
    ranking_csv = (destination / f"{basename}_ranking.csv").resolve()
    ranking_json = (destination / f"{basename}_ranking.json").resolve()

    ground_truth_image_rows_to_dataframe(image_rows).to_csv(image_csv, index=False)
    image_json.write_text(
        json.dumps([row.to_dict() for row in image_rows], indent=2),
        encoding="utf-8",
    )
    ground_truth_variant_scores_to_dataframe(variant_scores).to_csv(
        ranking_csv,
        index=False,
    )
    ranking_json.write_text(
        json.dumps([score.to_dict() for score in variant_scores], indent=2),
        encoding="utf-8",
    )
    return {
        "image_csv": image_csv,
        "image_json": image_json,
        "ranking_csv": ranking_csv,
        "ranking_json": ranking_json,
    }


def load_ground_truth_image_comparisons(
    path: str | Path,
) -> list[GroundTruthImageComparison]:
    """Load per-image GT comparison rows from JSON."""
    json_path = Path(path)
    if json_path.suffix.lower() != ".json":
        json_path = json_path.with_name(f"{json_path.stem}.json")
    if not json_path.is_file():
        return []

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []

    rows: list[GroundTruthImageComparison] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        rows.append(
            GroundTruthImageComparison(
                variant_id=str(entry.get("variant_id", "")),
                display_name=str(entry.get("display_name", "")),
                image_name=str(entry.get("image_name", "")),
                success=bool(entry.get("success", False)),
                reference_available=bool(entry.get("reference_available", False)),
                predicted_mask_available=bool(
                    entry.get("predicted_mask_available", False)
                ),
                iou=entry.get("iou"),
                dice=entry.get("dice"),
                precision=entry.get("precision"),
                recall=entry.get("recall"),
                f1=entry.get("f1"),
                true_positives=entry.get("true_positives"),
                false_positives=entry.get("false_positives"),
                false_negatives=entry.get("false_negatives"),
                count_error=entry.get("count_error"),
                count_ratio=entry.get("count_ratio"),
                warnings=list(entry.get("warnings", [])),
            )
        )
    return rows


def load_ground_truth_variant_scores(
    path: str | Path,
) -> list[GroundTruthVariantScore]:
    """Load aggregate GT variant scores from JSON."""
    json_path = Path(path)
    if not json_path.is_file():
        return []

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []

    scores: list[GroundTruthVariantScore] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        scores.append(
            GroundTruthVariantScore(
                variant_id=str(entry.get("variant_id", "")),
                display_name=str(entry.get("display_name", "")),
                gt_rank=int(entry.get("gt_rank", 0)),
                gt_score=float(entry.get("gt_score", 0.0)),
                gt_label=str(entry.get("gt_label", GT_LABEL_POOR)),
                mean_f1=entry.get("mean_f1"),
                mean_dice=entry.get("mean_dice"),
                mean_iou=entry.get("mean_iou"),
                mean_count_error=entry.get("mean_count_error"),
                annotated_image_count=int(entry.get("annotated_image_count", 0)),
                success=bool(entry.get("success", False)),
                warnings=list(entry.get("warnings", [])),
            )
        )
    return scores
