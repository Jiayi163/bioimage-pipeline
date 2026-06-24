"""Compare CellProfiler object measurements across threshold pipeline variants."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from bioimage_pipeline.cellprofiler_runner import (
    CellProfilerTableMetadata,
    load_cellprofiler_measurements_lenient,
)
from bioimage_pipeline.threshold_variant_runner import ThresholdVariantRunResult

_OBJECT_TABLE_SIGNAL_COLUMNS = (
    "AreaShape_Area",
    "Location_Center_X",
    "Location_Center_Y",
)
_AREA_COLUMN_CANDIDATES = ("AreaShape_Area", "area")


@dataclass(frozen=True)
class ThresholdVariantSizeThresholds:
    """Configurable area cutoffs for tiny/huge object fractions."""

    tiny_area_px: float = 2.0
    huge_area_px: float = 200.0
    high_foreground_coverage: float = 0.10
    object_count_ratio_max: float = 5.0
    object_count_ratio_min: float = 0.2
    count_instability_cv: float = 0.75


@dataclass
class ThresholdVariantMeasurementSummary:
    """Measurement summary for one threshold variant run."""

    variant_id: str
    display_name: str
    success: bool
    object_count: int | None = None
    median_area: float | None = None
    mean_area: float | None = None
    tiny_frac: float | None = None
    huge_frac: float | None = None
    normal_frac: float | None = None
    median_intensity: float | None = None
    mean_intensity: float | None = None
    object_table_names: list[str] = field(default_factory=list)
    measurements_dir: Path | None = None
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.measurements_dir is not None:
            payload["measurements_dir"] = str(self.measurements_dir)
        return payload


@dataclass
class ThresholdVariantPerImageSummary:
    """Per-image measurement summary for one variant."""

    variant_id: str
    display_name: str
    image_number: int
    image_name: str | None
    success: bool
    object_count: int | None = None
    median_area: float | None = None
    tiny_frac: float | None = None
    huge_frac: float | None = None
    normal_frac: float | None = None
    foreground_coverage: float | None = None
    object_count_ratio_vs_baseline: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pick_area_column(dataframe: pd.DataFrame) -> str | None:
    for column in _AREA_COLUMN_CANDIDATES:
        if column in dataframe.columns:
            return column
    return None


def _pick_intensity_column(dataframe: pd.DataFrame) -> str | None:
    intensity_columns = sorted(
        column for column in dataframe.columns if column.startswith("Intensity_")
    )
    if not intensity_columns:
        for column in dataframe.columns:
            if "MeanIntensity" in column or "IntegratedIntensity" in column:
                return column
        return None

    for column in intensity_columns:
        if "MeanIntensity" in column:
            return column
    for column in intensity_columns:
        if "IntegratedIntensity" in column:
            return column
    return intensity_columns[0]


def _is_object_measurement_table(
    dataframe: pd.DataFrame,
    metadata: CellProfilerTableMetadata,
) -> bool:
    if metadata.table_type == "object":
        return True

    columns = set(dataframe.columns)
    if "ObjectNumber" not in columns:
        return False

    if any(column in columns for column in _OBJECT_TABLE_SIGNAL_COLUMNS):
        return True

    return any(column.startswith("Intensity_") for column in columns)


def _find_object_measurement_tables(
    tables: dict[str, pd.DataFrame],
    metadata: dict[str, CellProfilerTableMetadata],
) -> list[tuple[str, pd.DataFrame]]:
    object_tables: list[tuple[str, pd.DataFrame]] = []
    for table_name, dataframe in tables.items():
        table_metadata = metadata.get(table_name)
        if table_metadata is None:
            continue
        if _is_object_measurement_table(dataframe, table_metadata):
            object_tables.append((table_name, dataframe))
    return object_tables


def _finite_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric[np.isfinite(numeric)]


def _summarize_object_table(
    dataframe: pd.DataFrame,
    *,
    size_thresholds: ThresholdVariantSizeThresholds,
) -> tuple[dict[str, float | int | None], list[str]]:
    warnings: list[str] = []
    if dataframe.empty:
        return {
            "object_count": 0,
            "median_area": None,
            "mean_area": None,
            "tiny_frac": None,
            "huge_frac": None,
            "normal_frac": None,
            "median_intensity": None,
            "mean_intensity": None,
        }, warnings

    area_column = _pick_area_column(dataframe)
    intensity_column = _pick_intensity_column(dataframe)

    object_count = int(len(dataframe))
    summary: dict[str, float | int | None] = {"object_count": object_count}

    if area_column is None:
        warnings.append("Object table has no area column; area metrics unavailable.")
        summary.update(
            {
                "median_area": None,
                "mean_area": None,
                "tiny_frac": None,
                "huge_frac": None,
                "normal_frac": None,
            }
        )
    else:
        areas = _finite_series(dataframe[area_column])
        if areas.empty:
            warnings.append("Object table area column contains no numeric values.")
            summary.update(
                {
                    "median_area": None,
                    "mean_area": None,
                    "tiny_frac": None,
                    "huge_frac": None,
                    "normal_frac": None,
                }
            )
        else:
            tiny_mask = areas < size_thresholds.tiny_area_px
            huge_mask = areas > size_thresholds.huge_area_px
            normal_mask = ~tiny_mask & ~huge_mask
            summary.update(
                {
                    "median_area": float(areas.median()),
                    "mean_area": float(areas.mean()),
                    "tiny_frac": float(tiny_mask.mean()),
                    "huge_frac": float(huge_mask.mean()),
                    "normal_frac": float(normal_mask.mean()),
                }
            )

    if intensity_column is None:
        warnings.append(
            "Object table has no Intensity_* column; intensity metrics unavailable."
        )
        summary.update({"median_intensity": None, "mean_intensity": None})
    else:
        intensities = _finite_series(dataframe[intensity_column])
        if intensities.empty:
            warnings.append(
                f"Intensity column {intensity_column!r} contains no numeric values."
            )
            summary.update({"median_intensity": None, "mean_intensity": None})
        else:
            summary.update(
                {
                    "median_intensity": float(intensities.median()),
                    "mean_intensity": float(intensities.mean()),
                }
            )

    return summary, warnings


def _combine_object_tables(
    object_tables: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    if not object_tables:
        return pd.DataFrame()
    if len(object_tables) == 1:
        return object_tables[0][1]
    return pd.concat(
        [dataframe for _, dataframe in object_tables],
        ignore_index=True,
    )


def summarize_threshold_variant_measurements(
    measurements_dir: str | Path,
    *,
    variant_id: str,
    display_name: str,
    success: bool,
    size_thresholds: ThresholdVariantSizeThresholds | None = None,
    error_message: str | None = None,
) -> ThresholdVariantMeasurementSummary:
    """Summarize object measurements for one variant measurements folder."""
    thresholds = size_thresholds or ThresholdVariantSizeThresholds()
    measurements_path = Path(measurements_dir)

    if not success:
        return ThresholdVariantMeasurementSummary(
            variant_id=variant_id,
            display_name=display_name,
            success=False,
            measurements_dir=measurements_path.resolve()
            if measurements_path.exists()
            else measurements_path,
            error_message=error_message,
        )

    load_result = load_cellprofiler_measurements_lenient(measurements_path)
    warnings = list(load_result.warnings)
    object_tables = _find_object_measurement_tables(
        load_result.tables,
        load_result.metadata,
    )

    if not object_tables:
        warnings.append("No object measurement tables found in measurements folder.")
        return ThresholdVariantMeasurementSummary(
            variant_id=variant_id,
            display_name=display_name,
            success=True,
            measurements_dir=measurements_path.resolve(),
            warnings=warnings,
            error_message=error_message,
        )

    combined = _combine_object_tables(object_tables)
    metrics, metric_warnings = _summarize_object_table(
        combined,
        size_thresholds=thresholds,
    )
    warnings.extend(metric_warnings)

    return ThresholdVariantMeasurementSummary(
        variant_id=variant_id,
        display_name=display_name,
        success=True,
        object_count=int(metrics["object_count"] or 0),
        median_area=_optional_float(metrics.get("median_area")),
        mean_area=_optional_float(metrics.get("mean_area")),
        tiny_frac=_optional_float(metrics.get("tiny_frac")),
        huge_frac=_optional_float(metrics.get("huge_frac")),
        normal_frac=_optional_float(metrics.get("normal_frac")),
        median_intensity=_optional_float(metrics.get("median_intensity")),
        mean_intensity=_optional_float(metrics.get("mean_intensity")),
        object_table_names=[name for name, _ in object_tables],
        measurements_dir=measurements_path.resolve(),
        warnings=warnings,
        error_message=error_message,
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def compare_threshold_variant_run_results(
    run_results: Iterable[ThresholdVariantRunResult],
    *,
    size_thresholds: ThresholdVariantSizeThresholds | None = None,
) -> list[ThresholdVariantMeasurementSummary]:
    """Summarize measurements for each threshold variant run result."""
    summaries: list[ThresholdVariantMeasurementSummary] = []
    for result in run_results:
        summaries.append(
            summarize_threshold_variant_measurements(
                result.measurements_dir,
                variant_id=result.spec.variant_id,
                display_name=result.spec.display_name,
                success=result.success,
                size_thresholds=size_thresholds,
                error_message=result.error_message,
            )
        )
    return summaries


def threshold_variant_comparison_to_dataframe(
    summaries: Sequence[ThresholdVariantMeasurementSummary],
) -> pd.DataFrame:
    """Convert variant measurement summaries to a comparison table."""
    rows = [
        {
            "variant_id": summary.variant_id,
            "name": summary.display_name,
            "success": summary.success,
            "object_count": summary.object_count,
            "median_area": summary.median_area,
            "mean_area": summary.mean_area,
            "tiny_frac": summary.tiny_frac,
            "normal_frac": summary.normal_frac,
            "huge_frac": summary.huge_frac,
            "median_intensity": summary.median_intensity,
            "mean_intensity": summary.mean_intensity,
            "object_table_names": ",".join(summary.object_table_names),
            "error_message": summary.error_message,
        }
        for summary in summaries
    ]
    return pd.DataFrame(rows)


def _image_number_column(dataframe: pd.DataFrame) -> str | None:
    for column in ("Image_Number", "ImageNumber"):
        if column in dataframe.columns:
            return column
    return None


def _filename_column(dataframe: pd.DataFrame) -> str | None:
    if "FileName" in dataframe.columns:
        return "FileName"
    filename_columns = [
        column for column in dataframe.columns if column.startswith("FileName")
    ]
    return filename_columns[0] if filename_columns else None


def _image_name_lookup(
    tables: dict[str, pd.DataFrame],
    metadata: dict[str, CellProfilerTableMetadata],
) -> dict[int, str]:
    mapping: dict[int, str] = {}
    image_number_column = None
    filename_column = None
    image_table: pd.DataFrame | None = None

    for table_name, dataframe in tables.items():
        table_metadata = metadata.get(table_name)
        if table_metadata is None or table_metadata.table_type != "image":
            continue
        image_number_column = _image_number_column(dataframe)
        filename_column = _filename_column(dataframe)
        if image_number_column and filename_column:
            image_table = dataframe
            break

    if image_table is None or image_number_column is None or filename_column is None:
        return mapping

    for _, row in image_table.iterrows():
        image_number = row.get(image_number_column)
        filename = row.get(filename_column)
        if pd.isna(image_number) or pd.isna(filename):
            continue
        mapping[int(image_number)] = str(Path(str(filename)).name)
    return mapping


def _image_pixel_count(image_path: Path) -> int | None:
    if not image_path.is_file():
        return None
    try:
        from bioimage_pipeline.io import read_tiff

        array = np.asarray(read_tiff(image_path))
        if array.ndim == 3 and array.shape[0] <= 4:
            plane = array[0]
        else:
            plane = array
        plane = np.squeeze(plane)
        if plane.ndim != 2:
            return None
        return int(plane.size)
    except (OSError, ValueError):
        return None


def _summarize_object_table_per_image(
    dataframe: pd.DataFrame,
    *,
    image_number: int,
    image_name: str | None,
    image_pixels: int | None,
    size_thresholds: ThresholdVariantSizeThresholds,
) -> ThresholdVariantPerImageSummary:
    warnings: list[str] = []
    image_column = _image_number_column(dataframe)
    if image_column is None:
        subset = dataframe
    else:
        subset = dataframe[dataframe[image_column] == image_number]

    metrics, metric_warnings = _summarize_object_table(
        subset,
        size_thresholds=size_thresholds,
    )
    warnings.extend(metric_warnings)

    foreground_coverage = None
    area_column = _pick_area_column(subset) if not subset.empty else _pick_area_column(dataframe)
    if area_column is not None and image_pixels and image_pixels > 0:
        areas = _finite_series(subset[area_column]) if not subset.empty else pd.Series(dtype=float)
        if not areas.empty:
            foreground_coverage = float(min(1.0, areas.sum() / image_pixels))
            if foreground_coverage >= size_thresholds.high_foreground_coverage:
                warnings.append(
                    f"High foreground coverage ({foreground_coverage:.1%}) suggests "
                    "over-segmentation for sparse spot images."
                )

    return ThresholdVariantPerImageSummary(
        variant_id="",
        display_name="",
        image_number=image_number,
        image_name=image_name,
        success=True,
        object_count=int(metrics["object_count"] or 0),
        median_area=_optional_float(metrics.get("median_area")),
        tiny_frac=_optional_float(metrics.get("tiny_frac")),
        huge_frac=_optional_float(metrics.get("huge_frac")),
        normal_frac=_optional_float(metrics.get("normal_frac")),
        foreground_coverage=foreground_coverage,
        warnings=warnings,
    )


def summarize_threshold_variant_per_image(
    measurements_dir: str | Path,
    *,
    variant_id: str,
    display_name: str,
    success: bool,
    subset_dir: str | Path | None = None,
    image_names: Sequence[str] | None = None,
    size_thresholds: ThresholdVariantSizeThresholds | None = None,
) -> list[ThresholdVariantPerImageSummary]:
    """Summarize object measurements per image for one variant."""
    thresholds = size_thresholds or ThresholdVariantSizeThresholds()
    measurements_path = Path(measurements_dir)
    if not success:
        return []

    load_result = load_cellprofiler_measurements_lenient(measurements_path)
    object_tables = _find_object_measurement_tables(
        load_result.tables,
        load_result.metadata,
    )
    if not object_tables:
        return []

    combined = _combine_object_tables(object_tables)
    image_column = _image_number_column(combined)
    if image_column is None:
        return []

    name_lookup = _image_name_lookup(load_result.tables, load_result.metadata)
    image_numbers = sorted(int(value) for value in combined[image_column].dropna().unique())

    per_image: list[ThresholdVariantPerImageSummary] = []
    staged_path = Path(subset_dir) if subset_dir is not None else None
    ordered_names = list(image_names or [])

    for image_number in image_numbers:
        image_name = name_lookup.get(image_number)
        if image_name is None and ordered_names and 1 <= image_number <= len(ordered_names):
            image_name = ordered_names[image_number - 1]

        image_pixels = None
        if staged_path is not None and image_name is not None:
            image_pixels = _image_pixel_count(staged_path / image_name)

        summary = _summarize_object_table_per_image(
            combined,
            image_number=image_number,
            image_name=image_name,
            image_pixels=image_pixels,
            size_thresholds=thresholds,
        )
        per_image.append(
            ThresholdVariantPerImageSummary(
                variant_id=variant_id,
                display_name=display_name,
                image_number=image_number,
                image_name=summary.image_name,
                success=True,
                object_count=summary.object_count,
                median_area=summary.median_area,
                tiny_frac=summary.tiny_frac,
                huge_frac=summary.huge_frac,
                normal_frac=summary.normal_frac,
                foreground_coverage=summary.foreground_coverage,
                warnings=list(summary.warnings),
            )
        )
    return per_image


def _apply_baseline_per_image_ratios(
    per_image_rows: Sequence[ThresholdVariantPerImageSummary],
    *,
    size_thresholds: ThresholdVariantSizeThresholds,
) -> list[ThresholdVariantPerImageSummary]:
    baseline_rows = {
        row.image_number: row
        for row in per_image_rows
        if "baseline" in row.variant_id.lower()
    }
    updated: list[ThresholdVariantPerImageSummary] = []
    for row in per_image_rows:
        warnings = list(row.warnings)
        ratio = None
        baseline = baseline_rows.get(row.image_number)
        if baseline is not None and baseline.object_count and baseline.object_count > 0:
            if row.object_count is not None:
                ratio = row.object_count / baseline.object_count
                if ratio > size_thresholds.object_count_ratio_max:
                    warnings.append(
                        f"Per-image object_count is {ratio:.1f}x baseline on "
                        f"{row.image_name or f'image {row.image_number}'}."
                    )
                elif ratio < size_thresholds.object_count_ratio_min:
                    warnings.append(
                        f"Per-image object_count is {ratio:.1f}x baseline on "
                        f"{row.image_name or f'image {row.image_number}'}."
                    )
        updated.append(
            ThresholdVariantPerImageSummary(
                variant_id=row.variant_id,
                display_name=row.display_name,
                image_number=row.image_number,
                image_name=row.image_name,
                success=row.success,
                object_count=row.object_count,
                median_area=row.median_area,
                tiny_frac=row.tiny_frac,
                huge_frac=row.huge_frac,
                normal_frac=row.normal_frac,
                foreground_coverage=row.foreground_coverage,
                object_count_ratio_vs_baseline=ratio,
                warnings=warnings,
            )
        )
    return updated


def _instability_warnings(
    per_image_rows: Sequence[ThresholdVariantPerImageSummary],
    *,
    size_thresholds: ThresholdVariantSizeThresholds,
) -> list[str]:
    by_variant: dict[str, list[int]] = {}
    for row in per_image_rows:
        if row.object_count is None:
            continue
        by_variant.setdefault(row.variant_id, []).append(row.object_count)

    warnings: list[str] = []
    for variant_id, counts in by_variant.items():
        if len(counts) < 2:
            continue
        mean_count = float(np.mean(counts))
        if mean_count <= 0:
            continue
        cv = float(np.std(counts) / mean_count)
        if cv >= size_thresholds.count_instability_cv:
            warnings.append(
                f"Variant {variant_id}: object counts vary strongly across subset "
                f"images (CV={cv:.2f})."
            )
    return warnings


def compare_threshold_variant_per_image(
    run_results: Iterable[ThresholdVariantRunResult],
    *,
    subset_dir: str | Path | None = None,
    image_names: Sequence[str] | None = None,
    size_thresholds: ThresholdVariantSizeThresholds | None = None,
) -> list[ThresholdVariantPerImageSummary]:
    """Build per-image summaries for all variant runs."""
    thresholds = size_thresholds or ThresholdVariantSizeThresholds()
    rows: list[ThresholdVariantPerImageSummary] = []
    for result in run_results:
        rows.extend(
            summarize_threshold_variant_per_image(
                result.measurements_dir,
                variant_id=result.spec.variant_id,
                display_name=result.spec.display_name,
                success=result.success,
                subset_dir=subset_dir,
                image_names=image_names,
                size_thresholds=thresholds,
            )
        )
    rows = _apply_baseline_per_image_ratios(rows, size_thresholds=thresholds)
    for warning in _instability_warnings(rows, size_thresholds=thresholds):
        for row in rows:
            if row.variant_id in warning:
                row.warnings.append(warning)
                break
    return rows


def threshold_variant_per_image_to_dataframe(
    rows: Sequence[ThresholdVariantPerImageSummary],
) -> pd.DataFrame:
    """Convert per-image summaries to a table."""
    payload = [
        {
            "variant_id": row.variant_id,
            "name": row.display_name,
            "image_number": row.image_number,
            "image_name": row.image_name,
            "success": row.success,
            "object_count": row.object_count,
            "object_count_ratio_vs_baseline": row.object_count_ratio_vs_baseline,
            "median_area": row.median_area,
            "tiny_frac": row.tiny_frac,
            "normal_frac": row.normal_frac,
            "huge_frac": row.huge_frac,
            "foreground_coverage": row.foreground_coverage,
            "warnings": "; ".join(row.warnings),
        }
        for row in rows
    ]
    return pd.DataFrame(payload)


def save_threshold_variant_per_image_comparison(
    rows: Sequence[ThresholdVariantPerImageSummary],
    output_dir: str | Path,
    *,
    basename: str = "threshold_variant_per_image_comparison",
) -> dict[str, Path]:
    """Write per-image comparison CSV and JSON."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = (destination / f"{basename}.csv").resolve()
    json_path = (destination / f"{basename}.json").resolve()
    dataframe = threshold_variant_per_image_to_dataframe(rows)
    dataframe.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps([row.to_dict() for row in rows], indent=2),
        encoding="utf-8",
    )
    return {"csv": csv_path, "json": json_path}


def load_threshold_variant_per_image_comparison(
    path: str | Path,
) -> list[ThresholdVariantPerImageSummary]:
    """Load per-image comparison rows from a saved JSON file."""
    json_path = Path(path)
    if json_path.suffix.lower() != ".json":
        json_path = json_path.with_suffix(".json")
    if not json_path.is_file():
        return []

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []

    rows: list[ThresholdVariantPerImageSummary] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        rows.append(
            ThresholdVariantPerImageSummary(
                variant_id=str(entry.get("variant_id", "")),
                display_name=str(entry.get("display_name", "")),
                image_number=int(entry.get("image_number", 0)),
                image_name=entry.get("image_name"),
                success=bool(entry.get("success", True)),
                object_count=entry.get("object_count"),
                median_area=entry.get("median_area"),
                tiny_frac=entry.get("tiny_frac"),
                huge_frac=entry.get("huge_frac"),
                normal_frac=entry.get("normal_frac"),
                foreground_coverage=entry.get("foreground_coverage"),
                object_count_ratio_vs_baseline=entry.get(
                    "object_count_ratio_vs_baseline"
                ),
                warnings=list(entry.get("warnings", [])),
            )
        )
    return rows


def save_threshold_variant_comparison(
    summaries: Sequence[ThresholdVariantMeasurementSummary],
    output_dir: str | Path,
    *,
    basename: str = "threshold_variant_comparison",
) -> dict[str, Path]:
    """Write the comparison table to CSV and JSON under ``output_dir``."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    csv_path = (destination / f"{basename}.csv").resolve()
    json_path = (destination / f"{basename}.json").resolve()

    dataframe = threshold_variant_comparison_to_dataframe(summaries)
    dataframe.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps([summary.to_dict() for summary in summaries], indent=2),
        encoding="utf-8",
    )

    return {"csv": csv_path, "json": json_path}
