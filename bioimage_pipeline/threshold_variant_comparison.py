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
