"""Post-projection validation and dtype normalization for CellProfiler input."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from bioimage_pipeline.io import extract_2d_plane, interpret_tiff_axes, read_tiff, save_tiff
from bioimage_pipeline.z_projection import (
    DEFAULT_Z_PROJECTION_METHOD,
    fiji_projection_label,
    normalize_projection_method,
)

logger = logging.getLogger(__name__)

PROJECTION_POSTPROCESS_LOG = "projection_postprocess.txt"
PROJECTION_POSTPROCESS_JSON = "projection_postprocess.json"
UINT16_MAX = 65535


@dataclass
class ProjectionPostprocessRecord:
    """Validation record for one projected TIFF."""

    path: str
    engine: str
    projection_method: str
    shape: tuple[int, ...]
    dtype_before: str
    min_before: float
    max_before: float
    nonzero_before: int
    dtype_after: str
    min_after: float
    max_after: float
    nonzero_after: int
    action: str
    rewritten: bool


@dataclass
class ProjectionTiffComparison:
    """Pixel-level comparison between two projected TIFFs."""

    reference_path: str
    candidate_path: str
    reference_shape: tuple[int, ...]
    candidate_shape: tuple[int, ...]
    reference_dtype: str
    candidate_dtype: str
    reference_min: float
    candidate_min: float
    reference_max: float
    candidate_max: float
    reference_nonzero: int
    candidate_nonzero: int
    projection_method: str
    channel_count: int
    max_abs_diff: float | None
    mean_abs_diff: float | None
    identical: bool


def _array_min_max(array: np.ndarray) -> tuple[float, float]:
    if array.size == 0:
        return 0.0, 0.0
    return float(np.min(array)), float(np.max(array))


def _nonzero_count(array: np.ndarray) -> int:
    return int(np.count_nonzero(array))


def _plane_for_stats(image: np.ndarray) -> np.ndarray:
    """Return one 2D plane for stats without changing on-disk layout."""
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        axes = interpret_tiff_axes(array.shape)
        if axes.c_count > 1 and axes.z_count == 1:
            return extract_2d_plane(array, frame_index=0)
        if axes.z_count > 1 and axes.c_count == 1:
            return extract_2d_plane(array, frame_index=0)
        return extract_2d_plane(array, frame_index=0)
    raise ValueError(f"Expected a 2D projected plane, got shape {array.shape}")


def _channel_count(image: np.ndarray) -> int:
    array = np.asarray(image)
    if array.ndim == 2:
        return 1
    if array.ndim == 3:
        axes = interpret_tiff_axes(array.shape)
        return max(1, axes.c_count)
    return 1


def describe_projected_tiff(path: str | Path) -> dict[str, Any]:
    """Return shape/dtype/range stats for one projected TIFF."""
    tiff_path = Path(path).resolve()
    image = read_tiff(tiff_path)
    plane = _plane_for_stats(image)
    min_value, max_value = _array_min_max(plane)
    return {
        "path": str(tiff_path),
        "shape": tuple(int(dim) for dim in image.shape),
        "dtype": str(plane.dtype),
        "min": min_value,
        "max": max_value,
        "nonzero": _nonzero_count(plane),
        "channel_count": _channel_count(image),
    }


def log_projection_output_diagnostics(
    *,
    engine: str,
    projection_method: str | None,
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Log selected engine/method and output TIFF stats after projection."""
    method = normalize_projection_method(projection_method)
    fiji_arg = fiji_projection_label(method)
    try:
        output_stats = describe_projected_tiff(output_path)
    except OSError as exc:
        logger.warning(
            "Could not read projected TIFF for diagnostics: %s (%s)",
            Path(output_path).resolve(),
            exc,
        )
        output_stats = {
            "path": str(Path(output_path).resolve()),
            "shape": (),
            "dtype": "unreadable",
            "min": None,
            "max": None,
            "nonzero": None,
            "channel_count": None,
        }
    payload = {
        "engine": engine,
        "projection_method": method,
        "fiji_projection_argument": fiji_arg,
        "input_path": str(Path(input_path).resolve()),
        "output_path": str(Path(output_path).resolve()),
        **output_stats,
    }
    logger.info(
        "OIR projection output: engine=%s method=%s fiji_arg=%s "
        "input=%s output=%s shape=%s dtype=%s min=%s max=%s nonzero=%s channels=%s",
        engine,
        method,
        fiji_arg,
        payload["input_path"],
        payload["output_path"],
        payload["shape"],
        payload["dtype"],
        payload["min"],
        payload["max"],
        payload["nonzero"],
        payload["channel_count"],
    )
    return payload


def compare_projected_tiffs(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    projection_method: str | None = DEFAULT_Z_PROJECTION_METHOD,
) -> ProjectionTiffComparison:
    """Compare candidate output against a reference Fiji/manual TIFF."""
    reference = read_tiff(reference_path)
    candidate = read_tiff(candidate_path)
    ref_plane = _plane_for_stats(reference)
    cand_plane = _plane_for_stats(candidate)
    method = normalize_projection_method(projection_method)

    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None
    identical = False
    if ref_plane.shape == cand_plane.shape:
        diff = np.abs(
            cand_plane.astype(np.float64) - ref_plane.astype(np.float64)
        )
        max_abs_diff = float(np.max(diff)) if diff.size else 0.0
        mean_abs_diff = float(np.mean(diff)) if diff.size else 0.0
        identical = bool(np.array_equal(ref_plane, cand_plane))

    ref_min, ref_max = _array_min_max(ref_plane)
    cand_min, cand_max = _array_min_max(cand_plane)
    return ProjectionTiffComparison(
        reference_path=str(Path(reference_path).resolve()),
        candidate_path=str(Path(candidate_path).resolve()),
        reference_shape=tuple(int(dim) for dim in reference.shape),
        candidate_shape=tuple(int(dim) for dim in candidate.shape),
        reference_dtype=str(ref_plane.dtype),
        candidate_dtype=str(cand_plane.dtype),
        reference_min=ref_min,
        candidate_min=cand_min,
        reference_max=ref_max,
        candidate_max=cand_max,
        reference_nonzero=_nonzero_count(ref_plane),
        candidate_nonzero=_nonzero_count(cand_plane),
        projection_method=method,
        channel_count=max(_channel_count(reference), _channel_count(candidate)),
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        identical=identical,
    )


def format_projection_comparison_report(comparison: ProjectionTiffComparison) -> str:
    """Format a human-readable TIFF comparison report."""
    lines = [
        "Projection TIFF comparison:",
        f"  reference_path: {comparison.reference_path}",
        f"  candidate_path: {comparison.candidate_path}",
        f"  projection_method: {comparison.projection_method}",
        f"  reference_shape: {comparison.reference_shape}",
        f"  candidate_shape: {comparison.candidate_shape}",
        f"  reference_dtype: {comparison.reference_dtype}",
        f"  candidate_dtype: {comparison.candidate_dtype}",
        f"  reference_min: {comparison.reference_min}",
        f"  candidate_min: {comparison.candidate_min}",
        f"  reference_max: {comparison.reference_max}",
        f"  candidate_max: {comparison.candidate_max}",
        f"  reference_nonzero: {comparison.reference_nonzero}",
        f"  candidate_nonzero: {comparison.candidate_nonzero}",
        f"  channel_count: {comparison.channel_count}",
        f"  identical: {comparison.identical}",
        f"  max_abs_diff: {comparison.max_abs_diff}",
        f"  mean_abs_diff: {comparison.mean_abs_diff}",
    ]
    return "\n".join(lines)


def normalize_projected_image_for_cellprofiler(
    image: np.ndarray,
    *,
    projection_method: str | None = None,
) -> tuple[np.ndarray, str]:
    """Convert a projected image to uint16 for CellProfiler compatibility."""
    _ = normalize_projection_method(projection_method)
    plane = _plane_for_stats(image)
    vmin, vmax = _array_min_max(plane)

    if plane.dtype == np.uint16:
        return plane, "unchanged_uint16"

    if plane.dtype == np.uint8:
        return plane.astype(np.uint16), "uint8_promoted_to_uint16"

    if np.issubdtype(plane.dtype, np.floating):
        non_negative = np.clip(plane, 0.0, None)
        adj_min, adj_max = _array_min_max(non_negative)
        if adj_max <= 1.0:
            scaled = np.clip(non_negative * UINT16_MAX, 0, UINT16_MAX)
            return scaled.astype(np.uint16), "float01_scaled_to_uint16"
        if adj_max <= UINT16_MAX:
            return np.clip(np.rint(non_negative), 0, UINT16_MAX).astype(
                np.uint16
            ), "float_intensity_clipped_to_uint16"
        scale = UINT16_MAX / adj_max if adj_max > 0 else 1.0
        scaled = np.clip(non_negative.astype(np.float64) * scale, 0, UINT16_MAX)
        return scaled.astype(np.uint16), "float_intensity_scaled_to_uint16"

    if np.issubdtype(plane.dtype, np.integer):
        if vmax <= UINT16_MAX and vmin >= 0:
            return plane.astype(np.uint16), "integer_cast_to_uint16"
        clipped = np.clip(plane, 0, None)
        clip_max = float(np.max(clipped)) if clipped.size else 0.0
        if clip_max <= UINT16_MAX:
            return clipped.astype(np.uint16), "integer_clipped_to_uint16"
        scale = UINT16_MAX / clip_max if clip_max > 0 else 1.0
        scaled = np.clip(clipped.astype(np.float64) * scale, 0, UINT16_MAX)
        return scaled.astype(np.uint16), "integer_scaled_to_uint16"

    return plane.astype(np.uint16), "cast_to_uint16"


def validate_and_normalize_projected_tiff(
    path: str | Path,
    *,
    engine: str,
    projection_method: str | None = None,
    audit_logs_dir: str | Path | None = None,
) -> ProjectionPostprocessRecord:
    """Read one projected TIFF, validate dtype/range, and normalize when needed."""
    tiff_path = Path(path).resolve()
    method = normalize_projection_method(projection_method)
    original = read_tiff(tiff_path)
    plane = _plane_for_stats(original)
    dtype_before = str(plane.dtype)
    min_before, max_before = _array_min_max(plane)
    nonzero_before = _nonzero_count(plane)
    shape = tuple(int(dim) for dim in original.shape)

    if engine == "fiji":
        action = "fiji_preserved"
        normalized = plane
        rewritten = False
    else:
        normalized, action = normalize_projected_image_for_cellprofiler(
            original,
            projection_method=method,
        )
        rewritten = (
            dtype_before != str(normalized.dtype)
            or not np.array_equal(plane, normalized)
            or original.ndim != normalized.ndim
        )
        if rewritten:
            save_tiff(
                tiff_path,
                normalized,
                audit_logs_dir=audit_logs_dir,
            )

    min_after, max_after = _array_min_max(normalized)
    dtype_after = str(normalized.dtype)
    nonzero_after = _nonzero_count(normalized)

    return ProjectionPostprocessRecord(
        path=str(tiff_path),
        engine=engine,
        projection_method=method,
        shape=shape,
        dtype_before=dtype_before,
        min_before=min_before,
        max_before=max_before,
        nonzero_before=nonzero_before,
        dtype_after=dtype_after,
        min_after=min_after,
        max_after=max_after,
        nonzero_after=nonzero_after,
        action=action,
        rewritten=rewritten,
    )


def format_projection_postprocess_report(
    records: Sequence[ProjectionPostprocessRecord],
    *,
    engine: str,
    projection_method: str,
) -> str:
    """Format post-projection validation records for logs."""
    lines = [
        "Projection postprocess report:",
        f"  engine: {engine}",
        f"  projection_method: {normalize_projection_method(projection_method)}",
        f"  fiji_projection_argument: {fiji_projection_label(projection_method)}",
        f"  files_validated: {len(records)}",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"  - path: {record.path}",
                f"    shape: {record.shape}",
                f"    dtype_before: {record.dtype_before}",
                f"    min_before: {record.min_before}",
                f"    max_before: {record.max_before}",
                f"    nonzero_before: {record.nonzero_before}",
                f"    dtype_after: {record.dtype_after}",
                f"    min_after: {record.min_after}",
                f"    max_after: {record.max_after}",
                f"    nonzero_after: {record.nonzero_after}",
                f"    action: {record.action}",
                f"    rewritten: {record.rewritten}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def validate_projected_tiffs_for_cellprofiler(
    output_tifs: Sequence[str | Path],
    *,
    engine: str,
    projection_method: str | None = None,
    logs_dir: str | Path | None = None,
) -> list[ProjectionPostprocessRecord]:
    """Validate projected TIFFs before CellProfiler without altering Fiji output."""
    method = normalize_projection_method(projection_method)
    fiji_arg = fiji_projection_label(method)
    records: list[ProjectionPostprocessRecord] = []

    logger.info(
        "Projection postprocess start: engine=%s method=%s fiji_arg=%s files=%d",
        engine,
        method,
        fiji_arg,
        len(output_tifs),
    )

    for output_tif in output_tifs:
        tiff_path = Path(output_tif)
        if not tiff_path.is_file():
            continue
        record = validate_and_normalize_projected_tiff(
            tiff_path,
            engine=engine,
            projection_method=method,
            audit_logs_dir=logs_dir,
        )
        records.append(record)
        logger.info(
            "Projection postprocess %s: engine=%s method=%s fiji_arg=%s "
            "shape=%s dtype %s->%s range [%.3g, %.3g] -> [%.3g, %.3g] "
            "nonzero %d->%d action=%s rewritten=%s",
            tiff_path.name,
            engine,
            method,
            fiji_arg,
            record.shape,
            record.dtype_before,
            record.dtype_after,
            record.min_before,
            record.max_before,
            record.min_after,
            record.max_after,
            record.nonzero_before,
            record.nonzero_after,
            record.action,
            record.rewritten,
        )

    if logs_dir is not None:
        logs_path = Path(logs_dir)
        logs_path.mkdir(parents=True, exist_ok=True)
        report = format_projection_postprocess_report(
            records,
            engine=engine,
            projection_method=method,
        )
        (logs_path / PROJECTION_POSTPROCESS_LOG).write_text(report, encoding="utf-8")
        payload: dict[str, Any] = {
            "engine": engine,
            "projection_method": method,
            "fiji_projection_argument": fiji_arg,
            "files_validated": len(records),
            "records": [asdict(record) for record in records],
        }
        (logs_path / PROJECTION_POSTPROCESS_JSON).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    return records
