"""Batch processing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from bioimage_pipeline.export import (
    export_intensity_tiff,
    export_label_tiff,
    export_mask_tiff,
    export_measurements_csv,
)
from bioimage_pipeline.io import read_tiff
from bioimage_pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from bioimage_pipeline.stack import ImageStack


def _collect_image_paths(input_folder: Path, pattern: str) -> list[Path]:
    paths = sorted(input_folder.glob(pattern))
    if pattern == "*.tif":
        paths.extend(sorted(input_folder.glob("*.tiff")))
    return [path for path in paths if path.is_file()]


def run_pipeline_on_folder(
    pipeline: Pipeline,
    input_folder: str | Path,
    output_folder: str | Path,
    pattern: str = "*.tif",
) -> dict[str, Any]:
    """Run a pipeline on every TIFF image in a folder.

    Args:
        pipeline: Pipeline instance to run on each image.
        input_folder: Folder containing input TIFF images.
        output_folder: Folder where outputs are written.
        pattern: Glob pattern for input images.

    Returns:
        Dictionary with ``processed`` and ``failed`` lists.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.is_dir():
        raise ValueError(f"Input folder does not exist: {input_path}")

    image_paths = _collect_image_paths(input_path, pattern)
    processed: list[str] = []
    failed: list[dict[str, str]] = []
    combined_frames: list[pd.DataFrame] = []

    for image_path in image_paths:
        try:
            image = read_tiff(image_path)
            data = pipeline.run({"image": image, "filename": image_path.name})
            stem = image_path.stem

            if "mask" in data:
                export_mask_tiff(output_path / f"{stem}_mask.tif", data["mask"])
            if "labels" in data:
                export_label_tiff(output_path / f"{stem}_labels.tif", data["labels"])
            if "measurements" in data:
                measurements = data["measurements"].copy()
                measurements.insert(0, "filename", image_path.name)
                export_measurements_csv(
                    output_path / f"{stem}_measurements.csv",
                    measurements,
                )
                combined_frames.append(measurements)

            processed.append(image_path.name)
        except Exception as exc:
            failed.append({"filename": image_path.name, "error": str(exc)})

    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        export_measurements_csv(output_path / "all_measurements.csv", combined)

    return {"processed": processed, "failed": failed}


def run_pipeline_on_stack(
    pipeline: Pipeline,
    stack: "ImageStack",
    output_dir: str | Path,
    *,
    export_processed: bool = False,
) -> dict[str, Any]:
    """Run a pipeline on every frame of an :class:`~bioimage_pipeline.stack.ImageStack`.

    Applies the same pipeline steps to each 2D frame and saves per-frame
    outputs.  Per-frame files are named ``{stem}_f{index:03d}_<kind>.tif``.
    A combined ``all_measurements.csv`` with ``stack_id``, ``frame_index``,
    ``z_index``, and ``filename`` columns is written when at least one frame
    produces measurements.

    Args:
        pipeline: :class:`~bioimage_pipeline.pipeline.Pipeline` to run on each frame.
        stack: :class:`~bioimage_pipeline.stack.ImageStack` (from a file or folder).
        output_dir: Folder where outputs are written.
        export_processed: When ``True``, also write the ``"processed"`` pipeline
            key (blurred / corrected image) as ``{stem}_f{index:03d}_processed.tif``.

    Returns:
        Dictionary with:

        * ``processed`` — list of successfully processed frame stems.
        * ``failed``    — list of dicts with ``frame_index``, ``filename``, ``error``.
        * ``measurements`` — combined :class:`pandas.DataFrame` or ``None``.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed_names: list[str] = []
    failed: list[dict[str, Any]] = []
    combined_frames: list[pd.DataFrame] = []
    stack_id = stack.source.name if stack.source.is_file() else stack.source.name

    for frame in stack:
        if frame.source_path is not None and frame.source_path.is_file():
            stem = frame.source_path.stem
            filename = frame.source_path.name
        else:
            stem = f"frame_{frame.index:03d}"
            filename = stem

        frame_tag = f"f{frame.index:03d}"
        prefix = f"{stem}_{frame_tag}"

        try:
            data = pipeline.run(
                {
                    "image": frame.array,
                    "filename": filename,
                    "frame_index": frame.index,
                    "z_index": frame.z_index,
                    "t_index": frame.t_index,
                    "c_index": frame.c_index,
                }
            )

            if "mask" in data:
                export_mask_tiff(output_path / f"{prefix}_mask.tif", data["mask"])
            if "labels" in data:
                export_label_tiff(output_path / f"{prefix}_labels.tif", data["labels"])
            if export_processed and "processed" in data:
                export_intensity_tiff(
                    output_path / f"{prefix}_processed.tif", data["processed"]
                )
            if "measurements" in data:
                m = data["measurements"].copy()
                identity_cols: list[tuple[str, Any]] = [
                    ("stack_id", stack_id),
                    ("frame_index", frame.index),
                    ("z_index", frame.z_index),
                ]
                if frame.t_index is not None:
                    identity_cols.append(("t_index", frame.t_index))
                if frame.c_index is not None:
                    identity_cols.append(("c_index", frame.c_index))
                identity_cols.append(("filename", filename))
                for offset, (col_name, col_value) in enumerate(identity_cols):
                    m.insert(offset, col_name, col_value)
                export_measurements_csv(
                    output_path / f"{prefix}_measurements.csv", m
                )
                combined_frames.append(m)

            processed_names.append(stem)
        except Exception as exc:
            failed.append(
                {
                    "frame_index": frame.index,
                    "filename": filename,
                    "error": str(exc),
                }
            )

    measurements: pd.DataFrame | None = None
    if combined_frames:
        measurements = pd.concat(combined_frames, ignore_index=True)
        export_measurements_csv(output_path / "all_measurements.csv", measurements)

    return {"processed": processed_names, "failed": failed, "measurements": measurements}
