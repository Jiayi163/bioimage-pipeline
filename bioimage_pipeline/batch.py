"""Batch processing helpers."""

from pathlib import Path
from typing import Any

import pandas as pd

from bioimage_pipeline.export import (
    export_label_tiff,
    export_mask_tiff,
    export_measurements_csv,
)
from bioimage_pipeline.io import read_tiff
from bioimage_pipeline.pipeline import Pipeline


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
