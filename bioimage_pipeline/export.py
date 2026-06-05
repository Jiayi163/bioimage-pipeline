"""Export helpers for TIFF and CSV output.

Python ImageJ-compatible TIFF writing (``fiji_tiff``) is the **fallback /
intermediate** export path (Phase 12). Production final TIFF export through
headless Fiji/ImageJ is Phase 14 (``fiji_runner.py``, planned).

Functions such as :func:`organize_cellprofiler_tiffs_for_fiji` reorganize
CellProfiler outputs via an **in-process** Python loop (acceptable fallback —
no CellProfiler or Fiji relaunch). Production export should use **one Fiji batch
run per folder** (Phase 14 ``fiji_runner.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from bioimage_pipeline.fiji_tiff import (
    TiffExportMetadata,
    prepare_intensity_for_export,
    prepare_labels_for_export,
    prepare_mask_for_export,
    save_fiji_compatible_tiff,
)
from bioimage_pipeline.cellprofiler_runner import discover_cellprofiler_tiff_files
from bioimage_pipeline.io import read_tiff

FijiImageKind = Literal["mask", "label", "intensity"]


@dataclass
class OrganizedFijiExports:
    """Fiji-compatible TIFF exports grouped by image kind."""

    masks: list[Path]
    labels: list[Path]
    intensity: list[Path]

    @property
    def all_exports(self) -> list[Path]:
        return [*self.masks, *self.labels, *self.intensity]


def export_mask_tiff(
    path: str | Path,
    mask: np.ndarray,
    *,
    metadata: TiffExportMetadata | None = None,
) -> Path:
    """Save a boolean mask as a Fiji-friendly 0/255 uint8 TIFF."""
    fiji_mask = prepare_mask_for_export(mask)
    return save_fiji_compatible_tiff(path, fiji_mask, metadata=metadata)


def export_label_tiff(
    path: str | Path,
    labels: np.ndarray,
    *,
    metadata: TiffExportMetadata | None = None,
) -> Path:
    """Save a labeled image as a Fiji-friendly unsigned integer TIFF."""
    label_image = prepare_labels_for_export(labels)
    return save_fiji_compatible_tiff(path, label_image, metadata=metadata)


def export_intensity_tiff(
    path: str | Path,
    image: np.ndarray,
    *,
    metadata: TiffExportMetadata | None = None,
) -> Path:
    """Save an intensity image, preserving safe integer dtypes when possible."""
    intensity_image = prepare_intensity_for_export(image)
    return save_fiji_compatible_tiff(path, intensity_image, metadata=metadata)


def export_measurements_csv(path: str | Path, dataframe: pd.DataFrame) -> None:
    """Save measurement results to a CSV file readable by Excel."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def classify_tiff_for_fiji_export(
    image: np.ndarray,
    *,
    filename: str = "",
) -> FijiImageKind:
    """Guess whether a CellProfiler TIFF is a mask, label image, or intensity."""
    array = np.asarray(image)
    name_lower = filename.lower()

    if "mask" in name_lower and "labeled" not in name_lower:
        return "mask"
    if any(keyword in name_lower for keyword in ("label", "objects", "segmented")):
        return "label"

    unique = np.unique(array)
    if len(unique) <= 3 and set(unique.tolist()).issubset({0, 1, 255}):
        return "mask"
    if np.issubdtype(array.dtype, np.integer) and len(unique) > 2 and int(unique.max()) > 1:
        return "label"
    return "intensity"


def export_tiff_for_fiji(
    path: str | Path,
    image: np.ndarray,
    *,
    kind: FijiImageKind | None = None,
    metadata: TiffExportMetadata | None = None,
) -> Path:
    """Save a TIFF using the Fiji-friendly export path for the detected image kind."""
    filename = Path(path).name
    resolved_kind = kind or classify_tiff_for_fiji_export(image, filename=filename)

    if resolved_kind == "mask":
        return export_mask_tiff(path, image, metadata=metadata)
    if resolved_kind == "label":
        return export_label_tiff(path, image, metadata=metadata)
    return export_intensity_tiff(path, image, metadata=metadata)


def export_cellprofiler_tiff_for_fiji(
    source_path: str | Path,
    dest_path: str | Path,
    *,
    kind: FijiImageKind | None = None,
    metadata: TiffExportMetadata | None = None,
) -> Path:
    """Read a CellProfiler TIFF and rewrite it in Fiji-friendly format."""
    image = read_tiff(source_path)
    return export_tiff_for_fiji(
        dest_path,
        image,
        kind=kind,
        metadata=metadata,
    )


def organize_cellprofiler_tiffs_for_fiji(
    output_dir: str | Path,
    masks_dir: str | Path,
    labels_dir: str | Path,
    *,
    pattern: str = "*.tif",
    exclude_dirs: Sequence[str] | None = None,
) -> OrganizedFijiExports:
    """Convert CellProfiler TIFF outputs into organized folders (Python fallback).

    Re-exports mask and label TIFFs via :func:`export_tiff_for_fiji`. When Phase 14
    Fiji headless export is available, the workflow should prefer that path for
    final outputs and use this function only as fallback.
    """
    masks_root = Path(masks_dir)
    labels_root = Path(labels_dir)
    masks_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    masks: list[Path] = []
    labels: list[Path] = []
    intensity: list[Path] = []

    discovery_kwargs: dict[str, object] = {"pattern": pattern}
    if exclude_dirs is not None:
        discovery_kwargs["exclude_dirs"] = exclude_dirs

    for tiff_path in discover_cellprofiler_tiff_files(output_dir, **discovery_kwargs):
        image = read_tiff(tiff_path)
        kind = classify_tiff_for_fiji_export(image, filename=tiff_path.name)
        if kind == "mask":
            destination = masks_root / tiff_path.name
            masks.append(export_tiff_for_fiji(destination, image, kind="mask"))
        elif kind == "label":
            destination = labels_root / tiff_path.name
            labels.append(export_tiff_for_fiji(destination, image, kind="label"))
        else:
            intensity.append(tiff_path.resolve())

    return OrganizedFijiExports(masks=masks, labels=labels, intensity=intensity)


def export_cellprofiler_images_for_fiji(
    output_dir: str | Path,
    export_dir: str | Path,
    *,
    pattern: str = "*.tif",
    exclude_dirs: Sequence[str] = ("fiji_exports",),
) -> list[Path]:
    """Convert CellProfiler ``SaveImages`` TIFF outputs to Fiji-friendly files.

    Legacy helper that writes all exports into one folder. Prefer
    :func:`organize_cellprofiler_tiffs_for_fiji` for structured results.
    """
    export_root = Path(export_dir)
    export_root.mkdir(parents=True, exist_ok=True)

    exported: list[Path] = []
    for tiff_path in discover_cellprofiler_tiff_files(
        output_dir,
        pattern=pattern,
        exclude_dirs=exclude_dirs,
    ):
        destination = export_root / tiff_path.name
        exported.append(export_cellprofiler_tiff_for_fiji(tiff_path, destination))
    return exported
