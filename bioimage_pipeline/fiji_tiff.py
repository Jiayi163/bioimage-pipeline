"""Fiji/ImageJ-compatible TIFF export helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


@dataclass
class TiffExportMetadata:
    """Optional metadata embedded in ImageJ-compatible TIFF files."""

    pixel_size_x: float | None = None
    pixel_size_y: float | None = None
    unit: str = "um"
    channel_name: str | None = None
    description: str | None = None


def label_export_dtype(labels: np.ndarray) -> np.dtype:
    """Choose a Fiji-friendly unsigned integer dtype for label images."""
    label_arr = np.asarray(labels)
    if label_arr.size == 0:
        return np.dtype(np.uint16)

    max_label = int(label_arr.max())
    if max_label <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def prepare_mask_for_export(mask: np.ndarray) -> np.ndarray:
    """Convert a boolean mask to Fiji-friendly uint8 values (0/255)."""
    mask_arr = np.asarray(mask).astype(bool)
    return mask_arr.astype(np.uint8) * 255


def prepare_labels_for_export(labels: np.ndarray) -> np.ndarray:
    """Convert a label image to Fiji-friendly unsigned integer dtype."""
    label_arr = np.asarray(labels)
    dtype = label_export_dtype(label_arr)
    return label_arr.astype(dtype)


def prepare_intensity_for_export(image: np.ndarray) -> np.ndarray:
    """Preserve safe integer dtypes; cast unsupported types to float32."""
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.integer) or array.dtype == np.bool_:
        return array
    if array.dtype == np.float32:
        return array
    return array.astype(np.float32)


def _build_imagej_metadata(metadata: TiffExportMetadata | None) -> dict[str, Any]:
    if metadata is None:
        return {}

    imagej_metadata: dict[str, Any] = {"unit": metadata.unit}
    if metadata.channel_name:
        imagej_metadata["Labels"] = [metadata.channel_name]
    if metadata.description:
        imagej_metadata["Info"] = metadata.description
    return imagej_metadata


def _build_resolution(metadata: TiffExportMetadata | None) -> tuple[float, float] | None:
    if metadata is None:
        return None
    if metadata.pixel_size_x is None or metadata.pixel_size_y is None:
        return None
    if metadata.pixel_size_x <= 0 or metadata.pixel_size_y <= 0:
        raise ValueError("pixel_size_x and pixel_size_y must be positive")
    return (1.0 / metadata.pixel_size_x, 1.0 / metadata.pixel_size_y)


def save_fiji_compatible_tiff(
    path: str | Path,
    image: np.ndarray,
    *,
    metadata: TiffExportMetadata | None = None,
    imagej: bool = True,
) -> Path:
    """Save a TIFF intended for inspection in Fiji/ImageJ.

    Uses ImageJ-compatible TIFF tags when ``imagej=True``. For ordinary TIFF
    without ImageJ metadata, set ``imagej=False``.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a NumPy array")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_kwargs: dict[str, Any] = {}
    use_imagej = imagej and image.dtype != np.uint32
    if use_imagej:
        write_kwargs["imagej"] = True
        imagej_metadata = _build_imagej_metadata(metadata)
        if imagej_metadata:
            write_kwargs["metadata"] = imagej_metadata
        resolution = _build_resolution(metadata)
        if resolution is not None:
            write_kwargs["resolution"] = resolution
            write_kwargs["resolutionunit"] = "NONE"

    try:
        tifffile.imwrite(output_path, image, **write_kwargs)
    except Exception as exc:
        raise OSError(f"Could not save TIFF file: {output_path}") from exc

    return output_path.resolve()


def read_fiji_tiff_metadata(path: str | Path) -> dict[str, Any]:
    """Read ImageJ/Fiji metadata from a TIFF file when present."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"TIFF file not found: {image_path}")

    with tifffile.TiffFile(image_path) as tiff:
        imagej_metadata = tiff.imagej_metadata or {}
        page = tiff.pages[0]
        resolution = page.tags.get("XResolution")
        resolution_unit = page.tags.get("ResolutionUnit")

    result: dict[str, Any] = {
        "imagej_metadata": dict(imagej_metadata),
        "resolution": None,
        "resolution_unit": None,
    }

    if resolution is not None:
        values = resolution.value
        if isinstance(values, tuple) and len(values) == 2 and values[1]:
            pixels_per_unit = values[0] / values[1]
            result["resolution"] = float(pixels_per_unit)
    if resolution_unit is not None:
        result["resolution_unit"] = resolution_unit.value

    if metadata := result["imagej_metadata"]:
        unit = metadata.get("unit")
        if result["resolution"] and unit:
            result["pixel_size"] = 1.0 / result["resolution"]
            result["unit"] = unit
        if labels := metadata.get("Labels"):
            if isinstance(labels, str):
                result["channel_name"] = labels
            elif isinstance(labels, (list, tuple)) and labels:
                result["channel_name"] = labels[0]
        if info := metadata.get("Info"):
            result["description"] = info

    return result


def metadata_to_dict(metadata: TiffExportMetadata) -> dict[str, Any]:
    """Serialize export metadata for reporting or JSON output."""
    return asdict(metadata)
