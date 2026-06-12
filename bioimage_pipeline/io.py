"""TIFF image input/output helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import tifffile

from bioimage_pipeline.fiji_tiff import TiffExportMetadata, save_fiji_compatible_tiff


def read_tiff(path: str | Path) -> np.ndarray[Any, Any]:
    """Read a TIFF image from disk.

    Args:
        path: Path to the TIFF image.

    Returns:
        The image data as a NumPy array.

    Raises:
        FileNotFoundError: If the input path does not exist.
        ValueError: If the input path is not a file.
        OSError: If the TIFF file cannot be read.
    """
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"TIFF file not found: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"TIFF path is not a file: {image_path}")

    try:
        return tifffile.imread(image_path)
    except Exception as exc:
        raise OSError(f"Could not read TIFF file: {image_path}") from exc


def save_tiff(
    path: str | Path,
    image: np.ndarray[Any, Any],
    *,
    metadata: TiffExportMetadata | None = None,
    imagej_compatible: bool = False,
    audit_logs_dir: str | Path | None = None,
) -> Path:
    """Save image data to a TIFF file.

    Args:
        path: Destination path for the TIFF image.
        image: Image data to save.
        metadata: Optional ImageJ/Fiji metadata (requires
            ``imagej_compatible=True``).
        imagej_compatible: When ``True``, write ImageJ-compatible TIFF tags.

    Raises:
        ValueError: If image is not a NumPy array.
        OSError: If the TIFF file cannot be written.
    """
    if imagej_compatible:
        return save_fiji_compatible_tiff(path, image, metadata=metadata)

    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a NumPy array")

    image_path = Path(path)
    try:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(image_path, image)
    except Exception as exc:
        raise OSError(f"Could not save TIFF file: {image_path}") from exc
    resolved = image_path.resolve()
    if audit_logs_dir is not None:
        from bioimage_pipeline.oir_projection_lifecycle import (
            is_oir_projection_path,
            log_oir_projection_audit,
        )

        if is_oir_projection_path(resolved):
            log_oir_projection_audit(
                audit_logs_dir,
                "write_tif",
                {"path": str(resolved)},
            )
    return resolved


# ---------------------------------------------------------------------------
# Stack / multi-frame TIFF support (Phase 3.1)
# ---------------------------------------------------------------------------


@dataclass
class AxisInfo:
    """Axis interpretation for a multi-dimensional TIFF array.

    All counts are 1 for a plain 2D image.  ``frame_count`` is the total
    number of 2D planes (Z × T × C) that :func:`iter_stack_frames` will yield.
    """

    height: int
    width: int
    z_count: int = 1
    t_count: int = 1
    c_count: int = 1
    source: str = "inferred"

    @property
    def frame_count(self) -> int:
        return self.z_count * self.t_count * self.c_count


@dataclass
class StackFrame:
    """A single 2D plane extracted from an image stack.

    ``index`` is the sequential position (0-based) across all frames.
    ``z_index``, ``t_index``, and ``c_index`` are the per-axis positions;
    they are ``None`` when axis information is unavailable.
    """

    index: int
    array: np.ndarray
    z_index: int | None = None
    t_index: int | None = None
    c_index: int | None = None
    source_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def interpret_tiff_axes(
    shape: tuple[int, ...],
    imagej_metadata: dict[str, Any] | None = None,
) -> AxisInfo:
    """Infer Z / T / C / H / W axis counts from array shape and ImageJ metadata.

    When ``imagej_metadata`` is provided (e.g. from ``tifffile.TiffFile``),
    the ``slices``, ``frames``, and ``channels`` keys are used directly.
    Otherwise the shape alone is used with the following conventions:

    * 2D ``(H, W)`` → single frame.
    * 3D ``(N, H, W)`` → N Z-planes, one time point, one channel.
    * 4D ``(Z, C, H, W)`` → Z Z-planes, C channels.
    * 5D ``(T, Z, C, H, W)`` → full hyperstack.
    * Higher → all leading dimensions are collapsed into Z.

    Args:
        shape: Array shape from ``numpy.ndarray.shape``.
        imagej_metadata: Optional dict from ``tifffile.TiffFile.imagej_metadata``.

    Returns:
        :class:`AxisInfo` with per-axis counts and height/width.

    Raises:
        ValueError: If the shape has fewer than 2 dimensions.
    """
    if len(shape) < 2:
        raise ValueError(f"Image shape must have at least 2 dimensions, got {shape}")

    height, width = shape[-2], shape[-1]
    leading = shape[:-2]

    if imagej_metadata is not None:
        z = int(imagej_metadata.get("slices", 1))
        t = int(imagej_metadata.get("frames", 1))
        c = int(imagej_metadata.get("channels", 1))
        return AxisInfo(
            height=height,
            width=width,
            z_count=max(1, z),
            t_count=max(1, t),
            c_count=max(1, c),
            source="imagej_metadata",
        )

    if not leading:
        return AxisInfo(height=height, width=width, source="inferred")

    if len(leading) == 1:
        return AxisInfo(
            height=height, width=width, z_count=leading[0], source="inferred"
        )

    if len(leading) == 2:
        return AxisInfo(
            height=height,
            width=width,
            z_count=leading[0],
            c_count=leading[1],
            source="inferred",
        )

    if len(leading) == 3:
        return AxisInfo(
            height=height,
            width=width,
            t_count=leading[0],
            z_count=leading[1],
            c_count=leading[2],
            source="inferred",
        )

    total = int(np.prod(leading))
    return AxisInfo(height=height, width=width, z_count=total, source="inferred")


def extract_2d_plane(image: np.ndarray, *, frame_index: int = 0) -> np.ndarray:
    """Extract a single 2D plane from a potentially multi-dimensional array.

    For a 2D input the array is returned as-is.  For higher-dimensional arrays
    all leading dimensions are flattened and the plane at ``frame_index`` is
    returned.

    Args:
        image: Input array (any shape with at least 2 dimensions).
        frame_index: Sequential frame to extract when the image has more than
            one plane.  Defaults to 0 (first frame).

    Returns:
        2D NumPy array ``(H, W)``.

    Raises:
        ValueError: If the array has fewer than 2 dimensions or ``frame_index``
            is out of range.
    """
    arr = np.asarray(image)
    if arr.ndim < 2:
        raise ValueError(f"Image must have at least 2 dimensions, got {arr.ndim}D")
    if arr.ndim == 2:
        return arr

    h, w = arr.shape[-2], arr.shape[-1]
    flat = arr.reshape(-1, h, w)
    if frame_index < 0 or frame_index >= len(flat):
        raise ValueError(
            f"frame_index {frame_index} is out of range for {len(flat)} frame(s)"
        )
    return flat[frame_index]


def iter_stack_frames(path: str | Path) -> Iterator[StackFrame]:
    """Iterate over all 2D planes in a TIFF file, one :class:`StackFrame` at a time.

    Single-plane TIFFs yield exactly one frame.  Multi-page TIFFs (Z-stacks,
    time series, hyperstacks) yield one frame per page.  ImageJ hyperstack
    metadata is used when present to populate per-axis indices.

    Args:
        path: Path to a TIFF file (single-page or multi-page).

    Yields:
        :class:`StackFrame` instances in page order.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is not a file.
        OSError: If the TIFF cannot be opened.
    """
    tiff_path = Path(path)
    if not tiff_path.exists():
        raise FileNotFoundError(f"TIFF file not found: {tiff_path}")
    if not tiff_path.is_file():
        raise ValueError(f"TIFF path is not a file: {tiff_path}")

    try:
        with tifffile.TiffFile(tiff_path) as tif:
            imagej_meta: dict[str, Any] = tif.imagej_metadata or {}
            data = tif.asarray()
    except Exception as exc:
        raise OSError(f"Could not read TIFF file: {tiff_path}") from exc

    axes = interpret_tiff_axes(data.shape, imagej_metadata=imagej_meta or None)
    h, w = data.shape[-2], data.shape[-1]

    if data.ndim == 2:
        yield StackFrame(
            index=0,
            array=data,
            z_index=0,
            t_index=0,
            c_index=0,
            source_path=tiff_path,
            metadata={"axes": axes},
        )
        return

    flat = data.reshape(-1, h, w)
    z, t, c = axes.z_count, axes.t_count, axes.c_count

    for i, plane in enumerate(flat):
        if axes.source == "imagej_metadata":
            c_idx = i % c
            z_idx = (i // c) % z
            t_idx = i // (c * z)
        else:
            z_idx = i
            t_idx = None
            c_idx = None

        yield StackFrame(
            index=i,
            array=plane,
            z_index=z_idx,
            t_index=t_idx if axes.source == "imagej_metadata" else None,
            c_index=c_idx,
            source_path=tiff_path,
            metadata={"axes": axes},
        )
