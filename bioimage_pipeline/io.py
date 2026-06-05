"""TIFF image input/output helpers."""

from pathlib import Path
from typing import Any

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
    return image_path.resolve()
