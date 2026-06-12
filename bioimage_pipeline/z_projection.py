"""Z-max projection utilities matching Fiji ``Stacking+Drectly.ijm`` behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from bioimage_pipeline.io import save_tiff


PYTHON_OIR_MISSING_DEPS_MESSAGE = (
    "Python OIR projection requires aicsimageio/bfio. "
    "Install them or switch OIR projection engine to Fiji."
)


class OirPythonReadError(RuntimeError):
    """Raised when the optional Python .oir reader cannot load a file."""


def python_oir_dependencies_available() -> bool:
    """Return ``True`` when the optional Python ``.oir`` reader can be imported."""
    try:
        import aicsimageio  # noqa: F401
    except ImportError:
        return False
    return True


def oir_output_filename(source_name: str) -> str:
    """Return the projected TIFF filename for one ``.oir`` input basename."""
    if source_name.lower().endswith(".oir"):
        return f"{source_name[:-4]}.tif"
    return f"{Path(source_name).stem}.tif"


def oir_output_path(oir_path: str | Path, output_dir: str | Path) -> Path:
    """Return the projected TIFF path for one input ``.oir`` file."""
    source = Path(oir_path)
    return Path(output_dir) / oir_output_filename(source.name)


def iter_oir_files(root: str | Path) -> Iterator[Path]:
    """Recursively yield ``.oir`` files under *root*, mirroring the Fiji macro."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root_path}")

    for entry in sorted(root_path.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".oir":
            yield entry
        elif entry.is_dir():
            yield from iter_oir_files(entry)


def format_oir_read_dependency_error(exc: Exception) -> str:
    """Build a user-facing message for failed Python ``.oir`` reads."""
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()

    if isinstance(exc, ImportError) or (
        "java backend is not available" in lowered
        or "bfio" in lowered
        or "bioformats" in lowered
        or "aicsimageio" in lowered
        or "no module named 'aicsimageio'" in lowered
    ):
        return PYTHON_OIR_MISSING_DEPS_MESSAGE

    return (
        "Python .oir reading failed. This workflow is intended to run through "
        f"Fiji/ImageJ on .oir files.\nDetails: {message}\n"
        "Recommended fix: use --engine fiji --fiji /path/to/ImageJ-win64.exe"
    )


def zmax_intensity(stack: np.ndarray, *, axis: int = 0) -> np.ndarray:
    """Max-intensity projection along *axis* (Fiji ``Z Project`` equivalent)."""
    array = np.asarray(stack)
    if array.ndim == 2:
        return array
    if array.ndim < 3:
        raise ValueError(f"Expected a 2D or Z-stack array, got shape {array.shape}")
    return np.max(array, axis=axis)


def load_oir_stack(path: str | Path) -> np.ndarray:
    """Load an Olympus ``.oir`` file as a Z-stack array using Bio-Formats."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"OIR file not found: {image_path}")

    try:
        from aicsimageio import AICSImage
    except ImportError as exc:
        raise OirPythonReadError(format_oir_read_dependency_error(exc)) from exc

    try:
        image = AICSImage(image_path)
        data = image.get_image_data()
    except Exception as exc:
        raise OirPythonReadError(format_oir_read_dependency_error(exc)) from exc

    squeezed = np.squeeze(data)
    if squeezed.ndim == 2:
        return squeezed
    if squeezed.ndim == 3:
        return squeezed
    raise OirPythonReadError(
        format_oir_read_dependency_error(
            ValueError(
                f"Unsupported OIR dimensionality after squeeze: {squeezed.shape}. "
                "Use engine='fiji' for complex hyperstacks."
            )
        )
    )


def process_oir_file_python(
    oir_path: str | Path,
    output_dir: str | Path,
    *,
    z_axis: int = 0,
) -> Path:
    """Z-max project one ``.oir`` file and save a TIFF using macro naming rules."""
    output_path, _record = process_oir_file_python_timed(
        oir_path,
        output_dir,
        z_axis=z_axis,
    )
    return output_path


def process_oir_file_python_timed(
    oir_path: str | Path,
    output_dir: str | Path,
    *,
    z_axis: int = 0,
    audit_logs_dir: str | Path | None = None,
) -> tuple[Path, "PrepareInputFileRecord"]:
    """Z-max project one ``.oir`` file and return output path plus timing record."""
    import time

    from bioimage_pipeline.prepare_input_profile import (
        PrepareInputFileRecord,
        detect_file_type,
    )

    total_started = time.perf_counter()
    source = Path(oir_path).resolve()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = oir_output_path(source, out_dir)
    input_bytes = source.stat().st_size if source.is_file() else 0
    output_existed = output_path.is_file()

    read_started = time.perf_counter()
    stack = load_oir_stack(source)
    read_seconds = time.perf_counter() - read_started

    conversion_started = time.perf_counter()
    if stack.ndim == 3:
        projected = zmax_intensity(stack, axis=z_axis)
    else:
        projected = stack
    conversion_seconds = time.perf_counter() - conversion_started

    write_started = time.perf_counter()
    save_tiff(output_path, projected, audit_logs_dir=audit_logs_dir)
    write_seconds = time.perf_counter() - write_started

    resolved_output = output_path.resolve()
    output_bytes = resolved_output.stat().st_size if resolved_output.is_file() else None
    record = PrepareInputFileRecord(
        input_path=str(source),
        detected_type=detect_file_type(source),
        output_path=str(resolved_output),
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        read_seconds=read_seconds,
        conversion_seconds=conversion_seconds,
        write_seconds=write_seconds,
        total_seconds=time.perf_counter() - total_started,
        output_existed_before_run=output_existed,
    )
    if output_existed:
        record.notes.append(
            "Projected output already existed before this run but was overwritten."
        )
    return resolved_output, record
