"""Z-max projection utilities matching Fiji ``Stacking+Drectly.ijm`` behavior."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterator, Literal

import numpy as np

logger = logging.getLogger(__name__)

ZProjectionMethod = Literal[
    "average",
    "max",
    "min",
    "sum",
    "standard",
    "median",
]

Z_PROJECTION_METHODS: tuple[ZProjectionMethod, ...] = (
    "average",
    "max",
    "min",
    "sum",
    "standard",
    "median",
)

Z_PROJECTION_FIJI_LABELS: dict[ZProjectionMethod, str] = {
    "average": "Average Intensity",
    "max": "Max Intensity",
    "min": "Min Intensity",
    "sum": "Sum Slices",
    "standard": "Standard Deviation",
    "median": "Median",
}

Z_PROJECTION_GUI_LABELS: dict[ZProjectionMethod, str] = {
    "average": "Average Intensity",
    "max": "Max Intensity",
    "min": "Min Intensity",
    "sum": "Sum Slices",
    "standard": "Standard Deviation",
    "median": "Median",
}

DEFAULT_Z_PROJECTION_METHOD: ZProjectionMethod = "max"

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


def normalize_projection_method(value: str | ZProjectionMethod | None) -> ZProjectionMethod:
    """Normalize a projection method slug or return the default."""
    if value is None or not str(value).strip():
        return DEFAULT_Z_PROJECTION_METHOD
    normalized = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "avg": "average",
        "mean": "average",
        "maximum": "max",
        "max_intensity": "max",
        "minimum": "min",
        "min_intensity": "min",
        "std": "standard",
        "standard_deviation": "standard",
        "stdev": "standard",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in Z_PROJECTION_FIJI_LABELS:
        allowed = ", ".join(Z_PROJECTION_METHODS)
        raise ValueError(
            f"Unsupported Z projection method: {value!r}. Expected one of: {allowed}"
        )
    return normalized  # type: ignore[return-value]


def fiji_projection_label(method: str | ZProjectionMethod | None) -> str:
    """Return the ImageJ ``Z Project...`` option string for *method*."""
    slug = normalize_projection_method(method)
    return Z_PROJECTION_FIJI_LABELS[slug]


def _projection_reducer(method: ZProjectionMethod) -> Callable[..., np.ndarray]:
    reducers: dict[ZProjectionMethod, Callable[..., np.ndarray]] = {
        "average": np.mean,
        "max": np.max,
        "min": np.min,
        "sum": np.sum,
        "standard": np.std,
        "median": np.median,
    }
    return reducers[method]


def project_stack(
    stack: np.ndarray,
    method: str | ZProjectionMethod | None = DEFAULT_Z_PROJECTION_METHOD,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Project a Z-stack along *axis* using the Fiji ``Z Project`` method."""
    array = np.asarray(stack)
    if array.ndim == 2:
        return array
    if array.ndim < 3:
        raise ValueError(f"Expected a 2D or Z-stack array, got shape {array.shape}")
    slug = normalize_projection_method(method)
    return _projection_reducer(slug)(array, axis=axis)


def zmax_intensity(stack: np.ndarray, *, axis: int = 0) -> np.ndarray:
    """Max-intensity projection along *axis* (Fiji ``Z Project`` equivalent)."""
    return project_stack(stack, "max", axis=axis)


def load_oir_stack(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an Olympus ``.oir`` file as a Z-stack array using Bio-Formats.

    Returns:
        Tuple of ``(stack, z_axis)`` where *z_axis* is the NumPy axis index
        corresponding to Fiji/ImageJ ``nSlices`` (Z), not C/T/Y/X.
    """
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"OIR file not found: {image_path}")

    try:
        from aicsimageio import AICSImage
    except ImportError as exc:
        raise OirPythonReadError(format_oir_read_dependency_error(exc)) from exc

    try:
        image = AICSImage(image_path)
        dims = image.dims
        if "Z" in dims:
            stack = np.squeeze(image.get_image_data("ZYX"))
            z_axis = 0
        elif "T" in dims and "Z" not in dims:
            stack = np.squeeze(image.get_image_data("TYX"))
            z_axis = 0
        else:
            stack = np.squeeze(image.get_image_data())
            z_axis = 0
    except Exception as exc:
        raise OirPythonReadError(format_oir_read_dependency_error(exc)) from exc

    if stack.ndim == 2:
        return stack, z_axis
    if stack.ndim == 3:
        logger.info(
            "Python OIR import %s: dims=%s stack_shape=%s z_axis=%d",
            image_path.name,
            dims,
            stack.shape,
            z_axis,
        )
        return stack, z_axis
    raise OirPythonReadError(
        format_oir_read_dependency_error(
            ValueError(
                f"Unsupported OIR dimensionality after squeeze: {stack.shape}. "
                "Use engine='fiji' for complex hyperstacks."
            )
        )
    )


def process_oir_file_python(
    oir_path: str | Path,
    output_dir: str | Path,
    *,
    projection_method: ZProjectionMethod | str = DEFAULT_Z_PROJECTION_METHOD,
) -> Path:
    """Z-max project one ``.oir`` file and save a TIFF using macro naming rules."""
    output_path, _record = process_oir_file_python_timed(
        oir_path,
        output_dir,
        projection_method=projection_method,
    )
    return output_path


def process_oir_file_python_timed(
    oir_path: str | Path,
    output_dir: str | Path,
    *,
    projection_method: ZProjectionMethod | str = DEFAULT_Z_PROJECTION_METHOD,
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
    stack, project_axis = load_oir_stack(source)
    read_seconds = time.perf_counter() - read_started

    method = normalize_projection_method(projection_method)
    fiji_arg = fiji_projection_label(method)
    conversion_started = time.perf_counter()
    if stack.ndim == 3:
        projected = project_stack(stack, method, axis=project_axis)
    else:
        projected = stack
    conversion_seconds = time.perf_counter() - conversion_started

    write_started = time.perf_counter()
    save_tiff(output_path, projected, audit_logs_dir=audit_logs_dir)
    write_seconds = time.perf_counter() - write_started

    resolved_output = output_path.resolve()
    output_bytes = resolved_output.stat().st_size if resolved_output.is_file() else None
    from bioimage_pipeline.projection_postprocess import log_projection_output_diagnostics

    log_projection_output_diagnostics(
        engine="python",
        projection_method=method,
        input_path=source,
        output_path=resolved_output,
    )
    logger.info(
        "Python OIR projection: method=%s fiji_arg=%s z_axis=%d "
        "input=%s output=%s projected_shape=%s dtype=%s min=%.6g max=%.6g",
        method,
        fiji_arg,
        project_axis,
        source,
        resolved_output,
        projected.shape,
        projected.dtype,
        float(np.min(projected)) if projected.size else 0.0,
        float(np.max(projected)) if projected.size else 0.0,
    )
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
