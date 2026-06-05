"""Run CellProfiler pipelines via the command-line interface."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_OBJECT_MERGE_KEYS = ("Image_Number", "ObjectNumber")
_IMAGE_MERGE_KEY = "Image_Number"
_CELLPROFILER_COLUMN_ALIASES: dict[str, str] = {
    "ImageNumber": "Image_Number",
}
CellProfilerTableType = Literal[
    "image", "object", "experiment", "other", "non_standard"
]

RESULTS_MEASUREMENTS_DIR = "measurements"
RESULTS_MASKS_DIR = "masks"
RESULTS_LABELS_DIR = "labels"
RESULTS_QC_DIR = "qc"
RESULTS_LOGS_DIR = "logs"
RESULTS_RAW_DIR = "cellprofiler_raw"


@dataclass
class CellProfilerTableMetadata:
    """Classification for a loaded CellProfiler CSV export."""

    table_type: CellProfilerTableType
    legacy: bool
    mergeable: bool
    columns_found: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class CellProfilerMeasurementsResult:
    """Loaded CellProfiler CSV tables plus import metadata and warnings."""

    tables: dict[str, pd.DataFrame]
    metadata: dict[str, CellProfilerTableMetadata]
    warnings: list[str] = field(default_factory=list)


@dataclass
class CellProfilerRunResult:
    """Captured output from a headless CellProfiler subprocess run."""

    output_dir: Path
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    log_files: dict[str, Path]

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


def _resolve_existing_path(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _validate_cellprofiler_executable(cellprofiler_executable: str) -> None:
    executable_path = Path(cellprofiler_executable)
    if executable_path.is_file():
        return
    if shutil.which(cellprofiler_executable) is not None:
        return

    raise RuntimeError(
        f"CellProfiler executable not found: {cellprofiler_executable}. "
        "Install CellProfiler or pass a valid path via cellprofiler_executable."
    )


def _build_cellprofiler_command(
    cellprofiler_executable: str,
    cppipe_path: Path,
    input_dir: Path,
    output_dir: Path,
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    command = [
        cellprofiler_executable,
        "-c",
        "-r",
        "-p",
        str(cppipe_path),
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
    ]
    if extra_args:
        command.extend(extra_args)
    return command


def _write_cellprofiler_logs(
    log_dir: Path,
    *,
    command: Sequence[str],
    stdout: str,
    stderr: str,
    returncode: int,
) -> dict[str, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_files = {
        "stdout": log_dir / "cellprofiler_stdout.log",
        "stderr": log_dir / "cellprofiler_stderr.log",
        "command": log_dir / "cellprofiler_command.txt",
        "exit_code": log_dir / "cellprofiler_exit_code.txt",
    }
    log_files["stdout"].write_text(stdout, encoding="utf-8")
    log_files["stderr"].write_text(stderr, encoding="utf-8")
    log_files["command"].write_text(" ".join(command), encoding="utf-8")
    log_files["exit_code"].write_text(str(returncode), encoding="utf-8")
    return log_files


def format_cellprofiler_failure(
    *,
    returncode: int,
    stdout: str,
    stderr: str,
    log_files: dict[str, Path] | None = None,
) -> str:
    details = (stderr or "").strip() or (stdout or "").strip()
    if not details:
        details = f"exit code {returncode}"
    if log_files:
        stderr_log = log_files.get("stderr")
        if stderr_log is not None:
            details = f"{details} (see {stderr_log})"
    return details


def run_cellprofiler_pipeline(
    cppipe_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    extra_args: Sequence[str] | None = None,
    cellprofiler_executable: str = "cellprofiler",
    *,
    log_dir: str | Path | None = None,
) -> Path:
    """Run a CellProfiler pipeline in headless mode.

    Args:
        cppipe_path: Path to a CellProfiler ``.cppipe`` pipeline file.
        input_dir: Folder containing input images.
        output_dir: Folder where CellProfiler writes outputs.
        extra_args: Optional extra CLI arguments appended to the command.
        cellprofiler_executable: CellProfiler command name or full path to the
            executable (for example ``cellprofiler`` or
            ``C:\\Program Files\\CellProfiler\\CellProfiler.exe``.
        log_dir: Optional folder where stdout, stderr, and command logs are
            written before errors are raised.

    Returns:
        Resolved path to the output directory.

    Raises:
        FileNotFoundError: If the pipeline file or input directory is missing.
        RuntimeError: If CellProfiler is not installed or the command fails.
    """
    run_result = run_cellprofiler_pipeline_logged(
        cppipe_path=cppipe_path,
        input_dir=input_dir,
        output_dir=output_dir,
        extra_args=extra_args,
        cellprofiler_executable=cellprofiler_executable,
        log_dir=log_dir,
    )
    if not run_result.succeeded:
        raise RuntimeError(
            "CellProfiler command failed: "
            + format_cellprofiler_failure(
                returncode=run_result.returncode,
                stdout=run_result.stdout,
                stderr=run_result.stderr,
                log_files=run_result.log_files,
            )
        )
    return run_result.output_dir


def run_cellprofiler_pipeline_logged(
    cppipe_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    extra_args: Sequence[str] | None = None,
    cellprofiler_executable: str = "cellprofiler",
    *,
    log_dir: str | Path | None = None,
) -> CellProfilerRunResult:
    """Run CellProfiler headlessly and return captured stdout/stderr."""
    pipeline_path = _resolve_existing_path(cppipe_path, "Pipeline file")
    input_path = _resolve_existing_path(input_dir, "Input directory")
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _validate_cellprofiler_executable(cellprofiler_executable)

    command = _build_cellprofiler_command(
        cellprofiler_executable,
        pipeline_path,
        input_path,
        output_path,
        extra_args=extra_args,
    )

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Failed to launch CellProfiler: {exc}"
        ) from exc

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    log_files: dict[str, Path] = {}
    if log_dir is not None:
        log_files = _write_cellprofiler_logs(
            Path(log_dir),
            command=command,
            stdout=stdout,
            stderr=stderr,
            returncode=completed.returncode,
        )

    return CellProfilerRunResult(
        output_dir=output_path.resolve(),
        command=command,
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        log_files=log_files,
    )


def read_cellprofiler_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read a CellProfiler-exported CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Measurement table as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    csv_file = _resolve_existing_path(csv_path, "CSV file")
    if not csv_file.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    dataframe = pd.read_csv(csv_file)
    return normalize_cellprofiler_dataframe(dataframe, table_name=csv_file.stem)[0]


def normalize_cellprofiler_dataframe(
    dataframe: pd.DataFrame,
    *,
    table_name: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Map known CellProfiler column variants to canonical merge keys."""
    notes: list[str] = []
    rename_map: dict[str, str] = {}
    for column in dataframe.columns:
        canonical = _CELLPROFILER_COLUMN_ALIASES.get(column)
        if canonical is not None and canonical not in dataframe.columns:
            rename_map[column] = canonical
    if rename_map:
        dataframe = dataframe.rename(columns=rename_map)
        details = ", ".join(f"{source} -> {target}" for source, target in rename_map.items())
        notes.append(
            f"{table_name}: normalized CellProfiler columns ({details})."
        )
    return dataframe, notes


def _find_filename_column(dataframe: pd.DataFrame) -> str | None:
    if "FileName" in dataframe.columns:
        return "FileName"
    filename_columns = sorted(
        column for column in dataframe.columns if column.startswith("FileName")
    )
    if filename_columns:
        return filename_columns[0]
    return None


def validate_cellprofiler_columns(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    table_name: str = "table",
) -> None:
    """Raise ``ValueError`` when expected CellProfiler columns are missing."""
    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"{table_name} is missing required columns: {joined}"
        )


def _missing_columns(
    dataframe: pd.DataFrame,
    required_columns: Sequence[str],
) -> list[str]:
    return [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]


def classify_cellprofiler_table(
    table_name: str,
    dataframe: pd.DataFrame,
) -> CellProfilerTableMetadata:
    """Detect CellProfiler table type from filename and columns."""
    warnings: list[str] = []
    columns = set(dataframe.columns)
    has_image_number = _IMAGE_MERGE_KEY in columns
    has_object_number = _OBJECT_MERGE_KEYS[1] in columns
    has_filename = "FileName" in columns

    if table_name.endswith("_Experiment") or {"Key", "Value"}.issubset(columns):
        table_type: CellProfilerTableType = "experiment"
    elif table_name.endswith("_Image"):
        table_type = "image"
    elif "IdentifyPrimaryObjects" in table_name or "IdentifySecondaryObjects" in table_name:
        table_type = "object"
    elif has_object_number:
        table_type = "object"
    elif has_image_number and has_filename:
        table_type = "image"
    elif has_image_number:
        table_type = "image"
    else:
        table_type = "other"

    mergeable = False
    legacy = False

    if table_type == "object":
        missing = _missing_columns(dataframe, _OBJECT_MERGE_KEYS)
        if not missing:
            mergeable = True
        else:
            legacy = True
            table_type = "non_standard"
            warnings.append(
                f"{table_name}: object table missing {', '.join(missing)}; "
                "loaded as legacy/non-standard table and excluded from merge."
            )
    elif table_type in ("image", "experiment"):
        detected_type = table_type
        if has_image_number:
            mergeable = True
        else:
            legacy = True
            table_type = "non_standard"
            warnings.append(
                f"{table_name}: {detected_type} table missing Image_Number; "
                "loaded as legacy/non-standard table and excluded from merge."
            )
    elif has_image_number:
        mergeable = True
    else:
        legacy = True
        table_type = "non_standard"
        warnings.append(
            f"{table_name}: table missing Image_Number; "
            "loaded as non-standard table and excluded from merge."
        )

    return CellProfilerTableMetadata(
        table_type=table_type,
        legacy=legacy,
        mergeable=mergeable,
        columns_found=list(dataframe.columns),
        warnings=warnings,
    )


def load_cellprofiler_measurements(
    output_dir: str | Path,
    *,
    pattern: str = "*.csv",
    strict: bool = False,
) -> CellProfilerMeasurementsResult:
    """Load all CellProfiler CSV exports from an output directory.

    Args:
        output_dir: Directory containing ``ExportToSpreadsheet`` CSV files.
        pattern: Glob pattern for CSV discovery (default: all ``*.csv``).
        strict: When ``True``, raise on legacy/non-standard tables that lack
            merge keys required for their detected type. Default is lenient import.

    Returns:
        :class:`CellProfilerMeasurementsResult` with tables, metadata, and warnings.

    Raises:
        FileNotFoundError: If the directory or any CSV file is missing.
        ValueError: In strict mode when a table lacks required merge columns.
    """
    output_path = _resolve_existing_path(output_dir, "Output directory")
    if not output_path.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_path}")

    csv_files = sorted(output_path.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files matching {pattern!r} in {output_path}"
        )

    tables: dict[str, pd.DataFrame] = {}
    metadata: dict[str, CellProfilerTableMetadata] = {}
    warnings: list[str] = []

    for csv_file in csv_files:
        table_name = csv_file.stem
        raw_dataframe = pd.read_csv(csv_file)
        raw_columns = list(raw_dataframe.columns)
        column_log = f"{table_name}: columns found — {', '.join(raw_columns)}"
        logger.info(column_log)
        warnings.append(column_log)

        dataframe, normalization_notes = normalize_cellprofiler_dataframe(
            raw_dataframe,
            table_name=table_name,
        )
        warnings.extend(normalization_notes)

        table_metadata = classify_cellprofiler_table(table_name, dataframe)
        tables[table_name] = dataframe
        metadata[table_name] = table_metadata
        warnings.extend(table_metadata.warnings)

        if strict and table_metadata.legacy:
            if _OBJECT_MERGE_KEYS[1] in dataframe.columns:
                validate_cellprofiler_columns(
                    dataframe,
                    _OBJECT_MERGE_KEYS,
                    table_name=table_name,
                )
            else:
                validate_cellprofiler_columns(
                    dataframe,
                    (_IMAGE_MERGE_KEY,),
                    table_name=table_name,
                )

    return CellProfilerMeasurementsResult(
        tables=tables,
        metadata=metadata,
        warnings=warnings,
    )


def _merge_object_tables(
    object_tables: dict[str, pd.DataFrame],
    *,
    strict: bool,
    warnings: list[str],
) -> pd.DataFrame:
    names = list(object_tables.keys())
    merged = object_tables[names[0]]
    if strict:
        validate_cellprofiler_columns(
            merged,
            _OBJECT_MERGE_KEYS,
            table_name=names[0],
        )
    for name in names[1:]:
        table = object_tables[name]
        if strict:
            validate_cellprofiler_columns(
                table,
                _OBJECT_MERGE_KEYS,
                table_name=name,
            )
        merged = merged.merge(
            table,
            on=list(_OBJECT_MERGE_KEYS),
            how="outer",
            suffixes=("", f"_{name}"),
        )
    return merged


def _merge_image_tables(
    image_tables: dict[str, pd.DataFrame],
    *,
    strict: bool,
    warnings: list[str],
) -> pd.DataFrame:
    names = list(image_tables.keys())
    merged = image_tables[names[0]]
    if strict:
        validate_cellprofiler_columns(
            merged,
            (_IMAGE_MERGE_KEY,),
            table_name=names[0],
        )
    for name in names[1:]:
        table = image_tables[name]
        if strict:
            validate_cellprofiler_columns(
                table,
                (_IMAGE_MERGE_KEY,),
                table_name=name,
            )
        merged = merged.merge(
            table,
            on=_IMAGE_MERGE_KEY,
            how="outer",
            suffixes=("", f"_{name}"),
        )
    return merged


def merge_cellprofiler_tables(
    tables: dict[str, pd.DataFrame],
    *,
    strict: bool = False,
    metadata: dict[str, CellProfilerTableMetadata] | None = None,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Merge CellProfiler export tables into one DataFrame.

    Object-level tables (containing ``ObjectNumber``) are merged on
    ``Image_Number`` and ``ObjectNumber``. Image-level tables are left-joined
    on ``Image_Number``. When only image-level tables exist, they are merged
    on ``Image_Number``.

    By default (``strict=False``), tables that lack required merge keys are
    skipped with warnings instead of failing the import.

    Args:
        tables: Mapping of table name to DataFrame (as returned by
            :func:`load_cellprofiler_measurements`).
        strict: When ``True``, require merge keys on every participating table.
        metadata: Optional table metadata from
            :func:`load_cellprofiler_measurements`; inferred when omitted.

    Returns:
        Tuple of merged measurement table (or ``None`` when nothing is mergeable)
        and warning messages.

    Raises:
        ValueError: In strict mode when ``tables`` is empty or lacks merge keys.
    """
    warnings: list[str] = []
    if not tables:
        message = "No CellProfiler tables to merge"
        if strict:
            raise ValueError(message)
        warnings.append(message)
        return None, warnings

    if metadata is None:
        metadata = {
            name: classify_cellprofiler_table(name, dataframe)
            for name, dataframe in tables.items()
        }

    object_tables: dict[str, pd.DataFrame] = {}
    image_tables: dict[str, pd.DataFrame] = {}

    for name, dataframe in tables.items():
        table_metadata = metadata[name]
        if _OBJECT_MERGE_KEYS[1] in dataframe.columns:
            if table_metadata.mergeable:
                object_tables[name] = dataframe
            elif strict:
                validate_cellprofiler_columns(
                    dataframe,
                    _OBJECT_MERGE_KEYS,
                    table_name=name,
                )
            elif not table_metadata.warnings:
                missing = _missing_columns(dataframe, _OBJECT_MERGE_KEYS)
                warnings.append(
                    f"{name}: skipping object table merge — missing {', '.join(missing)}."
                )
            continue

        if table_metadata.mergeable:
            image_tables[name] = dataframe
        elif strict:
            validate_cellprofiler_columns(
                dataframe,
                (_IMAGE_MERGE_KEY,),
                table_name=name,
            )
        else:
            warnings.append(
                f"{name}: skipping image table merge — missing Image_Number."
            )

    if object_tables:
        merged = _merge_object_tables(object_tables, strict=strict, warnings=warnings)
        for name, table in image_tables.items():
            merged = merged.merge(
                table,
                on=_IMAGE_MERGE_KEY,
                how="left",
                suffixes=("", f"_{name}"),
            )
        return merged, warnings

    if image_tables:
        return _merge_image_tables(image_tables, strict=strict, warnings=warnings), warnings

    message = "No mergeable CellProfiler tables found; merged measurements unavailable."
    if strict:
        raise ValueError(message)
    warnings.append(message)
    return None, warnings


def extract_processed_image_names(
    tables: dict[str, pd.DataFrame],
) -> list[str]:
    """Return input image filenames from CellProfiler Image export tables."""
    filenames: list[str] = []
    for table_name, dataframe in tables.items():
        filename_column = _find_filename_column(dataframe)
        if filename_column is None:
            continue
        for value in dataframe[filename_column].dropna().astype(str):
            basename = Path(value).name
            if basename not in filenames:
                filenames.append(basename)
    return filenames


def summarize_cellprofiler_tables(
    tables: dict[str, pd.DataFrame],
) -> dict[str, dict[str, int]]:
    """Summarize loaded CellProfiler CSV tables for workflow reporting."""
    return {
        table_name: {
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
        }
        for table_name, dataframe in tables.items()
    }


def discover_cellprofiler_csv_files(
    output_dir: str | Path,
    *,
    pattern: str = "*.csv",
) -> list[Path]:
    """Return CSV files written by CellProfiler in an output directory."""
    output_path = Path(output_dir)
    if not output_path.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_path}")
    return sorted(path for path in output_path.glob(pattern) if path.is_file())


def discover_cellprofiler_tiff_files(
    output_dir: str | Path,
    *,
    pattern: str = "*.tif",
    exclude_dirs: Sequence[str] = (
        RESULTS_MEASUREMENTS_DIR,
        RESULTS_MASKS_DIR,
        RESULTS_LABELS_DIR,
        RESULTS_QC_DIR,
        RESULTS_LOGS_DIR,
    ),
) -> list[Path]:
    """Return TIFF files written by CellProfiler, excluding organized results."""
    output_path = Path(output_dir)
    if not output_path.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_path}")

    excluded = {name.lower() for name in exclude_dirs}
    discovered: list[Path] = []

    for candidate in sorted(output_path.rglob(pattern)):
        if not candidate.is_file():
            continue
        if any(part.lower() in excluded for part in candidate.relative_to(output_path).parts):
            continue
        discovered.append(candidate)

    if pattern == "*.tif":
        for candidate in sorted(output_path.rglob("*.tiff")):
            if not candidate.is_file():
                continue
            if any(
                part.lower() in excluded
                for part in candidate.relative_to(output_path).parts
            ):
                continue
            discovered.append(candidate)

    return discovered


def copy_cellprofiler_measurements(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    pattern: str = "*.csv",
) -> list[Path]:
    """Copy CellProfiler CSV exports into a measurements results folder."""
    source_path = Path(source_dir)
    destination_path = Path(destination_dir)
    destination_path.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for csv_file in discover_cellprofiler_csv_files(source_path, pattern=pattern):
        destination = destination_path / csv_file.name
        destination.write_bytes(csv_file.read_bytes())
        copied.append(destination.resolve())
    return copied
