"""Run CellProfiler pipelines via the command-line interface."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

_OBJECT_MERGE_KEYS = ("Image_Number", "ObjectNumber")
_IMAGE_MERGE_KEY = "Image_Number"

RESULTS_MEASUREMENTS_DIR = "measurements"
RESULTS_MASKS_DIR = "masks"
RESULTS_LABELS_DIR = "labels"
RESULTS_QC_DIR = "qc"
RESULTS_LOGS_DIR = "logs"
RESULTS_RAW_DIR = "cellprofiler_raw"


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

    return pd.read_csv(csv_file)


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


def load_cellprofiler_measurements(
    output_dir: str | Path,
    *,
    pattern: str = "*.csv",
) -> dict[str, pd.DataFrame]:
    """Load all CellProfiler CSV exports from an output directory.

    Args:
        output_dir: Directory containing ``ExportToSpreadsheet`` CSV files.
        pattern: Glob pattern for CSV discovery (default: all ``*.csv``).

    Returns:
        Mapping of file stem (e.g. ``MyExpt_Image``) to DataFrame.

    Raises:
        FileNotFoundError: If the directory or any CSV file is missing.
    """
    output_path = _resolve_existing_path(output_dir, "Output directory")
    if not output_path.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_path}")

    csv_files = sorted(output_path.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files matching {pattern!r} in {output_path}"
        )

    return {
        csv_file.stem: read_cellprofiler_csv(csv_file) for csv_file in csv_files
    }


def merge_cellprofiler_tables(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge CellProfiler export tables into one DataFrame.

    Object-level tables (containing ``ObjectNumber``) are merged on
    ``Image_Number`` and ``ObjectNumber``. Image-level tables are left-joined
    on ``Image_Number``. When only image-level tables exist, they are merged
    on ``Image_Number``.

    Args:
        tables: Mapping of table name to DataFrame (as returned by
            :func:`load_cellprofiler_measurements`).

    Returns:
        Combined measurement table.

    Raises:
        ValueError: If ``tables`` is empty or lacks merge keys.
    """
    if not tables:
        raise ValueError("No CellProfiler tables to merge")

    object_tables = {
        name: dataframe
        for name, dataframe in tables.items()
        if _OBJECT_MERGE_KEYS[1] in dataframe.columns
    }
    image_tables = {
        name: dataframe
        for name, dataframe in tables.items()
        if _OBJECT_MERGE_KEYS[1] not in dataframe.columns
    }

    if object_tables:
        names = list(object_tables.keys())
        merged = object_tables[names[0]]
        validate_cellprofiler_columns(
            merged,
            _OBJECT_MERGE_KEYS,
            table_name=names[0],
        )
        for name in names[1:]:
            table = object_tables[name]
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
        for name, table in image_tables.items():
            if _IMAGE_MERGE_KEY not in table.columns:
                raise ValueError(
                    f"{name} is missing required column: {_IMAGE_MERGE_KEY}"
                )
            merged = merged.merge(
                table,
                on=_IMAGE_MERGE_KEY,
                how="left",
                suffixes=("", f"_{name}"),
            )
        return merged

    if not image_tables:
        raise ValueError("No CellProfiler tables to merge")

    names = list(image_tables.keys())
    merged = image_tables[names[0]]
    validate_cellprofiler_columns(
        merged,
        (_IMAGE_MERGE_KEY,),
        table_name=names[0],
    )
    for name in names[1:]:
        table = image_tables[name]
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


def extract_processed_image_names(
    tables: dict[str, pd.DataFrame],
) -> list[str]:
    """Return input image filenames from CellProfiler Image export tables."""
    filenames: list[str] = []
    for table_name, dataframe in tables.items():
        if "FileName" not in dataframe.columns:
            continue
        for value in dataframe["FileName"].dropna().astype(str):
            if value not in filenames:
                filenames.append(value)
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
