"""Run CellProfiler pipelines via the command-line interface."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import pandas as pd


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


def run_cellprofiler_pipeline(
    cppipe_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    extra_args: Sequence[str] | None = None,
    cellprofiler_executable: str = "cellprofiler",
) -> Path:
    """Run a CellProfiler pipeline in headless mode.

    Args:
        cppipe_path: Path to a CellProfiler ``.cppipe`` pipeline file.
        input_dir: Folder containing input images.
        output_dir: Folder where CellProfiler writes outputs.
        extra_args: Optional extra CLI arguments appended to the command.
        cellprofiler_executable: CellProfiler command name or full path to the
            executable (for example ``cellprofiler`` or
            ``C:\\Program Files\\CellProfiler\\CellProfiler.exe``).

    Returns:
        Resolved path to the output directory.

    Raises:
        FileNotFoundError: If the pipeline file or input directory is missing.
        RuntimeError: If CellProfiler is not installed or the command fails.
    """
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

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        details = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"CellProfiler command failed: {details}")

    return output_path.resolve()


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
