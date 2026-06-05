"""Helpers for the Phase 15.0 Streamlit workflow test UI."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd


def read_text_tail(path: str | Path, *, max_chars: int = 80_000) -> str:
    """Return the trailing text from a log file."""
    file_path = Path(path)
    if not file_path.is_file():
        return f"(missing file: {file_path})"
    text = file_path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return f"... (truncated, showing last {max_chars} characters)\n" + text[-max_chars:]


def list_qc_pngs(qc_dir: str | Path) -> list[Path]:
    """List QC overlay PNG files in stable order."""
    directory = Path(qc_dir)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.png"))


def load_measurements_for_display(measurements_dir: str | Path) -> pd.DataFrame | None:
    """Load merged measurements when available, otherwise the first CSV table."""
    directory = Path(measurements_dir)
    if not directory.is_dir():
        return None

    merged = directory / "merged_measurements.csv"
    if merged.is_file():
        return pd.read_csv(merged)

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        return None
    return pd.read_csv(csv_files[0])


def save_uploaded_cppipe(uploaded: BinaryIO, filename: str, dest_dir: str | Path) -> Path:
    """Persist an uploaded ``.cppipe`` file for workflow execution."""
    directory = Path(dest_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / Path(filename).name
    destination.write_bytes(uploaded.read())
    return destination.resolve()


def validate_workflow_inputs(
    input_dir: str | Path,
    output_dir: str | Path,
    cppipe_path: str | Path | None,
) -> list[str]:
    """Return human-readable validation errors for workflow form inputs."""
    errors: list[str] = []

    input_path = Path(input_dir)
    if not str(input_dir).strip():
        errors.append("Input image folder is required.")
    elif not input_path.is_dir():
        errors.append(f"Input image folder does not exist: {input_path}")

    if not str(output_dir).strip():
        errors.append("Output folder is required.")

    if cppipe_path is None or not str(cppipe_path).strip():
        errors.append("A CellProfiler pipeline (.cppipe) is required.")
    else:
        pipeline_path = Path(cppipe_path)
        if not pipeline_path.is_file():
            errors.append(f"Pipeline file does not exist: {pipeline_path}")
        elif pipeline_path.suffix.lower() != ".cppipe":
            errors.append(f"Pipeline file must end with .cppipe: {pipeline_path}")

    return errors
