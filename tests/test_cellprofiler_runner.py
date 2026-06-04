"""Tests for CellProfiler CLI integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bioimage_pipeline.cellprofiler_runner import (
    _build_cellprofiler_command,
    read_cellprofiler_csv,
    run_cellprofiler_pipeline,
)


def test_missing_cppipe_path_raises_file_not_found(tmp_path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Pipeline file"):
        run_cellprofiler_pipeline(
            tmp_path / "missing.cppipe",
            input_dir,
            tmp_path / "output",
        )


def test_missing_input_dir_raises_file_not_found(tmp_path) -> None:
    cppipe = tmp_path / "pipeline.cppipe"
    cppipe.write_text("pipeline", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Input directory"):
        run_cellprofiler_pipeline(
            cppipe,
            tmp_path / "missing_input",
            tmp_path / "output",
        )


def test_build_cellprofiler_command_includes_required_flags(tmp_path) -> None:
    command = _build_cellprofiler_command(
        tmp_path / "pipeline.cppipe",
        tmp_path / "input",
        tmp_path / "output",
    )

    assert command == [
        "cellprofiler",
        "-c",
        "-r",
        "-p",
        str(tmp_path / "pipeline.cppipe"),
        "-i",
        str(tmp_path / "input"),
        "-o",
        str(tmp_path / "output"),
    ]


@patch("bioimage_pipeline.cellprofiler_runner.shutil.which", return_value="cellprofiler")
@patch("bioimage_pipeline.cellprofiler_runner.subprocess.run")
def test_run_cellprofiler_pipeline_calls_subprocess(
    mock_run: MagicMock,
    mock_which: MagicMock,
    tmp_path: Path,
) -> None:
    cppipe = tmp_path / "pipeline.cppipe"
    cppipe.write_text("pipeline", encoding="utf-8")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    result = run_cellprofiler_pipeline(cppipe, input_dir, output_dir)

    assert result == output_dir.resolve()
    mock_which.assert_called_once_with("cellprofiler")
    mock_run.assert_called_once()
    command = mock_run.call_args[0][0]
    assert command[0] == "cellprofiler"
    assert "-c" in command
    assert "-r" in command
    assert "-p" in command
    assert "-i" in command
    assert "-o" in command
    assert str(cppipe) in command
    assert str(input_dir) in command
    assert str(output_dir) in command


@patch("bioimage_pipeline.cellprofiler_runner.shutil.which", return_value=None)
def test_run_cellprofiler_pipeline_raises_when_not_installed(
    mock_which: MagicMock,
    tmp_path: Path,
) -> None:
    cppipe = tmp_path / "pipeline.cppipe"
    cppipe.write_text("pipeline", encoding="utf-8")
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(RuntimeError, match="not installed"):
        run_cellprofiler_pipeline(cppipe, input_dir, tmp_path / "output")


@patch("bioimage_pipeline.cellprofiler_runner.shutil.which", return_value="cellprofiler")
@patch("bioimage_pipeline.cellprofiler_runner.subprocess.run")
def test_run_cellprofiler_pipeline_raises_on_nonzero_exit(
    mock_run: MagicMock,
    mock_which: MagicMock,
    tmp_path: Path,
) -> None:
    cppipe = tmp_path / "pipeline.cppipe"
    cppipe.write_text("pipeline", encoding="utf-8")
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="pipeline error")

    with pytest.raises(RuntimeError, match="CellProfiler command failed"):
        run_cellprofiler_pipeline(cppipe, input_dir, tmp_path / "output")


def test_read_cellprofiler_csv_returns_dataframe(tmp_path) -> None:
    csv_path = tmp_path / "measurements.csv"
    csv_path.write_text("Image_Number,Area\n1,100\n", encoding="utf-8")

    dataframe = read_cellprofiler_csv(csv_path)

    assert isinstance(dataframe, pd.DataFrame)
    assert list(dataframe.columns) == ["Image_Number", "Area"]
    assert dataframe.loc[0, "Area"] == 100


def test_read_cellprofiler_csv_missing_file_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="CSV file"):
        read_cellprofiler_csv(tmp_path / "missing.csv")
