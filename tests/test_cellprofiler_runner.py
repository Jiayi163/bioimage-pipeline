"""Tests for CellProfiler CLI integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bioimage_pipeline.cellprofiler_runner import (
    _build_cellprofiler_command,
    discover_cellprofiler_csv_files,
    load_cellprofiler_measurements,
    merge_cellprofiler_tables,
    read_cellprofiler_csv,
    run_cellprofiler_pipeline,
    run_cellprofiler_pipeline_logged,
    validate_cellprofiler_columns,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "cellprofiler"


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
        "cellprofiler",
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

    with pytest.raises(RuntimeError, match="executable not found"):
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


@patch("bioimage_pipeline.cellprofiler_runner.subprocess.run")
def test_run_cellprofiler_pipeline_uses_custom_executable(
    mock_run: MagicMock,
    tmp_path: Path,
) -> None:
    cppipe = tmp_path / "pipeline.cppipe"
    cppipe.write_text("pipeline", encoding="utf-8")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    executable = tmp_path / "CellProfiler.exe"
    executable.write_text("", encoding="utf-8")

    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    run_cellprofiler_pipeline(
        cppipe,
        input_dir,
        output_dir,
        cellprofiler_executable=str(executable),
    )

    command = mock_run.call_args[0][0]
    assert command[0] == str(executable)
    assert "-c" in command
    assert "-r" in command
    assert "-p" in command
    assert "-i" in command
    assert "-o" in command


def test_run_cellprofiler_pipeline_custom_executable_missing_raises(
    tmp_path: Path,
) -> None:
    cppipe = tmp_path / "pipeline.cppipe"
    cppipe.write_text("pipeline", encoding="utf-8")
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(RuntimeError, match="executable not found"):
        run_cellprofiler_pipeline(
            cppipe,
            input_dir,
            tmp_path / "output",
            cellprofiler_executable=tmp_path / "missing.exe",
        )


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


def test_validate_cellprofiler_columns_passes_when_present() -> None:
    dataframe = pd.DataFrame({"Image_Number": [1], "ObjectNumber": [1]})
    validate_cellprofiler_columns(
        dataframe,
        ["Image_Number", "ObjectNumber"],
        table_name="objects",
    )


def test_validate_cellprofiler_columns_raises_when_missing() -> None:
    dataframe = pd.DataFrame({"Image_Number": [1]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_cellprofiler_columns(
            dataframe,
            ["Image_Number", "ObjectNumber"],
            table_name="objects",
        )


def test_load_cellprofiler_measurements_reads_fixture_csvs() -> None:
    tables = load_cellprofiler_measurements(FIXTURES_DIR)

    assert set(tables) == {
        "MyExpt_Image",
        "MyExpt_Experiment",
        "MyExpt_IdentifyPrimaryObjects",
    }
    assert tables["MyExpt_Image"].loc[0, "FileName"] == "testimage.tif"
    assert len(tables["MyExpt_IdentifyPrimaryObjects"]) == 2


def test_load_cellprofiler_measurements_empty_dir_raises(tmp_path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No CSV files"):
        load_cellprofiler_measurements(empty_dir)


def test_load_cellprofiler_measurements_missing_dir_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Output directory"):
        load_cellprofiler_measurements(tmp_path / "missing")


def test_merge_cellprofiler_tables_combines_object_and_image_tables() -> None:
    tables = load_cellprofiler_measurements(FIXTURES_DIR)
    merged = merge_cellprofiler_tables(tables)

    assert len(merged) == 2
    assert "AreaShape_Area" in merged.columns
    assert "FileName" in merged.columns
    assert "Plate_Name" in merged.columns
    assert merged.loc[0, "AreaShape_Area"] == 120


def test_merge_cellprofiler_tables_empty_dict_raises() -> None:
    with pytest.raises(ValueError, match="No CellProfiler tables"):
        merge_cellprofiler_tables({})


@patch("bioimage_pipeline.cellprofiler_runner.shutil.which", return_value="cellprofiler")
@patch("bioimage_pipeline.cellprofiler_runner.subprocess.run")
def test_run_cellprofiler_pipeline_writes_logs_on_failure(
    mock_run: MagicMock,
    mock_which: MagicMock,
    tmp_path: Path,
) -> None:
    cppipe = tmp_path / "pipeline.cppipe"
    cppipe.write_text("pipeline", encoding="utf-8")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    log_dir = tmp_path / "logs"

    mock_run.return_value = MagicMock(
        returncode=1,
        stdout="stdout details",
        stderr="stderr details",
    )

    with pytest.raises(RuntimeError, match="CellProfiler command failed"):
        run_cellprofiler_pipeline(
            cppipe,
            input_dir,
            output_dir,
            log_dir=log_dir,
        )

    assert (log_dir / "cellprofiler_stdout.log").read_text(encoding="utf-8") == (
        "stdout details"
    )
    assert (log_dir / "cellprofiler_stderr.log").read_text(encoding="utf-8") == (
        "stderr details"
    )
    assert "cellprofiler" in (log_dir / "cellprofiler_command.txt").read_text(
        encoding="utf-8"
    )


@patch("bioimage_pipeline.cellprofiler_runner.shutil.which", return_value="cellprofiler")
@patch("bioimage_pipeline.cellprofiler_runner.subprocess.run")
def test_run_cellprofiler_pipeline_logged_returns_run_result(
    mock_run: MagicMock,
    mock_which: MagicMock,
    tmp_path: Path,
) -> None:
    cppipe = tmp_path / "pipeline.cppipe"
    cppipe.write_text("pipeline", encoding="utf-8")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    log_dir = tmp_path / "logs"

    mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

    result = run_cellprofiler_pipeline_logged(
        cppipe,
        input_dir,
        output_dir,
        log_dir=log_dir,
    )

    assert result.succeeded
    assert result.stdout == "ok"
    assert result.log_files["stdout"].exists()


def test_discover_cellprofiler_csv_files_finds_exports(tmp_path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "MyExpt_Image.csv").write_text("Image_Number\n1\n", encoding="utf-8")

    discovered = discover_cellprofiler_csv_files(output_dir)

    assert len(discovered) == 1
    assert discovered[0].name == "MyExpt_Image.csv"


def test_merge_cellprofiler_tables_image_only() -> None:
    tables = {
        "MyExpt_Image": pd.DataFrame(
            {"Image_Number": [1], "FileName": ["a.tif"]}
        ),
        "MyExpt_Experiment": pd.DataFrame(
            {"Image_Number": [1], "Plate_Name": ["Plate1"]}
        ),
    }
    merged = merge_cellprofiler_tables(tables)

    assert list(merged.columns) == ["Image_Number", "FileName", "Plate_Name"]
