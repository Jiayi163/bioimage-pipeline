"""Tests for CellProfiler CLI integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bioimage_pipeline.cellprofiler_runner import (
    _build_cellprofiler_command,
    cellprofiler_run_succeeded,
    classify_cellprofiler_table,
    discover_cellprofiler_csv_files,
    extract_cellprofiler_errors,
    inspect_cellprofiler_logs,
    load_cellprofiler_measurements,
    load_cellprofiler_measurements_lenient,
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


@patch("bioimage_pipeline.cellprofiler_runner.find_cellprofiler_executable", return_value=None)
def test_run_cellprofiler_pipeline_raises_when_not_installed(
    _mock_find: MagicMock,
    tmp_path: Path,
) -> None:
    cppipe = tmp_path / "pipeline.cppipe"
    cppipe.write_text("pipeline", encoding="utf-8")
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(RuntimeError, match="CellProfiler not found"):
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

    with pytest.raises(RuntimeError, match="CellProfiler not found"):
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
    load_result = load_cellprofiler_measurements(FIXTURES_DIR)

    assert set(load_result.tables) == {
        "MyExpt_Image",
        "MyExpt_Experiment",
        "MyExpt_IdentifyPrimaryObjects",
    }
    assert load_result.tables["MyExpt_Image"].loc[0, "FileName"] == "testimage.tif"
    assert len(load_result.tables["MyExpt_IdentifyPrimaryObjects"]) == 2
    assert not load_result.warnings


def test_load_cellprofiler_measurements_reads_cp1252_csv(tmp_path: Path) -> None:
    output_dir = tmp_path / "measurements"
    output_dir.mkdir()
    csv_path = output_dir / "MyExpt_Image.csv"
    csv_path.write_bytes(
        b"Image_Number,FileName,Count_Cells\r\n1,test\xcfimage.tif,2\r\n"
    )

    load_result = load_cellprofiler_measurements(output_dir)

    assert "MyExpt_Image" in load_result.tables
    assert load_result.tables["MyExpt_Image"].loc[0, "FileName"] == "testÏimage.tif"


def test_load_cellprofiler_measurements_empty_dir_raises(tmp_path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No CSV files"):
        load_cellprofiler_measurements(empty_dir)


def test_load_cellprofiler_measurements_lenient_empty_dir_returns_warning(
    tmp_path,
) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    load_result = load_cellprofiler_measurements_lenient(empty_dir)

    assert load_result.tables == {}
    assert load_result.metadata == {}
    assert any("No CSV files" in warning for warning in load_result.warnings)


def test_extract_cellprofiler_errors_detects_module_failures() -> None:
    stderr = (
        "Running pipeline\n"
        "Encountered error in module IdentifyPrimaryObjects\n"
        "Traceback (most recent call last):\n"
        "  File \"cp\", line 1, in <module>\n"
        "KeyError: 'DNA'\n"
    )
    errors = extract_cellprofiler_errors("", stderr)

    assert errors == [
        "Encountered error in module IdentifyPrimaryObjects",
        "KeyError: 'DNA'",
    ]


def test_extract_cellprofiler_errors_surfaces_saveimages_assertion() -> None:
    stderr = (
        "Times reported are CPU seconds\n"
        "Error detected during run of module SaveImages\n"
        "Traceback (most recent call last):\n"
        "  File \"cellprofiler/modules/saveimages.py\", line 42, in run\n"
        "    self.save_image(...)\n"
        "AssertionError: Feature FileName_OriginalImage does not exist\n"
    )
    errors = extract_cellprofiler_errors("", stderr)

    assert errors == [
        "Error detected during run of module SaveImages",
        "AssertionError: Feature FileName_OriginalImage does not exist",
    ]
    assert not any("Traceback" in line for line in errors)


def test_inspect_cellprofiler_logs_reads_on_disk_files(tmp_path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "cellprofiler_stdout.log").write_text(
        "Times reported are CPU seconds\n",
        encoding="utf-8",
    )
    (log_dir / "cellprofiler_stderr.log").write_text(
        "Error in module ExportToSpreadsheet: missing input image\n",
        encoding="utf-8",
    )

    errors = inspect_cellprofiler_logs(log_dir=log_dir)

    assert errors == ["Error in module ExportToSpreadsheet: missing input image"]


def test_load_cellprofiler_measurements_missing_dir_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Output directory"):
        load_cellprofiler_measurements(tmp_path / "missing")


def test_merge_cellprofiler_tables_combines_object_and_image_tables() -> None:
    load_result = load_cellprofiler_measurements(FIXTURES_DIR)
    merged, warnings = merge_cellprofiler_tables(
        load_result.tables,
        metadata=load_result.metadata,
    )

    assert merged is not None
    assert len(merged) == 2
    assert "AreaShape_Area" in merged.columns
    assert "FileName" in merged.columns
    assert "Plate_Name" in merged.columns
    assert merged.loc[0, "AreaShape_Area"] == 120
    assert not warnings


def test_merge_cellprofiler_tables_empty_dict_returns_none_with_warning() -> None:
    merged, warnings = merge_cellprofiler_tables({})
    assert merged is None
    assert warnings == ["No CellProfiler tables to merge"]


def test_merge_cellprofiler_tables_empty_dict_raises_in_strict_mode() -> None:
    with pytest.raises(ValueError, match="No CellProfiler tables"):
        merge_cellprofiler_tables({}, strict=True)


def test_load_cellprofiler_measurements_legacy_object_table_without_image_number(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "MyExpt_Image.csv").write_text(
        "Image_Number,FileName\n1,sample.tif\n",
        encoding="utf-8",
    )
    (output_dir / "MyExpt_IdentifyPrimaryObjects.csv").write_text(
        "ObjectNumber,AreaShape_Area\n1,120\n2,95\n",
        encoding="utf-8",
    )

    load_result = load_cellprofiler_measurements(output_dir)

    assert "MyExpt_IdentifyPrimaryObjects" in load_result.tables
    assert len(load_result.tables["MyExpt_IdentifyPrimaryObjects"]) == 2
    assert load_result.metadata["MyExpt_IdentifyPrimaryObjects"].legacy is True
    assert load_result.metadata["MyExpt_IdentifyPrimaryObjects"].mergeable is False
    assert any("Image_Number" in warning for warning in load_result.warnings)


def test_merge_cellprofiler_tables_skips_legacy_object_table_by_default(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "MyExpt_Image.csv").write_text(
        "Image_Number,FileName\n1,sample.tif\n",
        encoding="utf-8",
    )
    (output_dir / "MyExpt_IdentifyPrimaryObjects.csv").write_text(
        "ObjectNumber,AreaShape_Area\n1,120\n2,95\n",
        encoding="utf-8",
    )
    load_result = load_cellprofiler_measurements(output_dir)

    merged, merge_warnings = merge_cellprofiler_tables(
        load_result.tables,
        metadata=load_result.metadata,
    )

    assert merged is not None
    assert list(merged.columns) == ["Image_Number", "FileName"]
    assert any(
        "MyExpt_IdentifyPrimaryObjects" in warning
        for warning in load_result.warnings
    )
    assert not merge_warnings


def test_merge_cellprofiler_tables_strict_mode_raises_on_missing_image_number(
    tmp_path: Path,
) -> None:
    tables = {
        "MyExpt_IdentifyPrimaryObjects": pd.DataFrame(
            {"ObjectNumber": [1, 2], "AreaShape_Area": [120, 95]}
        )
    }
    with pytest.raises(ValueError, match="missing required columns: Image_Number"):
        merge_cellprofiler_tables(tables, strict=True)


def test_load_cellprofiler_measurements_strict_mode_raises_on_legacy_object_table(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "MyExpt_IdentifyPrimaryObjects.csv").write_text(
        "ObjectNumber,AreaShape_Area\n1,120\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns: Image_Number"):
        load_cellprofiler_measurements(output_dir, strict=True)


def test_classify_cellprofiler_table_detects_object_table_from_filename() -> None:
    dataframe = pd.DataFrame({"ObjectNumber": [1], "AreaShape_Area": [100]})
    metadata = classify_cellprofiler_table("MyExpt_IdentifyPrimaryObjects", dataframe)
    assert metadata.table_type == "non_standard"
    assert metadata.legacy is True


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
    merged, warnings = merge_cellprofiler_tables(tables)

    assert merged is not None
    assert list(merged.columns) == ["Image_Number", "FileName", "Plate_Name"]
    assert not warnings


def test_cellprofiler_run_succeeded_rejects_pipeline_load_failures() -> None:
    assert cellprofiler_run_succeeded(0, stderr="Failed to load pipeline") is False
    assert cellprofiler_run_succeeded(0, stderr="Times reported are CPU") is True
    assert cellprofiler_run_succeeded(1, stderr="") is False
