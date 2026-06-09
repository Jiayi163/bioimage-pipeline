"""Tests for Phase 14 Fiji/ImageJ batch export."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from bioimage_pipeline.analysis import run_cellprofiler_workflow
from bioimage_pipeline.cellprofiler_runner import (
    CellProfilerMeasurementsResult,
    CellProfilerRunResult,
    RESULTS_LABELS_DIR,
    RESULTS_LOGS_DIR,
    RESULTS_MASKS_DIR,
    RESULTS_RAW_DIR,
)
from bioimage_pipeline.fiji_runner import (
    DEFAULT_FIJI_EXPORT_MACRO,
    FijiExportResult,
    run_fiji_batch_export,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_SCRIPT = REPO_ROOT / "examples" / "run_fiji_export.py"


def _successful_cp_run(output_dir: Path, log_dir: Path) -> CellProfilerRunResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = log_dir / "cellprofiler_stdout.log"
    stderr = log_dir / "cellprofiler_stderr.log"
    command = log_dir / "cellprofiler_command.txt"
    exit_code = log_dir / "cellprofiler_exit_code.txt"
    for path in (stdout, stderr, command, exit_code):
        path.write_text("ok", encoding="utf-8")
    return CellProfilerRunResult(
        output_dir=output_dir.resolve(),
        command=["cellprofiler", "-c", "-r"],
        returncode=0,
        stdout="done",
        stderr="",
        log_files={
            "stdout": stdout,
            "stderr": stderr,
            "command": command,
            "exit_code": exit_code,
        },
    )


def test_default_fiji_export_macro_exists() -> None:
    assert DEFAULT_FIJI_EXPORT_MACRO.is_file()


@patch("bioimage_pipeline.fiji_runner.subprocess.run")
def test_run_fiji_batch_export_uses_one_subprocess(
    mock_run: MagicMock,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "cellprofiler_raw"
    masks_dir = tmp_path / "masks"
    labels_dir = tmp_path / "labels"
    log_dir = tmp_path / "logs"
    macro = tmp_path / "export_folder.ijm"
    executable = tmp_path / "ImageJ-win64.exe"
    input_dir.mkdir()
    macro.write_text("// macro", encoding="utf-8")
    executable.write_text("", encoding="utf-8")

    def complete_export(*args, **kwargs):
        masks_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        (masks_dir / "sample_mask.tif").write_bytes(b"mask")
        (labels_dir / "sample_objects.tif").write_bytes(b"label")
        return subprocess.CompletedProcess(args[0], 0, stdout="done", stderr="")

    mock_run.side_effect = complete_export

    result = run_fiji_batch_export(
        input_dir,
        masks_dir,
        labels_dir,
        macro_path=macro,
        fiji_executable=executable,
        log_dir=log_dir,
    )

    mock_run.assert_called_once()
    command = mock_run.call_args.args[0]
    assert command[:2] == [str(executable.resolve()), "--headless"]
    assert command[2:4] == ["-macro", str(macro.resolve())]
    assert command[4].split("|")[:3] == [
        str(input_dir.resolve()),
        str(masks_dir.resolve()),
        str(labels_dir.resolve()),
    ]
    assert result.succeeded
    assert result.mask_exports == [masks_dir / "sample_mask.tif"]
    assert result.label_exports == [labels_dir / "sample_objects.tif"]
    assert (log_dir / "fiji_command.txt").exists()


def test_run_fiji_batch_export_missing_executable_raises(tmp_path: Path) -> None:
    input_dir = tmp_path / "cellprofiler_raw"
    masks_dir = tmp_path / "masks"
    labels_dir = tmp_path / "labels"
    macro = tmp_path / "export_folder.ijm"
    input_dir.mkdir()
    macro.write_text("// macro", encoding="utf-8")

    missing = tmp_path / "missing-imagej.exe"
    try:
        run_fiji_batch_export(
            input_dir,
            masks_dir,
            labels_dir,
            macro_path=macro,
            fiji_executable=missing,
        )
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("Expected missing Fiji executable to raise FileNotFoundError")


@patch("bioimage_pipeline.analysis.organize_cellprofiler_tiffs_for_fiji")
@patch("bioimage_pipeline.analysis.run_fiji_batch_export")
@patch("bioimage_pipeline.analysis.generate_qc_for_cellprofiler_results")
@patch("bioimage_pipeline.analysis.copy_cellprofiler_measurements")
@patch("bioimage_pipeline.analysis.load_cellprofiler_measurements")
@patch("bioimage_pipeline.analysis.merge_cellprofiler_tables")
@patch("bioimage_pipeline.analysis.run_cellprofiler_pipeline_logged")
def test_workflow_uses_batch_fiji_export_when_available(
    mock_run_logged: MagicMock,
    mock_merge: MagicMock,
    mock_load: MagicMock,
    mock_copy: MagicMock,
    mock_generate_qc: MagicMock,
    mock_run_fiji: MagicMock,
    mock_organize: MagicMock,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    cppipe = tmp_path / "pipeline.cppipe"
    input_dir.mkdir()
    cppipe.write_text("pipeline", encoding="utf-8")

    raw_dir = results_dir / RESULTS_RAW_DIR
    logs_dir = results_dir / RESULTS_LOGS_DIR
    masks_dir = results_dir / RESULTS_MASKS_DIR
    labels_dir = results_dir / RESULTS_LABELS_DIR
    mask_export = masks_dir / "sample_mask.tif"
    label_export = labels_dir / "sample_objects.tif"

    mock_run_logged.return_value = _successful_cp_run(raw_dir, logs_dir)
    mock_copy.return_value = []
    mock_load.return_value = CellProfilerMeasurementsResult(
        tables={"MyExpt_Image": pd.DataFrame({"Image_Number": [1], "FileName": ["sample.tif"]})},
        metadata={},
        warnings=[],
    )
    mock_merge.return_value = (pd.DataFrame({"Image_Number": [1]}), [])
    mock_generate_qc.return_value = {}
    mock_run_fiji.return_value = FijiExportResult(
        input_dir=raw_dir.resolve(),
        masks_dir=masks_dir.resolve(),
        labels_dir=labels_dir.resolve(),
        macro_path=DEFAULT_FIJI_EXPORT_MACRO.resolve(),
        executable=(tmp_path / "ImageJ-win64.exe").resolve(),
        command=["ImageJ-win64.exe", "--headless", "-macro", "export_folder.ijm"],
        returncode=0,
        stdout="done",
        stderr="",
        log_files={"stdout": logs_dir / "fiji_stdout.log"},
        mask_exports=[mask_export],
        label_exports=[label_export],
    )

    result = run_cellprofiler_workflow(
        input_dir,
        results_dir,
        cppipe,
        fiji_executable=tmp_path / "ImageJ-win64.exe",
    )

    mock_run_fiji.assert_called_once()
    mock_organize.assert_not_called()
    assert result.export_engine == "fiji"
    assert result.export_mode == "batch"
    assert result.mask_exports == [mask_export]
    assert result.label_exports == [label_export]
    assert result.timing is not None
    assert set(result.timing) == {
        "cellprofiler_seconds",
        "fiji_export_seconds",
        "qc_seconds",
        "total_seconds",
    }
    assert (logs_dir / "workflow_summary.json").exists()


def test_run_fiji_export_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, str(CLI_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--input-dir" in result.stdout
    assert "--masks-dir" in result.stdout
    assert "--labels-dir" in result.stdout
