"""Tests for the Phase 13 CellProfiler-to-Fiji workflow."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bioimage_pipeline.analysis import (
    CellProfilerWorkflowConfig,
    run_cellprofiler_workflow,
    run_cellprofiler_workflow_from_config,
)
from bioimage_pipeline.cellprofiler_runner import (
    CellProfilerMeasurementsResult,
    CellProfilerRunResult,
    RESULTS_LABELS_DIR,
    RESULTS_LOGS_DIR,
    RESULTS_MASKS_DIR,
    RESULTS_MEASUREMENTS_DIR,
    RESULTS_QC_DIR,
    RESULTS_RAW_DIR,
    copy_cellprofiler_measurements,
    discover_cellprofiler_tiff_files,
    extract_processed_image_names,
    format_cellprofiler_failure,
    summarize_cellprofiler_tables,
)
from bioimage_pipeline.export import (
    OrganizedFijiExports,
    classify_tiff_for_fiji_export,
    export_cellprofiler_tiff_for_fiji,
    organize_cellprofiler_tiffs_for_fiji,
)
from bioimage_pipeline.io import read_tiff, save_tiff
from bioimage_pipeline.qc import generate_qc_for_cellprofiler_results


@pytest.fixture(autouse=True)
def _stub_cellprofiler_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Workflow validation checks for CellProfiler before mocked runs start."""
    executable = tmp_path / "cellprofiler"
    executable.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        "bioimage_pipeline.cellprofiler_runner.find_cellprofiler_executable",
        lambda preferred=None: executable,
    )


def _successful_run_result(output_dir: Path, log_dir: Path) -> CellProfilerRunResult:
    log_files = {
        "stdout": log_dir / "cellprofiler_stdout.log",
        "stderr": log_dir / "cellprofiler_stderr.log",
        "command": log_dir / "cellprofiler_command.txt",
        "exit_code": log_dir / "cellprofiler_exit_code.txt",
    }
    for path in log_files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")
    return CellProfilerRunResult(
        output_dir=output_dir.resolve(),
        command=["cellprofiler", "-c", "-r"],
        returncode=0,
        stdout="done",
        stderr="",
        log_files=log_files,
    )


def test_classify_tiff_for_fiji_export_detects_mask_from_values() -> None:
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert classify_tiff_for_fiji_export(mask) == "mask"


def test_classify_tiff_for_fiji_export_detects_label_from_values() -> None:
    labels = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint16)
    assert classify_tiff_for_fiji_export(labels) == "label"


def test_classify_tiff_for_fiji_export_uses_filename_hints() -> None:
    intensity = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    assert (
        classify_tiff_for_fiji_export(intensity, filename="DNA_mask.tif") == "mask"
    )
    assert (
        classify_tiff_for_fiji_export(intensity, filename="Cells_objects.tif")
        == "label"
    )


def test_export_cellprofiler_tiff_for_fiji_rewrites_mask(tmp_path) -> None:
    source = tmp_path / "cp_mask.tif"
    save_tiff(source, np.array([[0, 1], [1, 0]], dtype=np.uint8))

    exported = export_cellprofiler_tiff_for_fiji(
        source,
        tmp_path / "masks" / "cp_mask.tif",
    )
    loaded = read_tiff(exported)

    assert loaded.dtype == np.uint8
    assert set(np.unique(loaded)).issubset({0, 255})


def test_organize_cellprofiler_tiffs_for_fiji_sorts_masks_and_labels(
    tmp_path,
) -> None:
    cp_output = tmp_path / "cellprofiler_raw"
    cp_output.mkdir()
    save_tiff(cp_output / "sample_mask.tif", np.array([[0, 1], [1, 0]], dtype=np.uint8))
    save_tiff(
        cp_output / "sample_objects.tif",
        np.array([[0, 1, 2], [0, 2, 1]], dtype=np.uint16),
    )

    organized = organize_cellprofiler_tiffs_for_fiji(
        cp_output,
        tmp_path / "masks",
        tmp_path / "labels",
    )

    assert len(organized.masks) == 1
    assert len(organized.labels) == 1
    assert organized.masks[0].parent.name == "masks"
    assert organized.labels[0].parent.name == "labels"


def test_copy_cellprofiler_measurements_copies_csv_files(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "MyExpt_Image.csv").write_text(
        "Image_Number,FileName\n1,sample.tif\n",
        encoding="utf-8",
    )

    copied = copy_cellprofiler_measurements(raw_dir, tmp_path / "measurements")

    assert len(copied) == 1
    assert copied[0].name == "MyExpt_Image.csv"
    assert "sample.tif" in copied[0].read_text(encoding="utf-8")


def test_discover_cellprofiler_tiff_files_skips_organized_dirs(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    save_tiff(raw_dir / "mask.tif", np.zeros((2, 2), dtype=np.uint8))
    save_tiff(raw_dir / "masks" / "organized.tif", np.zeros((2, 2), dtype=np.uint8))

    discovered = discover_cellprofiler_tiff_files(raw_dir)

    assert len(discovered) == 1
    assert discovered[0].name == "mask.tif"


def test_extract_processed_image_names_reads_image_table() -> None:
    tables = {
        "MyExpt_Image": pd.DataFrame(
            {"Image_Number": [1, 2], "FileName": ["a.tif", "b.tif"]}
        ),
        "MyExpt_Objects": pd.DataFrame(
            {"Image_Number": [1], "ObjectNumber": [1], "AreaShape_Area": [100]}
        ),
    }

    assert extract_processed_image_names(tables) == ["a.tif", "b.tif"]


def test_summarize_cellprofiler_tables_reports_counts() -> None:
    tables = {
        "MyExpt_Image": pd.DataFrame({"Image_Number": [1], "FileName": ["a.tif"]}),
    }

    summary = summarize_cellprofiler_tables(tables)

    assert summary == {"MyExpt_Image": {"rows": 1, "columns": 2}}


def test_format_cellprofiler_failure_includes_log_path(tmp_path) -> None:
    stderr_log = tmp_path / "cellprofiler_stderr.log"
    message = format_cellprofiler_failure(
        returncode=1,
        stdout="",
        stderr="pipeline failed",
        log_files={"stderr": stderr_log},
    )

    assert "pipeline failed" in message
    assert str(stderr_log) in message


def test_generate_qc_for_cellprofiler_results_creates_overlays(tmp_path) -> None:
    input_dir = tmp_path / "input"
    masks_dir = tmp_path / "masks"
    labels_dir = tmp_path / "labels"
    qc_dir = tmp_path / "qc"
    input_dir.mkdir()
    masks_dir.mkdir()
    labels_dir.mkdir()

    image = np.array([[40, 200], [40, 200]], dtype=np.uint8)
    mask = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    labels = np.array([[0, 1], [0, 2]], dtype=np.uint16)
    save_tiff(input_dir / "sample.tif", image)
    save_tiff(masks_dir / "sample_mask.tif", mask)
    save_tiff(labels_dir / "sample_objects.tif", labels)

    artifacts = generate_qc_for_cellprofiler_results(
        input_dir,
        masks_dir,
        labels_dir,
        qc_dir,
        ["sample.tif"],
    )

    assert "sample.tif" in artifacts
    assert artifacts["sample.tif"]["mask_overlay"].exists()
    assert artifacts["sample.tif"]["label_overlay"].exists()


@patch("bioimage_pipeline.analysis.organize_cellprofiler_tiffs_for_fiji")
@patch("bioimage_pipeline.analysis.generate_qc_for_cellprofiler_results")
@patch("bioimage_pipeline.analysis.copy_cellprofiler_measurements")
@patch("bioimage_pipeline.analysis.load_cellprofiler_measurements_lenient")
@patch("bioimage_pipeline.analysis.merge_cellprofiler_tables")
@patch("bioimage_pipeline.analysis.run_cellprofiler_pipeline_logged")
def test_run_cellprofiler_workflow_organizes_results(
    mock_run_logged: MagicMock,
    mock_merge: MagicMock,
    mock_load: MagicMock,
    mock_copy: MagicMock,
    mock_generate_qc: MagicMock,
    mock_organize: MagicMock,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    cppipe = tmp_path / "pipeline.cppipe"
    input_dir.mkdir()
    cppipe.write_text("pipeline", encoding="utf-8")
    save_tiff(input_dir / "sample.tif", np.zeros((8, 8), dtype=np.uint8))

    raw_dir = results_dir / RESULTS_RAW_DIR
    logs_dir = results_dir / RESULTS_LOGS_DIR
    masks_dir = results_dir / RESULTS_MASKS_DIR
    labels_dir = results_dir / RESULTS_LABELS_DIR
    qc_dir = results_dir / RESULTS_QC_DIR

    tables = {
        "MyExpt_Image": pd.DataFrame(
            {"Image_Number": [1], "FileName": ["sample.tif"]}
        ),
        "MyExpt_Objects": pd.DataFrame(
            {
                "Image_Number": [1],
                "ObjectNumber": [1],
                "AreaShape_Area": [120],
            }
        ),
    }
    merged = pd.DataFrame(
        {
            "Image_Number": [1],
            "ObjectNumber": [1],
            "AreaShape_Area": [120],
            "FileName": ["sample.tif"],
        }
    )
    mask_export = masks_dir / "sample_mask.tif"
    label_export = labels_dir / "sample_objects.tif"
    qc_artifact = qc_dir / "sample_qc_mask_overlay.png"

    mock_run_logged.return_value = _successful_run_result(raw_dir, logs_dir)
    mock_copy.return_value = [results_dir / RESULTS_MEASUREMENTS_DIR / "MyExpt_Image.csv"]
    mock_load.return_value = CellProfilerMeasurementsResult(
        tables=tables,
        metadata={},
        warnings=[],
    )
    mock_merge.return_value = (merged, [])
    mock_organize.return_value = OrganizedFijiExports(
        masks=[mask_export],
        labels=[label_export],
        intensity=[],
    )
    mock_generate_qc.return_value = {
        "sample.tif": {"mask_overlay": qc_artifact},
    }

    result = run_cellprofiler_workflow(
        input_dir,
        results_dir,
        cppipe,
        cellprofiler_executable=r"C:\Program Files\CellProfiler\CellProfiler.exe",
    )

    mock_run_logged.assert_called_once()
    run_kwargs = mock_run_logged.call_args.kwargs
    assert run_kwargs["output_dir"] == raw_dir
    assert run_kwargs["log_dir"] == logs_dir
    mock_copy.assert_called_once_with(raw_dir, results_dir / RESULTS_MEASUREMENTS_DIR)
    mock_load.assert_called_once_with(results_dir / RESULTS_MEASUREMENTS_DIR)
    mock_organize.assert_called_once_with(
        raw_dir,
        masks_dir,
        labels_dir,
        pattern="*.tif",
    )
    mock_generate_qc.assert_called_once()

    assert result.results_dir == results_dir.resolve()
    assert result.raw_output_dir == raw_dir.resolve()
    assert result.measurements_dir == (results_dir / RESULTS_MEASUREMENTS_DIR).resolve()
    assert result.masks_dir == masks_dir.resolve()
    assert result.labels_dir == labels_dir.resolve()
    assert result.qc_dir == qc_dir.resolve()
    assert result.logs_dir == logs_dir.resolve()
    assert result.processed_images == ["sample.tif"]
    assert result.mask_exports == [mask_export]
    assert result.label_exports == [label_export]
    assert result.qc_artifacts["sample.tif"]["mask_overlay"] == qc_artifact
    assert (logs_dir / "workflow_summary.json").exists()

    summary = result.to_dict()
    assert summary["results_dir"] == str(results_dir.resolve())
    assert summary["mask_exports"] == [str(mask_export)]
    assert summary["cellprofiler_returncode"] == 0


@patch("bioimage_pipeline.analysis.organize_cellprofiler_tiffs_for_fiji")
@patch("bioimage_pipeline.analysis.generate_qc_for_cellprofiler_results")
@patch("bioimage_pipeline.analysis.copy_cellprofiler_measurements")
@patch("bioimage_pipeline.analysis.load_cellprofiler_measurements_lenient")
@patch("bioimage_pipeline.analysis.merge_cellprofiler_tables")
@patch("bioimage_pipeline.analysis.run_cellprofiler_pipeline_logged")
def test_run_cellprofiler_workflow_can_skip_fiji_and_qc(
    mock_run_logged: MagicMock,
    mock_merge: MagicMock,
    mock_load: MagicMock,
    mock_copy: MagicMock,
    mock_generate_qc: MagicMock,
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
    tables = {"MyExpt_Image": pd.DataFrame({"Image_Number": [1]})}

    mock_run_logged.return_value = _successful_run_result(raw_dir, logs_dir)
    mock_load.return_value = CellProfilerMeasurementsResult(
        tables=tables,
        metadata={},
        warnings=[],
    )

    result = run_cellprofiler_workflow(
        input_dir,
        results_dir,
        cppipe,
        export_fiji_tiffs=False,
        generate_qc=False,
        merge_measurements=False,
    )

    mock_organize.assert_not_called()
    mock_generate_qc.assert_not_called()
    mock_merge.assert_not_called()
    assert result.mask_exports == []
    assert result.label_exports == []
    assert result.qc_artifacts == {}
    assert result.measurements is None


@patch("bioimage_pipeline.analysis.run_cellprofiler_pipeline_logged")
def test_run_cellprofiler_workflow_raises_on_failed_run(
    mock_run_logged: MagicMock,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    cppipe = tmp_path / "pipeline.cppipe"
    input_dir.mkdir()
    cppipe.write_text("pipeline", encoding="utf-8")

    logs_dir = results_dir / RESULTS_LOGS_DIR
    logs_dir.mkdir(parents=True)
    stderr_log = logs_dir / "cellprofiler_stderr.log"
    stderr_log.write_text("boom", encoding="utf-8")

    mock_run_logged.return_value = CellProfilerRunResult(
        output_dir=(results_dir / RESULTS_RAW_DIR).resolve(),
        command=["cellprofiler"],
        returncode=1,
        stdout="",
        stderr="boom",
        log_files={"stderr": stderr_log},
    )

    with pytest.raises(RuntimeError, match="CellProfiler command failed"):
        run_cellprofiler_workflow(input_dir, results_dir, cppipe)

    assert (logs_dir / "workflow_summary.json").exists()


def test_run_cellprofiler_workflow_from_config_uses_structured_dirs(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    cppipe = tmp_path / "pipeline.cppipe"
    input_dir.mkdir()
    cppipe.write_text("pipeline", encoding="utf-8")

    config = CellProfilerWorkflowConfig(
        input_dir=input_dir,
        output_dir=results_dir,
        cppipe_path=cppipe,
    )

    raw_dir = results_dir / RESULTS_RAW_DIR
    logs_dir = results_dir / RESULTS_LOGS_DIR

    with (
        patch(
            "bioimage_pipeline.analysis.run_cellprofiler_pipeline_logged",
            return_value=_successful_run_result(raw_dir, logs_dir),
        ),
        patch(
            "bioimage_pipeline.analysis.copy_cellprofiler_measurements",
            return_value=[],
        ),
        patch(
            "bioimage_pipeline.analysis.load_cellprofiler_measurements_lenient",
            return_value=CellProfilerMeasurementsResult(
                tables={"MyExpt_Image": pd.DataFrame({"Image_Number": [1]})},
                metadata={},
                warnings=[],
            ),
        ),
        patch(
            "bioimage_pipeline.analysis.merge_cellprofiler_tables",
            return_value=(pd.DataFrame({"Image_Number": [1]}), []),
        ),
        patch(
            "bioimage_pipeline.analysis.organize_cellprofiler_tiffs_for_fiji",
            return_value=OrganizedFijiExports(masks=[], labels=[], intensity=[]),
        ),
        patch(
            "bioimage_pipeline.analysis.generate_qc_for_cellprofiler_results",
            return_value={},
        ),
    ):
        result = run_cellprofiler_workflow_from_config(config)

    assert result.results_dir == results_dir.resolve()
    assert result.measurements_dir == (results_dir / RESULTS_MEASUREMENTS_DIR).resolve()


@patch("bioimage_pipeline.analysis.run_cellprofiler_pipeline_logged")
def test_run_cellprofiler_workflow_surfaces_module_error_from_logs(
    mock_run_logged: MagicMock,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    cppipe = tmp_path / "pipeline.cppipe"
    input_dir.mkdir()
    cppipe.write_text("pipeline", encoding="utf-8")

    raw_dir = results_dir / RESULTS_RAW_DIR
    logs_dir = results_dir / RESULTS_LOGS_DIR
    raw_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    stdout_log = logs_dir / "cellprofiler_stdout.log"
    stderr_log = logs_dir / "cellprofiler_stderr.log"
    stdout_log.write_text("Times reported are CPU seconds\n", encoding="utf-8")
    stderr_log.write_text(
        "Error in module IdentifyPrimaryObjects: missing input image DNA\n",
        encoding="utf-8",
    )

    mock_run_logged.return_value = CellProfilerRunResult(
        output_dir=raw_dir.resolve(),
        command=["cellprofiler"],
        returncode=0,
        stdout="Times reported are CPU seconds\n",
        stderr="",
        log_files={"stdout": stdout_log, "stderr": stderr_log},
    )

    with pytest.raises(RuntimeError, match="CellProfiler pipeline failed"):
        run_cellprofiler_workflow(
            input_dir,
            results_dir,
            cppipe,
            export_fiji_tiffs=False,
            generate_qc=False,
        )

    assert (logs_dir / "workflow_summary.json").exists()


@patch("bioimage_pipeline.analysis.run_cellprofiler_pipeline_logged")
def test_run_cellprofiler_workflow_continues_when_no_csv_exports(
    mock_run_logged: MagicMock,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    results_dir = tmp_path / "results"
    cppipe = tmp_path / "pipeline.cppipe"
    input_dir.mkdir()
    cppipe.write_text("pipeline", encoding="utf-8")

    raw_dir = results_dir / RESULTS_RAW_DIR
    logs_dir = results_dir / RESULTS_LOGS_DIR
    raw_dir.mkdir(parents=True)
    (results_dir / RESULTS_MEASUREMENTS_DIR).mkdir(parents=True)

    mock_run_logged.return_value = _successful_run_result(raw_dir, logs_dir)

    result = run_cellprofiler_workflow(
        input_dir,
        results_dir,
        cppipe,
        export_fiji_tiffs=False,
        generate_qc=False,
        merge_measurements=False,
    )

    assert result.tables == {}
    assert result.import_warnings is not None
    assert any("No CSV files" in warning for warning in result.import_warnings)
    assert (logs_dir / "workflow_summary.json").exists()
