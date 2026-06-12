"""Tests for Phase 15.1 GUI workflow shell helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bioimage_pipeline.analysis import CellProfilerWorkflowResult
from bioimage_pipeline.cellprofiler_runner import CellProfilerRunResult
from bioimage_pipeline.gui import (
    GuiWorkflowConfig,
    build_workflow_summary,
    read_log_tail,
    run_gui_workflow,
    validate_workflow_config,
)
from bioimage_pipeline.gui.workflow_shell import (
    default_oir_projection_engine_choice,
    load_measurements_preview,
)


@pytest.fixture(autouse=True)
def _mock_discovered_cellprofiler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    if request.node.name == "test_validate_workflow_config_reports_missing_cellprofiler":
        return

    executable = tmp_path / "CellProfiler.exe"
    executable.write_text("stub", encoding="utf-8")

    def _find(value: str | Path | None = None) -> Path | None:
        if value is None:
            return executable.resolve()
        text = str(value).strip()
        if not text or text == "cellprofiler":
            return executable.resolve()
        path = Path(text)
        if path.is_file():
            return path.resolve()
        return None

    monkeypatch.setattr(
        "bioimage_pipeline.gui.workflow_shell.find_cellprofiler_executable",
        _find,
    )


def _workflow_result(results_dir: Path) -> CellProfilerWorkflowResult:
    raw_dir = results_dir / "cellprofiler_raw"
    measurements_dir = results_dir / "measurements"
    masks_dir = results_dir / "masks"
    labels_dir = results_dir / "labels"
    qc_dir = results_dir / "qc"
    logs_dir = results_dir / "logs"
    for path in (raw_dir, measurements_dir, masks_dir, labels_dir, qc_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    (measurements_dir / "merged_measurements.csv").write_text(
        "Image_Number,ObjectNumber,AreaShape_Area\n1,1,42\n",
        encoding="utf-8",
    )
    (masks_dir / "sample_mask.tif").write_bytes(b"mask")
    (labels_dir / "sample_objects.tif").write_bytes(b"label")
    (qc_dir / "sample_qc_mask_overlay.png").write_bytes(b"png")
    stdout_log = logs_dir / "cellprofiler_stdout.log"
    stdout_log.write_text("done", encoding="utf-8")

    cp_run = CellProfilerRunResult(
        output_dir=raw_dir,
        command=["cellprofiler", "-c", "-r"],
        returncode=0,
        stdout="done",
        stderr="",
        log_files={"stdout": stdout_log},
    )
    return CellProfilerWorkflowResult(
        results_dir=results_dir,
        raw_output_dir=raw_dir,
        measurements_dir=measurements_dir,
        masks_dir=masks_dir,
        labels_dir=labels_dir,
        qc_dir=qc_dir,
        logs_dir=logs_dir,
        processed_images=["sample.tif"],
        tables={"MyExpt_Image": pd.DataFrame({"Image_Number": [1]})},
        table_summary={"MyExpt_Image": {"rows": 1, "columns": 1}},
        measurements=pd.DataFrame({"Image_Number": [1], "ObjectNumber": [1]}),
        mask_exports=[masks_dir / "sample_mask.tif"],
        label_exports=[labels_dir / "sample_objects.tif"],
        qc_artifacts={"sample.tif": {"mask_overlay": qc_dir / "sample_qc_mask_overlay.png"}},
        log_files={"stdout": stdout_log},
        cellprofiler_run=cp_run,
        timing={
            "cellprofiler_seconds": 1.0,
            "fiji_export_seconds": 2.0,
            "qc_seconds": 0.5,
            "total_seconds": 3.5,
        },
        export_engine="fiji",
        export_mode="batch",
    )


def test_validate_workflow_config_reports_missing_paths(tmp_path: Path) -> None:
    config = GuiWorkflowConfig(
        input_dir="",
        output_dir="",
        cppipe_path="",
    )

    errors = validate_workflow_config(config)

    assert "Select an input folder before running." in errors
    assert "Output folder is required." in errors


def test_validate_workflow_config_accepts_existing_paths(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "results"
    cppipe = tmp_path / "pipeline.cppipe"
    input_dir.mkdir()
    (input_dir / "sample.tif").write_bytes(b"image")
    cppipe.write_text("pipeline", encoding="utf-8")

    config = GuiWorkflowConfig(input_dir=input_dir, output_dir=output_dir, cppipe_path=cppipe)

    assert validate_workflow_config(config) == []


def test_validate_workflow_config_reports_empty_input_folder(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "results"
    cppipe = tmp_path / "pipeline.cppipe"
    input_dir.mkdir()
    cppipe.write_text("pipeline", encoding="utf-8")

    errors = validate_workflow_config(
        GuiWorkflowConfig(input_dir=input_dir, output_dir=output_dir, cppipe_path=cppipe)
    )

    assert errors == [f"No image files detected in: {input_dir}"]


def test_validate_workflow_config_reports_missing_cppipe(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample.tif").write_bytes(b"image")

    errors = validate_workflow_config(
        GuiWorkflowConfig(
            input_dir=input_dir,
            output_dir=tmp_path / "results",
            cppipe_path=tmp_path / "missing.cppipe",
        )
    )

    assert any("CellProfiler pipeline file does not exist" in error for error in errors)


def test_run_gui_workflow_delegates_to_headless_workflow(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "results"
    cppipe = tmp_path / "pipeline.cppipe"
    input_dir.mkdir()
    (input_dir / "sample.tif").write_bytes(b"image")
    cppipe.write_text("pipeline", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_runner(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return _workflow_result(output_dir)

    summary = run_gui_workflow(
        GuiWorkflowConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            cppipe_path=cppipe,
            cellprofiler_executable="cellprofiler",
            fiji_executable=None,
            export_fiji_tiffs=True,
            generate_qc=True,
        ),
        runner=fake_runner,
    )

    assert len(calls) == 1
    assert calls[0]["args"] == (input_dir, str(output_dir.resolve()), cppipe)
    assert calls[0]["kwargs"]["cellprofiler_executable"] == "cellprofiler"
    assert calls[0]["kwargs"]["export_fiji_tiffs"] is True
    assert summary.processed_count == 1
    assert summary.export_engine == "fiji"
    assert summary.export_mode == "batch"
    assert len(summary.qc_preview_files) == 1


def test_run_gui_workflow_passes_oir_projection_engine(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "results"
    cppipe = tmp_path / "pipeline.cppipe"
    input_dir.mkdir()
    (input_dir / "sample.tif").write_bytes(b"image")
    cppipe.write_text("pipeline", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_runner(*args, **kwargs):
        calls.append({"kwargs": kwargs})
        return _workflow_result(output_dir)

    run_gui_workflow(
        GuiWorkflowConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            cppipe_path=cppipe,
            oir_projection_engine="fiji",
        ),
        runner=fake_runner,
    )

    assert calls[0]["kwargs"]["oir_projection_engine"] == "fiji"


def test_validate_workflow_config_reports_missing_python_oir_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "results"
    cppipe = tmp_path / "pipeline.cppipe"
    input_dir.mkdir()
    (input_dir / "sample.oir").write_bytes(b"oir")
    cppipe.write_text("pipeline", encoding="utf-8")

    monkeypatch.setattr(
        "bioimage_pipeline.gui.workflow_shell.python_oir_dependencies_available",
        lambda: False,
    )

    errors = validate_workflow_config(
        GuiWorkflowConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            cppipe_path=cppipe,
            oir_projection_engine="python",
        )
    )

    assert any("aicsimageio/bfio" in error for error in errors)


def test_default_oir_projection_engine_choice_prefers_fiji_without_python_deps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fiji_exe = tmp_path / "ImageJ-win64.exe"
    fiji_exe.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        "bioimage_pipeline.gui.workflow_shell.python_oir_dependencies_available",
        lambda: False,
    )

    assert default_oir_projection_engine_choice(fiji_executable=fiji_exe) == "fiji"


def test_run_gui_workflow_reports_failed_subprocess_without_crashing(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "results"
    cppipe = tmp_path / "pipeline.cppipe"
    input_dir.mkdir()
    (input_dir / "sample.tif").write_bytes(b"image")
    cppipe.write_text("pipeline", encoding="utf-8")

    def failing_runner(*args, **kwargs):
        raise RuntimeError("CellProfiler command failed: boom")

    with pytest.raises(RuntimeError, match="CellProfiler command failed: boom"):
        run_gui_workflow(
            GuiWorkflowConfig(
                input_dir=input_dir,
                output_dir=output_dir,
                cppipe_path=cppipe,
            ),
            runner=failing_runner,
        )


def test_build_workflow_summary_lists_outputs(tmp_path: Path) -> None:
    result = _workflow_result(tmp_path / "results")

    summary = build_workflow_summary(result)

    assert summary.measurement_files[0].name == "merged_measurements.csv"
    assert summary.mask_files[0].name == "sample_mask.tif"
    assert summary.label_files[0].name == "sample_objects.tif"
    assert summary.qc_preview_files[0].name == "sample_qc_mask_overlay.png"
    assert any("Processed images: 1" in line for line in summary.to_display_lines())


def test_read_log_tail_returns_last_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "workflow.log"
    log_path.write_text("\n".join(f"line {index}" for index in range(5)), encoding="utf-8")

    assert read_log_tail(log_path, max_lines=2) == "line 3\nline 4"


def test_prepare_cellprofiler_input_dir_projects_oir(tmp_path: Path) -> None:
    from bioimage_pipeline.analysis import _prepare_cellprofiler_input_dir

    input_dir = tmp_path / "input"
    nested = input_dir / "plate"
    results_dir = tmp_path / "results"
    logs_dir = results_dir / "logs"
    nested.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    oir_path = nested / "sample.oir"
    oir_path.write_bytes(b"oir")

    projection_dir = results_dir / "oir_projection"
    fiji_exe = tmp_path / "ImageJ-win64.exe"
    fiji_exe.write_text("stub", encoding="utf-8")
    with patch(
        "bioimage_pipeline.oir_zmax_batch.run_oir_zmax_batch",
        return_value=MagicMock(
            input_dir=input_dir.resolve(),
            output_dir=projection_dir.resolve(),
            engine="fiji",
            processed=["sample.tif"],
            failed=[],
            files_created=[str(projection_dir / "sample.tif")],
            remapped_outputs=[],
            file_pairs=[MagicMock(input_oir=oir_path, output_tif=projection_dir / "sample.tif")],
            fiji_executable=fiji_exe.resolve(),
            fiji_headless=False,
            fiji_returncode=0,
            generated_macro_path=logs_dir / "stacking_zmax_generated.ijm",
            fiji_log_files={},
            file_profiles=[],
            cache_hits=[],
            reprojected=["sample.tif"],
            force_oir_reproject=False,
        ),
    ) as oir_batch:
        resolved_input, summary_log = _prepare_cellprofiler_input_dir(
            input_dir,
            results_dir=results_dir,
            logs_dir=logs_dir,
            oir_projection_engine="fiji",
            fiji_executable=fiji_exe,
        )

    oir_batch.assert_called_once_with(
        input_dir.resolve(),
        projection_dir,
        engine="fiji",
        logs_dir=logs_dir,
        fiji_executable=fiji_exe,
        fiji_headless=None,
        fiji_timeout=None,
        force_oir_reproject=False,
        lifecycle=None,
    )
    assert resolved_input == projection_dir.resolve()
    assert summary_log == logs_dir / "oir_projection_summary.json"
    assert (logs_dir / "prepare_input_profile.txt").exists()
    summary = json.loads(summary_log.read_text(encoding="utf-8"))
    assert summary["engine"] == "fiji"
    assert summary["fiji_executable"] == str(fiji_exe.resolve())
    assert summary["input_oir_files"] == [str(oir_path.resolve())]
    assert summary["output_tif_paths"] == [str((projection_dir / "sample.tif").resolve())]


def test_load_measurements_preview_limits_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "measurements.csv"
    csv_path.write_text("a\n1\n2\n3\n", encoding="utf-8")

    preview = load_measurements_preview(csv_path, max_rows=2)

    assert preview["a"].tolist() == [1, 2]


def test_load_imported_pipeline_validates_cppipe(tmp_path: Path) -> None:
    from bioimage_pipeline.gui.workflow_shell import load_imported_pipeline

    cppipe = tmp_path / "pipeline.cppipe"
    cppipe.write_text(
        """CellProfiler Pipeline: http://www.cellprofiler.org
Version:5
ModuleCount:1
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:'Unknown'|variable_revision_number:2|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Filter images?:Images only
""",
        encoding="utf-8",
    )

    state = load_imported_pipeline(cppipe)

    assert state.path == cppipe.resolve()
    assert state.pipeline.modules[0].name == "Images"
