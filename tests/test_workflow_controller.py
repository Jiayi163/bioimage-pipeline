"""Tests for thin GUI workflow controller helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from bioimage_pipeline.gui.run_settings import (
    CPPIPE_PATH_KEY,
    FORCE_OIR_REPROJECT_KEY,
    INPUT_DIR_KEY,
    OUTPUT_DIR_KEY,
    collect_run_settings_from_values,
    parse_bool_setting,
)
from bioimage_pipeline.gui.workflow_controller import (
    WorkflowFormValues,
    build_gui_workflow_config,
    format_measurements_preview_text,
    prepare_workflow_run,
    workflow_form_values_from_settings,
    workflow_log_paths,
)

SAMPLE_CPPIPE = """CellProfiler Pipeline: http://www.cellprofiler.org
Version:5

Images:[module_num:1|svn_version:'Unknown'|variable_revision_number:1|show_window:False|notes:[]]
Filter images?:No

IdentifyPrimaryObjects:[module_num:2|svn_version:'Unknown'|variable_revision_number:1|show_window:False|notes:[]]
Select the input image:Green
Name the primary objects to be identified:Spots
Threshold strategy:Adaptive
Thresholding method:Otsu

ExportToSpreadsheet:[module_num:3|svn_version:'Unknown'|variable_revision_number:1|show_window:False|notes:[]]
Select the column delimiter:Comma
"""


def test_parse_bool_setting() -> None:
    assert parse_bool_setting("true", default=False) is True
    assert parse_bool_setting("false", default=True) is False
    assert parse_bool_setting(None, default=True) is True


def test_collect_run_settings_from_values_persists_workflow_paths() -> None:
    payload = collect_run_settings_from_values(
        cellprofiler_executable="cp.exe",
        fiji_executable="fiji.exe",
        cppipe_path="pipeline.cppipe",
        input_dir="input",
        output_dir="output",
        force_oir_reproject=True,
    )

    assert payload[CPPIPE_PATH_KEY] == "pipeline.cppipe"
    assert payload[INPUT_DIR_KEY] == "input"
    assert payload[OUTPUT_DIR_KEY] == "output"
    assert payload[FORCE_OIR_REPROJECT_KEY] == "true"


def test_build_gui_workflow_config_maps_force_oir_reproject() -> None:
    values = WorkflowFormValues(
        input_dir="input",
        output_dir="output",
        cppipe_path="pipeline.cppipe",
        cellprofiler_executable="cellprofiler",
        fiji_executable="",
        fiji_macro_path="",
        export_fiji_tiffs=True,
        generate_qc=True,
        oir_projection_engine="auto",
        oir_projection_method="max",
        force_oir_reproject=True,
    )

    config = build_gui_workflow_config(values)

    assert config.force_oir_reproject is True
    assert config.oir_projection_engine == "auto"


def test_workflow_form_values_from_settings_reads_saved_flags() -> None:
    values = workflow_form_values_from_settings(
        {
            INPUT_DIR_KEY: "in",
            OUTPUT_DIR_KEY: "out",
            FORCE_OIR_REPROJECT_KEY: "true",
        },
        default_oir_projection_engine="fiji",
    )

    assert values.input_dir == "in"
    assert values.output_dir == "out"
    assert values.force_oir_reproject is True


def test_format_measurements_preview_text_reads_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "merged_measurements.csv"
    csv_path.write_text("Image_Number,ObjectNumber\n1,1\n1,2\n", encoding="utf-8")

    text = format_measurements_preview_text([csv_path], max_rows=5)

    assert "ObjectNumber" in text
    assert "1" in text


def test_workflow_log_paths_under_output_dir(tmp_path: Path) -> None:
    paths = workflow_log_paths(tmp_path / "results")

    assert paths["stdout"].name == "cellprofiler_stdout.log"
    assert "logs" in str(paths["stdout"])


def test_prepare_workflow_run_validates_imported_pipeline(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "img.tif").write_bytes(b"fake")
    cppipe_path = tmp_path / "pipeline.cppipe"
    cppipe_path.write_text(SAMPLE_CPPIPE, encoding="utf-8")

    values = WorkflowFormValues(
        input_dir=str(input_dir),
        output_dir=str(tmp_path / "output"),
        cppipe_path=str(cppipe_path),
        cellprofiler_executable="cellprofiler",
        fiji_executable="",
        fiji_macro_path="",
        export_fiji_tiffs=True,
        generate_qc=True,
        oir_projection_engine="fiji",
        oir_projection_method="max",
        force_oir_reproject=False,
    )

    with patch(
        "bioimage_pipeline.gui.workflow_shell.find_cellprofiler_executable",
        return_value=tmp_path / "CellProfiler.exe",
    ):
        (tmp_path / "CellProfiler.exe").write_text("stub", encoding="utf-8")
        preparation = prepare_workflow_run(values)

    assert preparation.config.cppipe_path == str(cppipe_path.resolve())
    assert preparation.requires_oir_method_dialog is False
