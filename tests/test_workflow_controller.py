"""Tests for thin GUI workflow controller helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from bioimage_pipeline.gui.run_settings import (
    CELLPROFILER_SETTINGS_KEY,
    CPPIPE_PATH_KEY,
    FORCE_OIR_REPROJECT_KEY,
    INPUT_DIR_KEY,
    OUTPUT_DIR_KEY,
    collect_run_settings_from_values,
    extract_recent_workflow_paths,
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


def test_extract_recent_workflow_paths_returns_saved_experiment_paths() -> None:
    recent = extract_recent_workflow_paths(
        {
            INPUT_DIR_KEY: " C:\\data\\in ",
            OUTPUT_DIR_KEY: "C:\\data\\out",
            CPPIPE_PATH_KEY: "C:\\assay.cppipe",
        }
    )

    assert recent[INPUT_DIR_KEY] == "C:\\data\\in"
    assert recent[OUTPUT_DIR_KEY] == "C:\\data\\out"
    assert recent[CPPIPE_PATH_KEY] == "C:\\assay.cppipe"


def test_build_main_workflow_panel_starts_with_blank_experiment_paths() -> None:
    import tkinter as tk
    from tkinter import ttk

    from bioimage_pipeline.gui.panels.main_workflow_panel import build_main_workflow_panel

    root = tk.Tk()
    root.withdraw()
    frame = ttk.Frame(root)
    panel = build_main_workflow_panel(
        frame,
        saved_settings={
            INPUT_DIR_KEY: "C:\\old\\input",
            OUTPUT_DIR_KEY: "C:\\old\\output",
            CELLPROFILER_SETTINGS_KEY: "cellprofiler.exe",
        },
        browse_folder=lambda _entry: None,
        browse_file=lambda _entry: None,
    )

    assert panel.input_dir_entry.get() == ""
    assert panel.output_dir_entry.get() == ""
    assert panel.cellprofiler_executable_entry.get() == "cellprofiler.exe"
    assert panel.clear_session_button is not None
    root.destroy()


def test_clear_session_button_clears_experiment_fields_but_keeps_cellprofiler_path() -> None:
    import tkinter as tk
    from tkinter import ttk

    from bioimage_pipeline.gui.panels.main_workflow_panel import build_main_workflow_panel
    from bioimage_pipeline.gui.workflow_session_reset import clear_experiment_paths_from_settings

    root = tk.Tk()
    root.withdraw()
    frame = ttk.Frame(root)
    panel = build_main_workflow_panel(
        frame,
        saved_settings={CELLPROFILER_SETTINGS_KEY: "C:\\CellProfiler.exe"},
        browse_folder=lambda _entry: None,
        browse_file=lambda _entry: None,
    )
    panel.input_dir_entry.insert(0, "C:\\data\\in")
    panel.output_dir_entry.insert(0, "C:\\data\\out")

    panel.input_dir_entry.delete(0, "end")
    panel.output_dir_entry.delete(0, "end")

    cleared_settings = clear_experiment_paths_from_settings(
        {
            INPUT_DIR_KEY: "C:\\data\\in",
            OUTPUT_DIR_KEY: "C:\\data\\out",
            CELLPROFILER_SETTINGS_KEY: panel.cellprofiler_executable_entry.get(),
        }
    )

    assert panel.input_dir_entry.get() == ""
    assert panel.output_dir_entry.get() == ""
    assert panel.cellprofiler_executable_entry.get() == "C:\\CellProfiler.exe"
    assert INPUT_DIR_KEY not in cleared_settings
    assert OUTPUT_DIR_KEY not in cleared_settings
    assert cleared_settings[CELLPROFILER_SETTINGS_KEY] == "C:\\CellProfiler.exe"
    root.destroy()


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
