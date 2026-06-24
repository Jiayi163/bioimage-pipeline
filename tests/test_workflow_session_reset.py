"""Tests for experiment session reset helpers."""

from __future__ import annotations

from pathlib import Path

from bioimage_pipeline.gui.run_settings import (
    CELLPROFILER_SETTINGS_KEY,
    CPPIPE_PATH_KEY,
    FIJI_MACRO_PATH_KEY,
    FIJI_SETTINGS_KEY,
    INPUT_DIR_KEY,
    OUTPUT_DIR_KEY,
    save_gui_run_settings,
)
from bioimage_pipeline.gui.workflow_session_reset import (
    clear_experiment_paths_from_settings,
    needs_reset_confirmation,
    workflow_has_experiment_fields,
    workflow_run_actions_enabled,
)


def test_clear_experiment_paths_from_settings_removes_only_experiment_paths() -> None:
    settings = {
        CPPIPE_PATH_KEY: "C:\\assay.cppipe",
        INPUT_DIR_KEY: "C:\\in",
        OUTPUT_DIR_KEY: "C:\\out",
        CELLPROFILER_SETTINGS_KEY: "C:\\CellProfiler.exe",
        FIJI_SETTINGS_KEY: "C:\\Fiji.app\\ImageJ-win64.exe",
        FIJI_MACRO_PATH_KEY: "C:\\macros\\export.ijm",
    }

    cleared = clear_experiment_paths_from_settings(settings)

    assert INPUT_DIR_KEY not in cleared
    assert OUTPUT_DIR_KEY not in cleared
    assert CPPIPE_PATH_KEY not in cleared
    assert cleared[CELLPROFILER_SETTINGS_KEY] == "C:\\CellProfiler.exe"
    assert cleared[FIJI_SETTINGS_KEY] == "C:\\Fiji.app\\ImageJ-win64.exe"
    assert cleared[FIJI_MACRO_PATH_KEY] == "C:\\macros\\export.ijm"


def test_workflow_has_experiment_fields_detects_loaded_pipeline_without_path_text() -> None:
    assert workflow_has_experiment_fields(
        cppipe_path="",
        input_dir="",
        output_dir="",
        pipeline_loaded=True,
    )


def test_needs_reset_confirmation_when_results_are_displayed() -> None:
    assert needs_reset_confirmation(
        cppipe_path="",
        input_dir="",
        output_dir="",
        pipeline_loaded=False,
        has_result_display=True,
        run_in_progress=False,
    )


def test_workflow_run_actions_enabled_requires_all_paths_and_not_running() -> None:
    assert workflow_run_actions_enabled(
        cppipe_path="pipeline.cppipe",
        input_dir="input",
        output_dir="output",
        running=False,
    )
    assert not workflow_run_actions_enabled(
        cppipe_path="pipeline.cppipe",
        input_dir="input",
        output_dir="",
        running=False,
    )
    assert not workflow_run_actions_enabled(
        cppipe_path="pipeline.cppipe",
        input_dir="input",
        output_dir="output",
        running=True,
    )


def test_clear_experiment_paths_from_settings_does_not_delete_output_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    output_file = output_dir / "merged_measurements.csv"
    output_file.write_text("Image_Number\n1\n", encoding="utf-8")
    settings_path = tmp_path / "gui_run_settings.json"

    save_gui_run_settings(
        {
            OUTPUT_DIR_KEY: str(output_dir),
            CELLPROFILER_SETTINGS_KEY: str(tmp_path / "CellProfiler.exe"),
        },
        settings_path=settings_path,
    )

    cleared = clear_experiment_paths_from_settings(
        {
            OUTPUT_DIR_KEY: str(output_dir),
            CELLPROFILER_SETTINGS_KEY: str(tmp_path / "CellProfiler.exe"),
        }
    )
    save_gui_run_settings(cleared, settings_path=settings_path)

    assert output_file.is_file()
    assert output_file.read_text(encoding="utf-8") == "Image_Number\n1\n"
