"""Tests for Phase 15.2 CellProfiler pipeline I/O and module catalog."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from bioimage_pipeline.cppipe_io import (
    MINIMAL_GUI_PIPELINE_MODULES,
    append_module,
    create_pipeline_from_catalog,
    load_cppipe,
    move_module,
    parse_cppipe_text,
    remove_module,
    save_cppipe,
    summarize_modules,
    update_module_setting,
    validate_cppipe,
)
from bioimage_pipeline.cellprofiler_runner import run_cellprofiler_pipeline_logged
from bioimage_pipeline.gui import (
    add_catalog_module_to_pipeline,
    create_default_pipeline_builder_state,
    load_pipeline_builder_state,
    move_pipeline_module,
    remove_pipeline_module,
    save_pipeline_builder_state,
    select_pipeline_module,
    update_pipeline_module_setting,
)
from bioimage_pipeline.pipeline_catalog import (
    get_module_definition,
    list_categories,
    list_modules,
    search_modules,
)


SAMPLE_CPPIPE = """CellProfiler Pipeline: http://www.cellprofiler.org
Version:5

Images:[module_num:1|svn_version:'Unknown'|variable_revision_number:1|show_window:False|notes:[]]
Filter images?:No

IdentifyPrimaryObjects:[module_num:2|svn_version:'Unknown'|variable_revision_number:1|show_window:False|notes:[]]
Select the input image:DNA
Name the primary objects to be identified:Nuclei

ExportToSpreadsheet:[module_num:3|svn_version:'Unknown'|variable_revision_number:1|show_window:False|notes:[]]
Select the column delimiter:Comma
"""


def test_parse_cppipe_text_extracts_modules_and_settings() -> None:
    pipeline = parse_cppipe_text(SAMPLE_CPPIPE)

    assert [module.name for module in pipeline.modules] == [
        "Images",
        "IdentifyPrimaryObjects",
        "ExportToSpreadsheet",
    ]
    assert pipeline.modules[1].module_num == 2
    assert pipeline.modules[1].settings[0].key == "Select the input image"
    assert pipeline.modules[1].settings[0].value == "DNA"
    assert summarize_modules(pipeline.modules) == [
        "1. Images",
        "2. IdentifyPrimaryObjects",
        "3. ExportToSpreadsheet",
    ]


def test_save_and_load_cppipe_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "pipeline.cppipe"
    source.write_text(SAMPLE_CPPIPE, encoding="utf-8")

    pipeline = load_cppipe(source)
    saved = save_cppipe(pipeline, tmp_path / "saved.cppipe")
    loaded = load_cppipe(saved)

    assert [module.name for module in loaded.modules] == [
        "Images",
        "IdentifyPrimaryObjects",
        "ExportToSpreadsheet",
    ]
    assert validate_cppipe(loaded) == []


def test_update_move_remove_and_append_module() -> None:
    pipeline = parse_cppipe_text(SAMPLE_CPPIPE)

    updated = update_module_setting(
        pipeline,
        1,
        "Name the primary objects to be identified",
        "Cells",
    )
    assert "Name the primary objects to be identified:Cells" in updated.modules[1].lines

    moved = move_module(updated, 2, 1)
    assert [module.name for module in moved.modules] == [
        "Images",
        "ExportToSpreadsheet",
        "IdentifyPrimaryObjects",
    ]
    assert [module.module_num for module in moved.modules] == [1, 2, 3]

    removed = remove_module(moved, 1)
    assert [module.name for module in removed.modules] == [
        "Images",
        "IdentifyPrimaryObjects",
    ]

    appended = append_module(removed, "SaveImages")
    assert appended.modules[-1].name == "SaveImages"
    assert appended.modules[-1].module_num == 3


def test_module_catalog_search_and_lookup() -> None:
    names = [module.name for module in list_modules()]

    assert "IdentifyPrimaryObjects" in names
    assert "Output" in list_categories()
    assert search_modules("intensity")[0].name in {
        "MeasureObjectIntensity",
        "IdentifyPrimaryObjects",
    }
    assert get_module_definition("ExportToSpreadsheet").category == "Output"


def test_phase_15_2_catalog_includes_setting_metadata() -> None:
    module = get_module_definition("SaveImages")
    parameters = {parameter.label: parameter for parameter in module.parameters}

    assert module.display_name == "SaveImages"
    assert module.category == "Output"
    assert parameters["Select the type of image to save"].choices == (
        "Image",
        "Mask",
        "Objects",
    )
    assert parameters["Saved file format"].default == "tiff"
    assert parameters["Enter single file name"].visibility.mode == "conditional"


def test_conditional_visible_settings_follow_current_values() -> None:
    metadata = get_module_definition("Metadata")
    default_labels = [parameter.label for parameter in metadata.visible_parameters()]
    enabled_labels = [
        parameter.label
        for parameter in metadata.visible_parameters({"Extract metadata?": "Yes"})
    ]

    assert "Regular expression to extract from file name" not in default_labels
    assert "Regular expression to extract from file name" in enabled_labels

    save_images = get_module_definition("SaveImages")
    default_save_labels = [parameter.label for parameter in save_images.visible_parameters()]
    single_name_labels = [
        parameter.label
        for parameter in save_images.visible_parameters(
            {"Select method for constructing file names": "Single name"}
        )
    ]

    assert "Select the image to save" in default_save_labels
    assert "Enter single file name" not in default_save_labels
    assert "Enter single file name" in single_name_labels


def test_pipeline_builder_state_adds_and_saves_catalog_module(tmp_path: Path) -> None:
    cppipe = tmp_path / "pipeline.cppipe"
    cppipe.write_text(SAMPLE_CPPIPE, encoding="utf-8")

    state = load_pipeline_builder_state(cppipe, query="SaveImages")
    assert state.catalog_modules[0].name == "SaveImages"

    updated = add_catalog_module_to_pipeline(state, 0)
    assert updated.pipeline.modules[-1].name == "SaveImages"

    saved = save_pipeline_builder_state(updated, tmp_path / "edited.cppipe")
    loaded = load_cppipe(saved)
    assert loaded.modules[-1].name == "SaveImages"


def test_generated_minimal_cppipe_contains_expected_module_blocks() -> None:
    pipeline = create_pipeline_from_catalog()
    text = pipeline.to_text()

    assert [module.name for module in pipeline.modules] == list(MINIMAL_GUI_PIPELINE_MODULES)
    for index, module_name in enumerate(MINIMAL_GUI_PIPELINE_MODULES, start=1):
        assert f"{module_name}:[module_num:{index}|" in text
    assert "Name the primary objects to be identified:Nuclei" in text
    assert "ExportToSpreadsheet:" in text


def test_gui_builder_model_updates_settings_and_order() -> None:
    state = create_default_pipeline_builder_state()

    state = select_pipeline_module(state, 3)
    state = update_pipeline_module_setting(
        state,
        3,
        "Name the primary objects to be identified",
        "Cells",
    )
    assert state.pipeline.modules[3].settings[1].value == "Cells"

    state = move_pipeline_module(state, 3, 2)
    assert state.pipeline.modules[2].name == "IdentifyPrimaryObjects"

    state = remove_pipeline_module(state, 2)
    assert "IdentifyPrimaryObjects" not in [module.name for module in state.pipeline.modules]


def test_generated_pipeline_can_be_passed_to_cellprofiler_runner(tmp_path: Path) -> None:
    pipeline_path = save_cppipe(create_pipeline_from_catalog(), tmp_path / "generated.cppipe")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "sample.tif").write_bytes(b"image")
    completed = MagicMock(returncode=0, stdout="ok", stderr="")

    with (
        patch("bioimage_pipeline.cellprofiler_runner._validate_cellprofiler_executable"),
        patch("bioimage_pipeline.cellprofiler_runner.subprocess.run", return_value=completed) as run,
    ):
        result = run_cellprofiler_pipeline_logged(
            pipeline_path,
            input_dir,
            output_dir,
            cellprofiler_executable="cellprofiler",
        )

    assert result.succeeded
    command = run.call_args.args[0]
    assert str(pipeline_path) in command
    assert str(input_dir) in command
    assert str(output_dir) in command
