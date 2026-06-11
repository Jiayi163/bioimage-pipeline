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
    list_modules_by_category,
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
    assert "File Processing" in list_categories()
    assert search_modules("intensity")[0].name in {
        "MeasureObjectIntensity",
        "IdentifyPrimaryObjects",
    }
    assert get_module_definition("ExportToSpreadsheet").category == "File Processing"


def test_catalog_covers_core_cellprofiler_modules() -> None:
    names = {module.name for module in list_modules()}

    expected = {
        "CreateBatchFiles",
        "ColorToGray",
        "CorrectIlluminationCalculate",
        "CorrectIlluminationApply",
        "Crop",
        "EnhanceOrSuppressFeatures",
        "ImageMath",
        "RescaleIntensity",
        "Resize",
        "Threshold",
        "IdentifyPrimaryObjects",
        "IdentifySecondaryObjects",
        "IdentifyTertiaryObjects",
        "FilterObjects",
        "TrackObjects",
        "Watershed",
        "MeasureObjectIntensity",
        "MeasureGranularity",
        "MeasureColocalization",
        "SaveImages",
        "ExportToSpreadsheet",
        "ExportToDatabase",
        "FlagImage",
        "CalculateMath",
        "UntangleWorms",
    }
    missing = expected - names
    assert not missing, f"Missing catalog modules: {sorted(missing)}"


def test_catalog_grouping_excludes_input_and_is_ordered() -> None:
    grouped = list_modules_by_category()
    categories = [category for category, _ in grouped]

    assert "Input" not in categories
    assert categories[0] == "Image Processing"
    assert {"Object Processing", "Measurement", "File Processing", "Advanced"} <= set(categories)

    for _, modules in grouped:
        sorted_names = [module.name.lower() for module in modules]
        assert sorted_names == sorted(sorted_names)

    create_batch = get_module_definition("CreateBatchFiles")
    assert create_batch.category == "File Processing"


def test_phase_15_2_catalog_includes_setting_metadata() -> None:
    module = get_module_definition("SaveImages")
    parameters = {parameter.label: parameter for parameter in module.parameters}

    assert module.display_name == "SaveImages"
    assert module.category == "File Processing"
    assert parameters["Select the type of image to save"].choices == (
        "Image",
        "Mask",
        "Objects",
    )
    assert parameters["Saved file format"].default == "tiff"
    assert parameters["Enter single file name"].visibility.mode == "conditional"


def test_images_input_folder_is_managed_in_file_list_not_module_settings() -> None:
    images = get_module_definition("Images")
    visible_labels = [parameter.label for parameter in images.visible_parameters()]
    assert "Input folder path" not in visible_labels
    assert "Filter images?" in visible_labels


def test_save_images_defaults_export_segmentation_masks() -> None:
    module = get_module_definition("SaveImages")
    parameters = {parameter.label: parameter.default for parameter in module.parameters}

    assert parameters["Select the type of image to save"] == "Mask"
    assert parameters["Select the image to save"] == "Nuclei"
    assert parameters["Enter file prefix"] == "Nuclei_mask"
    assert parameters["Select method for constructing file names"] == "From image filename"
    assert parameters["Image bit depth"] == "8-bit integer"


def test_prepare_pipeline_save_images_uses_image_filename_not_sequential() -> None:
    from bioimage_pipeline.cppipe_io import prepare_pipeline_for_cellprofiler

    pipeline = create_pipeline_from_catalog()
    pipeline = append_module(pipeline, "IdentifyPrimaryObjects")
    pipeline = append_module(pipeline, "SaveImages")

    prepared = prepare_pipeline_for_cellprofiler(pipeline)
    save_images = next(module for module in prepared.modules if module.name == "SaveImages")
    settings = {setting.key: setting.value for setting in save_images.settings}

    assert settings["Select method for constructing file names"] == "From image filename"
    assert settings["Select image name for file prefix"] == "DNA"
    assert "Select object to save" not in settings
    assert prepared.to_text().count("Sequential numbers") == 0


def test_update_module_setting_appends_indented_lines() -> None:
    pipeline = create_pipeline_from_catalog()
    images_index = next(
        index for index, module in enumerate(pipeline.modules) if module.name == "Images"
    )
    updated = update_module_setting(
        pipeline,
        images_index,
        "Input folder path",
        r"C:\data\images",
    )
    images_module = updated.modules[images_index]
    assert any(
        line == r"    Input folder path:C:\data\images"
        for line in images_module.lines
    )
    assert "Input folder path:C:\\data\\images" not in images_module.lines


def test_prepare_pipeline_adds_export_to_spreadsheet_for_analysis_modules() -> None:
    from bioimage_pipeline.cppipe_io import prepare_pipeline_for_cellprofiler

    pipeline = create_pipeline_from_catalog()
    pipeline = append_module(pipeline, "IdentifyPrimaryObjects")
    pipeline = append_module(pipeline, "SaveImages")

    prepared = prepare_pipeline_for_cellprofiler(pipeline)
    assert "ExportToSpreadsheet" in [module.name for module in prepared.modules]


def test_prepare_pipeline_for_cellprofiler_strips_gui_only_settings() -> None:
    from bioimage_pipeline.cppipe_io import prepare_pipeline_for_cellprofiler

    corrupted = parse_cppipe_text(
        """CellProfiler Pipeline: http://www.cellprofiler.org
Version:5
ModuleCount:2
HasImagePlaneDetails:False

Images:[module_num:1|svn_version:'Unknown'|variable_revision_number:2|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    :
    Filter images?:Images only
Input folder path:C:\\data\\images
Metadata:[module_num:2|svn_version:'Unknown'|variable_revision_number:6|show_window:False|notes:[]|batch_state:array([], dtype=uint8)|enabled:True|wants_pause:False]
    Extract metadata?:No
"""
    )

    prepared = prepare_pipeline_for_cellprofiler(corrupted)
    prepared_text = prepared.to_text()
    assert "Input folder path" not in prepared_text
    assert "Metadata:" in prepared_text
    assert "Filter images?:Images only" in prepared_text


def test_normalize_save_images_rewrites_image_exports() -> None:
    from bioimage_pipeline.cppipe_io import (
        append_module,
        create_pipeline_from_catalog,
        normalize_save_images_in_pipeline,
        save_images_needs_normalization,
    )

    pipeline = create_pipeline_from_catalog()
    pipeline = append_module(pipeline, "IdentifyPrimaryObjects")
    pipeline = append_module(pipeline, "SaveImages")

    save_index = next(
        index for index, module in enumerate(pipeline.modules) if module.name == "SaveImages"
    )
    settings = {
        setting.key: setting.value
        for setting in pipeline.modules[save_index].settings
    }
    assert save_images_needs_normalization(settings) is False

    legacy = create_pipeline_from_catalog()
    legacy = append_module(legacy, "IdentifyPrimaryObjects")
    legacy = append_module(legacy, get_module_definition("SaveImages"))
    legacy_index = next(
        index for index, module in enumerate(legacy.modules) if module.name == "SaveImages"
    )
    legacy = update_module_setting(
        legacy,
        legacy_index,
        "Select the type of image to save",
        "Image",
    )
    legacy = update_module_setting(legacy, legacy_index, "Select the image to save", "DNA")
    legacy = update_module_setting(legacy, legacy_index, "Enter file prefix", "Nuclei")

    legacy_settings = {
        setting.key: setting.value for setting in legacy.modules[legacy_index].settings
    }
    assert save_images_needs_normalization(legacy_settings) is True

    normalized = normalize_save_images_in_pipeline(legacy)
    normalized_index = next(
        index for index, module in enumerate(normalized.modules) if module.name == "SaveImages"
    )
    normalized_settings = {
        setting.key: setting.value
        for setting in normalized.modules[normalized_index].settings
    }
    assert normalized_settings["Select the type of image to save"] == "Mask"
    assert normalized_settings["Select the image to save"] == "Nuclei"
    assert normalized_settings["Enter file prefix"] == "Nuclei_mask"


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


def test_names_and_types_visible_settings_show_one_assignment_block() -> None:
    names_and_types = get_module_definition("NamesAndTypes")
    visible_labels = [
        parameter.label for parameter in names_and_types.visible_parameters()
    ]

    assert visible_labels.count("Name to assign these images") == 1
    assert visible_labels.count("Select the image type") == 1
    assert visible_labels.count("Set intensity range from") == 1
    assert visible_labels.count("Maximum intensity") == 0
    assert "Name to assign these objects" not in visible_labels
    assert visible_labels[:4] == [
        "Assign a name to",
        "Select the image type",
        "Name to assign these images",
        "Image set matching method",
    ]


def test_names_and_types_visible_settings_follow_manual_intensity_choice() -> None:
    names_and_types = get_module_definition("NamesAndTypes")
    manual_labels = [
        parameter.label
        for parameter in names_and_types.visible_parameters(
            {"Set intensity range from": "Manual"}
        )
    ]

    assert "Maximum intensity" in manual_labels
    assert manual_labels.count("Maximum intensity") == 1


def test_names_and_types_pixel_spacing_hidden_unless_process_as_3d() -> None:
    names_and_types = get_module_definition("NamesAndTypes")
    default_labels = [
        parameter.label for parameter in names_and_types.visible_parameters()
    ]
    three_d_labels = [
        parameter.label
        for parameter in names_and_types.visible_parameters({"Process as 3D?": "Yes"})
    ]

    for axis in ("X", "Y", "Z"):
        label = f"Relative pixel spacing in {axis}"
        assert label not in default_labels
        assert label in three_d_labels


def test_groups_module_matches_cellprofiler_v2_structure() -> None:
    groups = get_module_definition("Groups")
    pipeline = create_pipeline_from_catalog()
    groups_module = next(module for module in pipeline.modules if module.name == "Groups")
    text = pipeline.to_text()

    assert groups.variable_revision_number == 2
    assert groups_module.lines.count("    Do you want to group your images?:No") == 1
    assert groups_module.lines.count("    grouping metadata count:1") == 1
    assert groups_module.lines.count("    Metadata category:None") == 1
    assert "variable_revision_number:2" in text


def test_groups_visible_settings_follow_grouping_choice() -> None:
    groups = get_module_definition("Groups")
    default_labels = [parameter.label for parameter in groups.visible_parameters()]
    grouping_labels = [
        parameter.label
        for parameter in groups.visible_parameters({"Do you want to group your images?": "Yes"})
    ]

    assert default_labels == ["Do you want to group your images?"]
    assert grouping_labels == ["Do you want to group your images?"]
    assert "Metadata category" not in default_labels


def test_rewrite_groups_module_settings_supports_multiple_metadata_categories() -> None:
    from bioimage_pipeline.cppipe_io import rewrite_groups_module_settings

    pipeline = create_pipeline_from_catalog()
    groups_index = next(
        index for index, module in enumerate(pipeline.modules) if module.name == "Groups"
    )

    updated = rewrite_groups_module_settings(
        pipeline,
        groups_index,
        wants_groups="Yes",
        metadata_categories=["Plate", "Well"],
    )
    module = updated.modules[groups_index]
    text = updated.to_text()

    assert module.lines.count("    Do you want to group your images?:Yes") == 1
    assert module.lines.count("    grouping metadata count:2") == 1
    assert module.lines.count("    Metadata category:Plate") == 1
    assert module.lines.count("    Metadata category:Well") == 1
    assert "Metadata category:None" not in text


def test_names_and_types_cppipe_has_single_assignment_block() -> None:
    pipeline = create_pipeline_from_catalog()
    names_module = next(
        module for module in pipeline.modules if module.name == "NamesAndTypes"
    )
    text = pipeline.to_text()

    assert names_module.lines.count("    Name to assign these images:DNA") == 1
    assert "    Name to assign these objects:Cell" not in text
    assert names_module.lines.count("    Select the image type:Grayscale image") == 1
    assert names_module.lines.count("    Set intensity range from:Image metadata") == 1
    assert names_module.lines.count("    Maximum intensity:255.0") == 0
    assert "NamesAndTypes:" in text


def test_update_module_setting_updates_single_names_and_types_assignment() -> None:
    pipeline = create_pipeline_from_catalog()
    names_index = next(
        index
        for index, module in enumerate(pipeline.modules)
        if module.name == "NamesAndTypes"
    )

    updated = update_module_setting(
        pipeline, names_index, "Name to assign these images", "DAPI",
    )
    module = updated.modules[names_index]

    assert module.lines.count("    Name to assign these images:DAPI") == 1
    assert module.lines.count("    Name to assign these images:DNA") == 0


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


def test_generated_default_pipeline_contains_only_required_setup_modules() -> None:
    pipeline = create_pipeline_from_catalog()
    text = pipeline.to_text()

    assert [module.name for module in pipeline.modules] == list(MINIMAL_GUI_PIPELINE_MODULES)
    assert [module.name for module in pipeline.modules] == [
        "Images",
        "Metadata",
        "NamesAndTypes",
        "Groups",
    ]
    for index, module_name in enumerate(MINIMAL_GUI_PIPELINE_MODULES, start=1):
        assert f"{module_name}:[module_num:{index}|" in text
    assert "IdentifyPrimaryObjects:" not in text
    assert "ExportToSpreadsheet:" not in text
    assert "SaveImages:" not in text


def test_gui_builder_model_updates_settings_and_order() -> None:
    state = create_default_pipeline_builder_state()

    assert [module.name for module in state.pipeline.modules] == [
        "Images",
        "Metadata",
        "NamesAndTypes",
        "Groups",
    ]

    state = select_pipeline_module(state, 1)
    state = update_pipeline_module_setting(state, 1, "Extract metadata?", "Yes")
    assert any(
        "Extract metadata?:Yes" in line for line in state.pipeline.modules[1].lines
    )

    state = move_pipeline_module(state, 3, 1)
    assert state.pipeline.modules[1].name == "Groups"

    state = remove_pipeline_module(state, 1)
    assert "Groups" not in [module.name for module in state.pipeline.modules]


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
