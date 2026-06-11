"""Tests for Phase 15.3 workflow editor helpers."""

from __future__ import annotations

from pathlib import Path

from bioimage_pipeline.gui import create_default_pipeline_builder_state
from bioimage_pipeline.gui.workflow_editor import (
    EditorSession,
    build_imageset_rows,
    get_images_input_folder,
    groups_wants_grouping,
    list_assigned_image_names,
    list_groups_metadata_categories,
    list_metadata_category_choices,
    list_metadata_keys,
    list_module_output_lines,
    module_settings_label,
    parse_module_notes,
    scan_detected_images,
    set_images_input_folder,
    should_show_imageset,
    should_show_path_list,
    update_groups_metadata_categories,
    window_title,
)
def test_module_settings_label_matches_cellprofiler_format() -> None:
    assert module_settings_label("Images", 1) == "Module settings (Images #01)"
    assert module_settings_label("IdentifyPrimaryObjects", 5) == (
        "Module settings (IdentifyPrimaryObjects #05)"
    )


def test_should_show_path_list_only_for_images_module() -> None:
    assert should_show_path_list("Images") is True
    assert should_show_path_list("Metadata") is False
    assert should_show_path_list(None) is False


def test_should_show_imageset_for_input_modules() -> None:
    assert should_show_imageset("NamesAndTypes") is True
    assert should_show_imageset("IdentifyPrimaryObjects") is False


def test_build_imageset_rows_uses_names_and_types_channels(tmp_path: Path) -> None:
    (tmp_path / "a.tif").write_bytes(b"x")
    state = create_default_pipeline_builder_state()
    columns, rows = build_imageset_rows(state.pipeline, tmp_path)
    assert columns == list_assigned_image_names(state.pipeline)
    assert len(rows) == 1
    assert rows[0][1][columns[0]] == "a.tif"


def test_parse_module_notes_reads_cppipe_header() -> None:
    from bioimage_pipeline.cppipe_io import module_template
    from bioimage_pipeline.pipeline_catalog import get_module_definition

    module = module_template(get_module_definition("Images"))
    module.lines[0] = module.lines[0].replace("notes:[]", "notes:['Test note']")
    assert parse_module_notes(module) == "Test note"


def test_window_title_shows_dirty_marker() -> None:
    assert window_title(path=Path("test.cppipe"), dirty=True).startswith("test.cppipe*")
    assert window_title(path=None, dirty=False).startswith("Untitled pipeline")


def test_editor_session_tracks_dirty_state() -> None:
    session = EditorSession()
    session.mark_saved(Path("a.cppipe"), "line1")
    session.sync_dirty("line1")
    assert session.dirty is False
    session.sync_dirty("line2")
    assert session.dirty is True


def test_images_input_folder_round_trip(tmp_path: Path) -> None:
    state = create_default_pipeline_builder_state()
    updated = set_images_input_folder(state, tmp_path)
    assert get_images_input_folder(updated) == str(tmp_path)


def test_scan_detected_images_finds_tiffs(tmp_path: Path) -> None:
    (tmp_path / "a.tif").write_bytes(b"x")
    (tmp_path / "b.txt").write_text("skip", encoding="utf-8")
    found = scan_detected_images(tmp_path)
    assert [p.name for p in found] == ["a.tif"]


def test_scan_detected_images_finds_nested_oir_files(tmp_path: Path) -> None:
    nested = tmp_path / "plate_a"
    nested.mkdir()
    (nested / "sample.oir").write_bytes(b"oir")
    (tmp_path / "skip.txt").write_text("skip", encoding="utf-8")

    found = scan_detected_images(tmp_path)

    assert [p.name for p in found] == ["sample.oir"]


def test_list_module_output_lines_includes_save_and_export() -> None:
    from bioimage_pipeline.gui import add_named_module_to_pipeline

    state = create_default_pipeline_builder_state()
    state = add_named_module_to_pipeline(state, "SaveImages")
    state = add_named_module_to_pipeline(state, "ExportToSpreadsheet")
    lines = list_module_output_lines(state)
    assert any("SaveImages" in line for line in lines)
    assert any("ExportToSpreadsheet" in line for line in lines)


def test_list_metadata_keys_reads_named_groups_from_metadata_module() -> None:
    from bioimage_pipeline.gui import update_pipeline_module_setting

    state = create_default_pipeline_builder_state()
    state = update_pipeline_module_setting(state, 1, "Extract metadata?", "Yes")
    keys = list_metadata_keys(state.pipeline)
    assert "Well" in keys


def test_groups_metadata_category_choices_default_to_none_without_metadata() -> None:
    state = create_default_pipeline_builder_state()
    assert list_metadata_category_choices(state.pipeline) == ["None"]
    assert groups_wants_grouping(state.pipeline) is False
    assert list_groups_metadata_categories(state.pipeline) == ["None"]


def test_update_groups_metadata_categories_rewrites_cppipe_lines() -> None:
    state = create_default_pipeline_builder_state()
    groups_index = next(
        index for index, module in enumerate(state.pipeline.modules) if module.name == "Groups"
    )
    pipeline = update_groups_metadata_categories(
        state.pipeline, groups_index, ["Plate", "Well"],
    )
    module = pipeline.modules[groups_index]
    assert module.lines.count("    grouping metadata count:2") == 1
    assert list_groups_metadata_categories(pipeline) == ["Plate", "Well"]
