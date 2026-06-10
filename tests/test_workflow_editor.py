"""Tests for Phase 15.3 workflow editor helpers."""

from __future__ import annotations

from pathlib import Path

from bioimage_pipeline.gui.workflow_editor import (
    EditorSession,
    get_images_input_folder,
    list_module_output_lines,
    scan_detected_images,
    set_images_input_folder,
    window_title,
)
from bioimage_pipeline.gui import create_default_pipeline_builder_state


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
