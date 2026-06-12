"""Tests for external executable discovery and GUI run-settings persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bioimage_pipeline.cellprofiler_runner import (
    find_cellprofiler_executable,
    find_cellprofiler_gui_executable,
)
from bioimage_pipeline.fiji_runner import find_fiji_executable
from bioimage_pipeline.gui import GuiWorkflowConfig
from bioimage_pipeline.gui.run_settings import (
    CELLPROFILER_SETTINGS_KEY,
    FIJI_SETTINGS_KEY,
    build_cached_run_executables,
    load_gui_run_settings,
    resolve_cellprofiler_executable,
    resolve_fiji_executable,
    save_gui_run_settings,
)
from bioimage_pipeline.gui.workflow_shell import validate_workflow_config


def test_find_cellprofiler_executable_from_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "CellProfiler.exe"
    executable.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("CELLPROFILER_EXECUTABLE", str(executable))

    assert find_cellprofiler_executable() == executable.resolve()


def test_find_cellprofiler_executable_from_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "cellprofiler.cmd"
    executable.write_text("stub", encoding="utf-8")
    monkeypatch.delenv("CELLPROFILER_EXECUTABLE", raising=False)

    def fake_which(name: str) -> str | None:
        if name == "cellprofiler":
            return str(executable)
        return None

    monkeypatch.setattr("bioimage_pipeline.cellprofiler_runner.shutil.which", fake_which)

    assert find_cellprofiler_executable() == executable.resolve()


def test_find_cellprofiler_executable_from_common_windows_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program_files = tmp_path / "Program Files"
    install_dir = program_files / "CellProfiler 4.2"
    executable = install_dir / "CellProfiler.exe"
    install_dir.mkdir(parents=True)
    executable.write_text("stub", encoding="utf-8")

    monkeypatch.delenv("CELLPROFILER_EXECUTABLE", raising=False)
    monkeypatch.setenv("ProgramFiles", str(program_files))
    monkeypatch.setattr("bioimage_pipeline.cellprofiler_runner.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "bioimage_pipeline.cellprofiler_runner.platform.system",
        lambda: "Windows",
    )

    assert find_cellprofiler_executable() == executable.resolve()


def test_find_cellprofiler_gui_executable_honors_explicit_file_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit = tmp_path / "custom" / "CellProfiler.exe"
    explicit.parent.mkdir(parents=True)
    explicit.write_text("gui", encoding="utf-8")
    monkeypatch.delenv("CELLPROFILER_EXECUTABLE", raising=False)

    assert find_cellprofiler_gui_executable(explicit) == explicit.resolve()


def test_find_cellprofiler_executable_explicit_path(tmp_path: Path) -> None:
    executable = tmp_path / "CellProfiler.exe"
    executable.write_text("stub", encoding="utf-8")

    assert find_cellprofiler_executable(executable) == executable.resolve()


def test_resolve_cellprofiler_executable_prefers_saved_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = tmp_path / "saved" / "CellProfiler.exe"
    discovered = tmp_path / "discovered" / "CellProfiler.exe"
    saved.parent.mkdir(parents=True)
    discovered.parent.mkdir(parents=True)
    saved.write_text("saved", encoding="utf-8")
    discovered.write_text("discovered", encoding="utf-8")
    monkeypatch.setenv("CELLPROFILER_EXECUTABLE", str(discovered))

    resolved = resolve_cellprofiler_executable(str(saved))

    assert resolved.source == "saved"
    assert resolved.resolved_path == saved.resolve()
    assert resolved.display_value == str(saved.resolve())


def test_resolve_cellprofiler_executable_ignores_invalid_saved_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovered = tmp_path / "CellProfiler.exe"
    discovered.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("CELLPROFILER_EXECUTABLE", str(discovered))

    resolved = resolve_cellprofiler_executable(str(tmp_path / "missing.exe"))

    assert resolved.source == "discovered"
    assert resolved.resolved_path == discovered.resolve()
    assert any("Ignoring invalid saved CellProfiler" in warning for warning in resolved.warnings)


def test_build_cached_run_executables_restores_saved_settings(tmp_path: Path) -> None:
    saved_cp = tmp_path / "CellProfiler.exe"
    saved_fiji = tmp_path / "ImageJ-win64.exe"
    saved_cp.write_text("cp", encoding="utf-8")
    saved_fiji.write_text("fiji", encoding="utf-8")
    settings_path = tmp_path / "gui_run_settings.json"
    settings_path.write_text(
        json.dumps(
            {
                CELLPROFILER_SETTINGS_KEY: str(saved_cp),
                FIJI_SETTINGS_KEY: str(saved_fiji),
            }
        ),
        encoding="utf-8",
    )

    cached = build_cached_run_executables(settings_path=settings_path)

    assert cached.cellprofiler.resolved_path == saved_cp.resolve()
    assert cached.fiji.resolved_path == saved_fiji.resolve()


def test_save_and_load_gui_run_settings_round_trip(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    save_gui_run_settings(
        {
            CELLPROFILER_SETTINGS_KEY: r"C:\Tools\CellProfiler.exe",
            FIJI_SETTINGS_KEY: r"C:\Tools\ImageJ-win64.exe",
        },
        settings_path=settings_path,
    )

    loaded = load_gui_run_settings(settings_path)

    assert loaded[CELLPROFILER_SETTINGS_KEY] == r"C:\Tools\CellProfiler.exe"
    assert loaded[FIJI_SETTINGS_KEY] == r"C:\Tools\ImageJ-win64.exe"


def test_validate_workflow_config_reports_missing_cellprofiler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "results"
    cppipe = tmp_path / "pipeline.cppipe"
    input_dir.mkdir()
    (input_dir / "sample.tif").write_bytes(b"image")
    cppipe.write_text("pipeline", encoding="utf-8")

    monkeypatch.setattr(
        "bioimage_pipeline.gui.workflow_shell.find_cellprofiler_executable",
        lambda _value: None,
    )

    errors = validate_workflow_config(
        GuiWorkflowConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            cppipe_path=cppipe,
            cellprofiler_executable="cellprofiler",
        )
    )

    assert any("CellProfiler not found" in error for error in errors)


def test_resolve_fiji_executable_prefers_saved_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = tmp_path / "saved-fiji.exe"
    discovered = tmp_path / "discovered-fiji.exe"
    saved.write_text("saved", encoding="utf-8")
    discovered.write_text("discovered", encoding="utf-8")
    monkeypatch.setenv("FIJI_EXECUTABLE", str(discovered))

    resolved = resolve_fiji_executable(str(saved))

    assert resolved.source == "saved"
    assert resolved.resolved_path == saved.resolve()


def test_find_fiji_executable_from_desktop_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    desktop = tmp_path / "Desktop"
    fiji_dir = desktop / "Fiji.app"
    executable = fiji_dir / "ImageJ-win64.exe"
    fiji_dir.mkdir(parents=True)
    executable.write_text("stub", encoding="utf-8")

    monkeypatch.delenv("FIJI_EXECUTABLE", raising=False)
    monkeypatch.setattr("bioimage_pipeline.fiji_runner.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "bioimage_pipeline.fiji_runner.platform.system",
        lambda: "Windows",
    )
    monkeypatch.setattr("bioimage_pipeline.fiji_runner.Path.home", lambda: tmp_path)

    assert find_fiji_executable() == executable.resolve()


def test_sync_discovered_executables_to_settings(tmp_path: Path) -> None:
    from bioimage_pipeline.gui.run_settings import (
        CachedRunExecutables,
        ResolvedExecutable,
        sync_discovered_executables_to_settings,
    )

    settings_path = tmp_path / "settings.json"
    fiji_path = tmp_path / "ImageJ-win64.exe"
    fiji_path.write_text("stub", encoding="utf-8")
    cached = CachedRunExecutables(
        cellprofiler=ResolvedExecutable("cellprofiler", None, "default"),
        fiji=ResolvedExecutable(str(fiji_path), fiji_path.resolve(), "discovered"),
    )

    merged = sync_discovered_executables_to_settings(cached, settings_path=settings_path)

    assert merged[FIJI_SETTINGS_KEY] == str(fiji_path)
    assert load_gui_run_settings(settings_path)[FIJI_SETTINGS_KEY] == str(fiji_path)


def test_find_fiji_executable_still_honors_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "ImageJ-win64.exe"
    executable.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("FIJI_EXECUTABLE", str(executable))

    assert find_fiji_executable() == executable.resolve()
