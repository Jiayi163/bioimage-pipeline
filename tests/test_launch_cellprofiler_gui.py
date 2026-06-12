"""Tests for CellProfiler GUI launch helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bioimage_pipeline.cellprofiler_runner import (
    build_cellprofiler_gui_command,
    find_cellprofiler_gui_executable,
    launch_cellprofiler_gui_process,
)
from bioimage_pipeline.gui.workflow_editor import launch_cellprofiler_gui


def test_find_cellprofiler_gui_executable_prefers_install_over_path_shim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program_files = tmp_path / "Program Files"
    install_dir = program_files / "CellProfiler"
    installed = install_dir / "CellProfiler.exe"
    install_dir.mkdir(parents=True)
    installed.write_text("gui", encoding="utf-8")

    shim = tmp_path / "venv" / "Scripts" / "cellprofiler.exe"
    shim.parent.mkdir(parents=True)
    shim.write_text("shim", encoding="utf-8")

    monkeypatch.delenv("CELLPROFILER_EXECUTABLE", raising=False)

    def fake_which(name: str) -> str | None:
        if name == "cellprofiler":
            return str(shim)
        return None

    monkeypatch.setattr(
        "bioimage_pipeline.cellprofiler_runner.shutil.which",
        fake_which,
    )
    monkeypatch.setenv("ProgramFiles", str(program_files))
    monkeypatch.setattr(
        "bioimage_pipeline.cellprofiler_runner.platform.system",
        lambda: "Windows",
    )

    assert find_cellprofiler_gui_executable("cellprofiler") == installed.resolve()


def test_build_cellprofiler_gui_command_resolves_pipeline_and_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "CellProfiler.exe"
    executable.write_text("gui", encoding="utf-8")
    pipeline = tmp_path / "pipeline.cppipe"
    pipeline.write_text("pipeline", encoding="utf-8")

    monkeypatch.setattr(
        "bioimage_pipeline.cellprofiler_runner.find_cellprofiler_gui_executable",
        lambda _explicit=None: executable.resolve(),
    )

    command = build_cellprofiler_gui_command(pipeline, cellprofiler_executable="cellprofiler")

    assert command == [str(executable.resolve()), str(pipeline.resolve())]


def test_launch_cellprofiler_gui_process_debounces_duplicate_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "CellProfiler.exe"
    pipeline = tmp_path / "pipeline.cppipe"
    executable.write_text("gui", encoding="utf-8")
    pipeline.write_text("pipeline", encoding="utf-8")
    command = [str(executable), str(pipeline)]

    shell_open = MagicMock()
    monkeypatch.setattr(
        "bioimage_pipeline.cellprofiler_runner.platform.system",
        lambda: "Windows",
    )
    monkeypatch.setattr(
        "bioimage_pipeline.cellprofiler_runner._windows_shell_open_executable",
        shell_open,
    )
    monkeypatch.setattr(
        "bioimage_pipeline.cellprofiler_runner._LAST_CELLPROFILER_GUI_LAUNCH",
        None,
    )
    monkeypatch.setattr(
        "bioimage_pipeline.cellprofiler_runner._LAST_CELLPROFILER_GUI_LAUNCH_MONOTONIC",
        0.0,
    )

    launch_cellprofiler_gui_process(command, debounce_seconds=30.0)
    launch_cellprofiler_gui_process(command, debounce_seconds=30.0)

    shell_open.assert_called_once()


def test_launch_cellprofiler_gui_uses_shell_execute_on_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "CellProfiler.exe"
    executable.write_text("gui", encoding="utf-8")
    pipeline = tmp_path / "pipeline.cppipe"
    pipeline.write_text("pipeline", encoding="utf-8")

    monkeypatch.setattr(
        "bioimage_pipeline.gui.workflow_editor.build_cellprofiler_gui_command",
        lambda *_args, **_kwargs: [str(executable), str(pipeline)],
    )
    process = MagicMock()
    monkeypatch.setattr(
        "bioimage_pipeline.gui.workflow_editor.launch_cellprofiler_gui_process",
        process,
    )

    launch_cellprofiler_gui(pipeline, cellprofiler_executable=str(executable))

    process.assert_called_once_with([str(executable), str(pipeline)])
