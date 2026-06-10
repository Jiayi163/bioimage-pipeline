"""Testable helpers for the Phase 15.3 editor-first workflow shell."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from bioimage_pipeline.cppipe_io import CppipePipeline, update_module_setting

if TYPE_CHECKING:
    from bioimage_pipeline.gui.workflow_shell import PipelineBuilderState

IMAGES_INPUT_FOLDER_KEY = "Input folder path"
OUTPUT_MODULES = frozenset({"SaveImages", "ExportToSpreadsheet"})
IMAGE_GLOB_PATTERNS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.oir", "*.czi")


@dataclass
class EditorSession:
    """Tracks the open pipeline file and modification state."""

    path: Path | None = None
    dirty: bool = False
    baseline_text: str = ""

    def mark_saved(self, path: Path, pipeline_text: str) -> None:
        self.path = path
        self.baseline_text = pipeline_text
        self.dirty = False

    def mark_dirty(self) -> None:
        self.dirty = True

    def sync_dirty(self, pipeline_text: str) -> None:
        self.dirty = pipeline_text != self.baseline_text


def window_title(*, base: str = "Bioimage Pipeline", path: Path | None, dirty: bool) -> str:
    """Format the main window title with optional path and modified marker."""
    if path is None:
        name = "Untitled pipeline"
    else:
        name = path.name
    prefix = f"{name}*" if dirty else name
    return f"{prefix} — {base}"


def find_module_index(pipeline: CppipePipeline, module_name: str) -> int | None:
    for index, module in enumerate(pipeline.modules):
        if module.name == module_name:
            return index
    return None


def get_module_setting_value(pipeline: CppipePipeline, module_name: str, key: str) -> str | None:
    index = find_module_index(pipeline, module_name)
    if index is None:
        return None
    for setting in pipeline.modules[index].settings:
        if setting.key == key:
            return setting.value
    return None


def get_images_input_folder(state: PipelineBuilderState) -> str:
    """Return the Images-module input folder path, if configured."""
    value = get_module_setting_value(state.pipeline, "Images", IMAGES_INPUT_FOLDER_KEY)
    return (value or "").strip()


def set_images_input_folder(state: PipelineBuilderState, folder: str | Path) -> PipelineBuilderState:
    """Store the input folder on the Images module."""
    from bioimage_pipeline.gui.workflow_shell import PipelineBuilderState as BuilderState

    index = find_module_index(state.pipeline, "Images")
    if index is None:
        raise ValueError("Pipeline has no Images module.")
    pipeline = update_module_setting(
        state.pipeline,
        index,
        IMAGES_INPUT_FOLDER_KEY,
        str(Path(folder)),
    )
    return BuilderState(
        pipeline=pipeline,
        catalog_modules=list(state.catalog_modules),
        selected_module_index=state.selected_module_index,
    )


def scan_detected_images(
    folder: str | Path,
    *,
    patterns: Iterable[str] = IMAGE_GLOB_PATTERNS,
    limit: int = 500,
) -> list[Path]:
    """List image-like files under a folder (non-recursive, sorted)."""
    root = Path(folder)
    if not root.is_dir():
        return []
    seen: set[str] = set()
    found: list[Path] = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            key = path.name.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
            if len(found) >= limit:
                return found
    return found


def resolve_workflow_input_dir(state: PipelineBuilderState) -> Path:
    """Resolve the run input directory from the Images module."""
    folder = get_images_input_folder(state)
    if not folder:
        raise ValueError("Select an input folder in the Images module before running.")
    path = Path(folder)
    if not path.is_dir():
        raise ValueError(f"Images input folder does not exist: {path}")
    if not scan_detected_images(path, limit=1):
        raise ValueError(f"No image files detected in: {path}")
    return path


def list_module_output_lines(state: PipelineBuilderState) -> list[str]:
    """Summarize SaveImages / ExportToSpreadsheet output settings in the pipeline."""
    lines: list[str] = []
    for module in state.pipeline.modules:
        if module.name not in OUTPUT_MODULES:
            continue
        values = {setting.key: setting.value for setting in module.settings}
        if module.name == "SaveImages":
            location = values.get("Output file location", "Default Output Folder|None")
            fmt = values.get("Saved file format", "tiff")
            lines.append(f"SaveImages → {location} ({fmt})")
        elif module.name == "ExportToSpreadsheet":
            location = values.get("Output file location", "Default Output Folder|.")
            prefix = values.get("Filename prefix", "MyExpt_")
            lines.append(f"ExportToSpreadsheet → {location} (prefix: {prefix})")
    return lines


def launch_cellprofiler_gui(
    cppipe_path: str | Path,
    *,
    cellprofiler_executable: str = "cellprofiler",
) -> None:
    """Open a pipeline in the native CellProfiler desktop application."""
    path = Path(cppipe_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline file not found: {path}")
    command = [cellprofiler_executable, str(path)]
    if sys.platform.startswith("win"):
        subprocess.Popen(command, creationflags=subprocess.DETACHED_PROCESS)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(command, start_new_session=True)
