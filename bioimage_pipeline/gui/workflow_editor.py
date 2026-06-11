"""Testable helpers for the Phase 15.3 editor-first workflow shell."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from bioimage_pipeline.cppipe_io import (
    CppipeModule,
    CppipePipeline,
    rewrite_groups_module_settings,
    update_module_setting,
)

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


def module_settings_label(module_name: str, module_num: int) -> str:
    """Format the CellProfiler-style module settings panel title."""
    return f"Module settings ({module_name} #{module_num:02d})"


def should_show_path_list(module_name: str | None) -> bool:
    """Return whether the file list sash should be visible for a module."""
    return module_name == "Images"


IMAGESET_MODULES = frozenset({"Images", "Metadata", "NamesAndTypes", "Groups"})
_MODULE_NOTES_RE = re.compile(r"notes:\[(.*)\]\|batch_state", re.DOTALL)
_NAMED_GROUP_RE = re.compile(r"\(\?P<([^>]+)>")
GROUPS_WANTS_SETTING = "Do you want to group your images?"
GROUPS_METADATA_CATEGORY = "Metadata category"
GROUPS_GROUPING_HELP = (
    "Each unique metadata value (or combination of values) will be defined as a group."
)


def should_show_imageset(module_name: str | None) -> bool:
    """Return whether the image-set panel should be visible for a module."""
    return module_name in IMAGESET_MODULES


def parse_module_notes(module: CppipeModule) -> str:
    """Extract module notes from a ``.cppipe`` module header line."""
    if not module.lines:
        return ""
    header = module.lines[0]
    match = _MODULE_NOTES_RE.search(header)
    if not match:
        return ""
    inner = match.group(1).strip()
    if not inner:
        return ""
    try:
        values = ast.literal_eval(f"[{inner}]")
    except (SyntaxError, ValueError):
        return inner
    if not isinstance(values, list):
        return str(values)
    return "\n".join(str(value) for value in values)


def list_assigned_image_names(pipeline: CppipePipeline) -> list[str]:
    """Return image names configured in NamesAndTypes."""
    index = find_module_index(pipeline, "NamesAndTypes")
    if index is None:
        return ["DNA"]
    names: list[str] = []
    for setting in pipeline.modules[index].settings:
        if setting.key != "Name to assign these images":
            continue
        if setting.value and setting.value not in names:
            names.append(setting.value)
    return names or ["DNA"]


def build_imageset_rows(
    pipeline: CppipePipeline,
    folder: str | Path,
    *,
    limit: int = 50,
) -> tuple[list[str], list[tuple[int, dict[str, str]]]]:
    """Build CellProfiler-style image-set rows from the input folder."""
    columns = list_assigned_image_names(pipeline)
    images = scan_detected_images(folder, limit=limit) if folder else []
    if not images:
        return columns, []
    rows: list[tuple[int, dict[str, str]]] = []
    for cycle, path in enumerate(images, start=1):
        values = {columns[0]: path.name}
        for column in columns[1:]:
            values[column] = ""
        rows.append((cycle, values))
    return columns, rows


def scan_folder_files(folder: str | Path, *, limit: int = 500) -> list[Path]:
    """List all files directly under a folder (for filter preview)."""
    root = Path(folder)
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.iterdir() if path.is_file()),
        key=lambda item: item.name.lower(),
    )[:limit]


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


def list_metadata_keys(pipeline: CppipePipeline) -> list[str]:
    """Return metadata column names configured in the Metadata module."""
    if get_module_setting_value(pipeline, "Metadata", "Extract metadata?") != "Yes":
        return []
    index = find_module_index(pipeline, "Metadata")
    if index is None:
        return []
    keys: list[str] = []
    seen: set[str] = set()
    for setting in pipeline.modules[index].settings:
        if not setting.key.startswith("Regular expression"):
            continue
        for match in _NAMED_GROUP_RE.finditer(setting.value):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                keys.append(name)
    return keys


def list_metadata_category_choices(pipeline: CppipePipeline) -> list[str]:
    """Return metadata category choices for the Groups module."""
    keys = list_metadata_keys(pipeline)
    return keys if keys else ["None"]


def groups_wants_grouping(pipeline: CppipePipeline) -> bool:
    """Return whether the Groups module is configured to group images."""
    return get_module_setting_value(pipeline, "Groups", GROUPS_WANTS_SETTING) == "Yes"


def list_groups_metadata_categories(pipeline: CppipePipeline) -> list[str]:
    """Return metadata categories configured on the Groups module."""
    index = find_module_index(pipeline, "Groups")
    if index is None:
        return ["None"]
    categories = [
        setting.value
        for setting in pipeline.modules[index].settings
        if setting.key == GROUPS_METADATA_CATEGORY
    ]
    return categories or ["None"]


def update_groups_metadata_categories(
    pipeline: CppipePipeline,
    module_index: int,
    categories: list[str],
) -> CppipePipeline:
    """Update the metadata categories used for Groups."""
    if not categories:
        categories = ["None"]
    return rewrite_groups_module_settings(
        pipeline, module_index, metadata_categories=categories,
    )


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
    """List image-like files under a folder (sorted).

    Common raster patterns are scanned non-recursively. ``.oir`` files are also
    discovered recursively so nested Olympus stacks match the batch workflow.
    """
    from bioimage_pipeline.z_projection import iter_oir_files

    root = Path(folder)
    if not root.is_dir():
        return []
    seen: set[str] = set()
    found: list[Path] = []
    for pattern in patterns:
        if pattern == "*.oir":
            continue
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
    for path in iter_oir_files(root):
        key = path.name.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(path)
        if len(found) >= limit:
            return found
    return sorted(found, key=lambda item: item.name.lower())


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
