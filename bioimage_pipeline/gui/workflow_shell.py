"""Phase 15.3 GUI workflow shell.

Editor-first pipeline workspace with optional launch of native CellProfiler for
advanced authoring. Headless CellProfiler/Fiji execution uses
:func:`bioimage_pipeline.analysis.run_cellprofiler_workflow`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from bioimage_pipeline.analysis import CellProfilerWorkflowResult
from bioimage_pipeline.cellprofiler_runner import find_cellprofiler_executable
from bioimage_pipeline.fiji_runner import find_fiji_executable, fiji_not_found_message
from bioimage_pipeline.gui.run_settings import (
    build_cached_run_executables,
    collect_run_settings_from_values,
    save_gui_run_settings,
    sync_discovered_executables_to_settings,
)
from bioimage_pipeline.z_projection import (
    PYTHON_OIR_MISSING_DEPS_MESSAGE,
    iter_oir_files,
    python_oir_dependencies_available,
)
from bioimage_pipeline.cppipe_io import (
    REQUIRED_SETUP_MODULES,
    CppipePipeline,
    append_module,
    create_pipeline_from_catalog,
    load_cppipe,
    move_module,
    remove_module,
    save_cppipe,
    summarize_modules,
    update_module_setting,
    validate_cppipe,
)
from bioimage_pipeline.pipeline_catalog import (
    ModuleDefinition,
    list_modules,
    search_modules,
)
from bioimage_pipeline.gui.add_module_dialog import open_add_module_dialog
from bioimage_pipeline.gui.workflow_editor import (
    EditorSession,
    GROUPS_GROUPING_HELP,
    GROUPS_METADATA_CATEGORY,
    build_imageset_rows,
    get_images_input_folder,
    groups_wants_grouping,
    launch_cellprofiler_gui,
    list_groups_metadata_categories,
    list_metadata_category_choices,
    module_settings_label,
    parse_module_notes,
    resolve_workflow_input_dir,
    scan_detected_images,
    scan_folder_files,
    set_images_input_folder,
    should_show_imageset,
    should_show_path_list,
    update_groups_metadata_categories,
    window_title,
)


@dataclass
class GuiWorkflowConfig:
    """User-selected workflow inputs for the GUI shell."""

    input_dir: str | Path
    output_dir: str | Path
    cppipe_path: str | Path
    cellprofiler_executable: str = "cellprofiler"
    fiji_executable: str | Path | None = None
    fiji_macro_path: str | Path | None = None
    export_fiji_tiffs: bool = True
    generate_qc: bool = True
    oir_projection_engine: str = "python"


@dataclass
class GuiWorkflowSummary:
    """Small display model for workflow results."""

    results_dir: Path
    processed_count: int
    measurements_dir: Path
    masks_dir: Path
    labels_dir: Path
    qc_dir: Path
    logs_dir: Path
    measurement_files: list[Path]
    qc_preview_files: list[Path]
    mask_files: list[Path]
    label_files: list[Path]
    timing: dict[str, float]
    export_engine: str | None
    export_mode: str | None
    warnings: list[str]

    def to_display_lines(self) -> list[str]:
        """Format the summary for a plain-text status panel."""
        lines = [
            f"Results: {self.results_dir}",
            f"Processed images: {self.processed_count}",
            f"Measurements: {self.measurements_dir} ({len(self.measurement_files)} file(s))",
            f"Masks: {self.masks_dir} ({len(self.mask_files)} file(s))",
            f"Labels: {self.labels_dir} ({len(self.label_files)} file(s))",
            f"QC: {self.qc_dir} ({len(self.qc_preview_files)} preview file(s))",
            f"Logs: {self.logs_dir}",
        ]
        if self.export_engine:
            mode = f" / {self.export_mode}" if self.export_mode else ""
            lines.append(f"Export: {self.export_engine}{mode}")
        if self.timing:
            for key, seconds in sorted(self.timing.items()):
                lines.append(f"{key}: {seconds:.2f}s")
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        return lines


@dataclass
class PipelineBuilderState:
    """Display state for the Phase 15.2 pipeline builder panel."""

    pipeline: CppipePipeline
    catalog_modules: list[ModuleDefinition]
    selected_module_index: int | None = None

    @property
    def module_rows(self) -> list[str]:
        return summarize_modules(self.pipeline.modules)

    @property
    def catalog_rows(self) -> list[str]:
        return [
            f"{module.name} ({module.category})"
            for module in self.catalog_modules
        ]

    @property
    def selected_module(self):
        """Return the selected pipeline module, if any."""
        if self.selected_module_index is None:
            return None
        if not 0 <= self.selected_module_index < len(self.pipeline.modules):
            return None
        return self.pipeline.modules[self.selected_module_index]

    @property
    def selected_definition(self) -> ModuleDefinition | None:
        """Return catalog metadata for the selected module, if known."""
        selected = self.selected_module
        if selected is None:
            return None
        try:
            from bioimage_pipeline.pipeline_catalog import get_module_definition

            return get_module_definition(selected.name)
        except KeyError:
            return None

    @property
    def selected_setting_values(self) -> dict[str, str]:
        """Return current setting values for the selected module."""
        selected = self.selected_module
        if selected is None:
            return {}
        return {setting.key: setting.value for setting in selected.settings}

    @property
    def selected_visible_parameters(self):
        """Return visible catalog settings for the selected module."""
        definition = self.selected_definition
        if definition is None:
            return []
        return definition.visible_parameters(self.selected_setting_values)


def load_pipeline_builder_state(
    cppipe_path: str | Path,
    *,
    query: str = "",
) -> PipelineBuilderState:
    """Load a pipeline and catalog rows for the GUI builder."""
    catalog_modules = search_modules(query) if query else list_modules()
    return PipelineBuilderState(
        pipeline=load_cppipe(cppipe_path),
        catalog_modules=catalog_modules,
        selected_module_index=0,
    )


def create_default_pipeline_builder_state(*, query: str = "") -> PipelineBuilderState:
    """Create the Phase 15.2 minimal GUI pipeline."""
    catalog_modules = search_modules(query) if query else list_modules()
    pipeline = create_pipeline_from_catalog()
    return PipelineBuilderState(
        pipeline=pipeline,
        catalog_modules=catalog_modules,
        selected_module_index=0 if pipeline.modules else None,
    )


def is_protected_module_name(name: str) -> bool:
    """Return whether a module is a required setup module that cannot be removed."""
    return name in REQUIRED_SETUP_MODULES


def leading_protected_count(modules: list[Any]) -> int:
    """Count the consecutive required setup modules at the top of a pipeline."""
    count = 0
    for module in modules:
        if is_protected_module_name(module.name):
            count += 1
        else:
            break
    return count


def split_pipeline_rows(modules: list[Any]) -> tuple[list[str], list[str], int]:
    """Split modules into CellProfiler-style display rows.

    Returns ``(input_rows, analysis_rows, protected_count)`` where the leading
    protected setup modules are shown unnumbered (like CellProfiler's input
    modules) and the remaining analysis modules are numbered from ``01``.
    """
    protected = leading_protected_count(modules)
    input_rows = [module.name for module in modules[:protected]]
    analysis_rows = [
        f"{index:02d} {module.name}"
        for index, module in enumerate(modules[protected:], start=1)
    ]
    return input_rows, analysis_rows, protected


def add_catalog_module_to_pipeline(
    state: PipelineBuilderState,
    catalog_index: int,
) -> PipelineBuilderState:
    """Append a selected catalog module to a builder pipeline."""
    module = state.catalog_modules[catalog_index]
    pipeline = append_module(state.pipeline, module)
    return PipelineBuilderState(
        pipeline=pipeline,
        catalog_modules=list(state.catalog_modules),
        selected_module_index=len(pipeline.modules) - 1,
    )


def add_named_module_to_pipeline(
    state: PipelineBuilderState,
    module_name: str,
) -> PipelineBuilderState:
    """Append a catalog module by name to a builder pipeline."""
    pipeline = append_module(state.pipeline, module_name)
    return PipelineBuilderState(
        pipeline=pipeline,
        catalog_modules=list(state.catalog_modules),
        selected_module_index=len(pipeline.modules) - 1,
    )


def remove_pipeline_module(
    state: PipelineBuilderState,
    module_index: int,
) -> PipelineBuilderState:
    """Remove a module from a builder pipeline."""
    pipeline = remove_module(state.pipeline, module_index)
    selected_index: int | None
    if not pipeline.modules:
        selected_index = None
    else:
        selected_index = min(module_index, len(pipeline.modules) - 1)
    return PipelineBuilderState(
        pipeline=pipeline,
        catalog_modules=list(state.catalog_modules),
        selected_module_index=selected_index,
    )


def move_pipeline_module(
    state: PipelineBuilderState,
    from_index: int,
    to_index: int,
) -> PipelineBuilderState:
    """Move a module within the builder pipeline."""
    bounded_to_index = max(0, min(to_index, len(state.pipeline.modules) - 1))
    pipeline = move_module(state.pipeline, from_index, bounded_to_index)
    return PipelineBuilderState(
        pipeline=pipeline,
        catalog_modules=list(state.catalog_modules),
        selected_module_index=bounded_to_index,
    )


def select_pipeline_module(
    state: PipelineBuilderState,
    module_index: int | None,
) -> PipelineBuilderState:
    """Select a pipeline module for settings display."""
    if module_index is not None and not 0 <= module_index < len(state.pipeline.modules):
        module_index = None
    return PipelineBuilderState(
        pipeline=state.pipeline,
        catalog_modules=list(state.catalog_modules),
        selected_module_index=module_index,
    )


def update_pipeline_module_setting(
    state: PipelineBuilderState,
    module_index: int,
    setting_label: str,
    value: str,
) -> PipelineBuilderState:
    """Update one setting in the internal pipeline model."""
    pipeline = update_module_setting(state.pipeline, module_index, setting_label, value)
    return PipelineBuilderState(
        pipeline=pipeline,
        catalog_modules=list(state.catalog_modules),
        selected_module_index=module_index,
    )


def save_pipeline_builder_state(state: PipelineBuilderState, path: str | Path) -> Path:
    """Validate and save the builder pipeline."""
    from bioimage_pipeline.cppipe_io import prepare_pipeline_for_cellprofiler

    pipeline = prepare_pipeline_for_cellprofiler(state.pipeline)
    errors = validate_cppipe(pipeline)
    if errors:
        raise ValueError("\n".join(errors))
    return save_cppipe(pipeline, path)


def default_oir_projection_engine_choice(
    *,
    fiji_executable: str | Path | None = None,
) -> str:
    """Return the GUI default OIR projection engine for this machine."""
    if python_oir_dependencies_available():
        return "python"
    if find_fiji_executable(fiji_executable) is not None:
        return "fiji"
    return "python"


def validate_workflow_config(config: GuiWorkflowConfig) -> list[str]:
    """Return validation errors for a GUI workflow config."""
    errors: list[str] = []
    if not str(config.input_dir).strip():
        errors.append("Set an input folder in the Images module before running.")
    if not str(config.output_dir).strip():
        errors.append("Output folder is required.")

    input_dir = Path(config.input_dir) if str(config.input_dir).strip() else None
    output_dir = Path(config.output_dir) if str(config.output_dir).strip() else None

    if input_dir is not None:
        if not input_dir.is_dir():
            errors.append(f"Images input folder does not exist: {input_dir}")
        elif not scan_detected_images(input_dir, limit=1):
            errors.append(f"No image files detected in: {input_dir}")
    if output_dir is not None and output_dir.exists() and not output_dir.is_dir():
        errors.append(f"Output path exists but is not a folder: {output_dir}")

    cppipe_path = str(config.cppipe_path).strip()
    if cppipe_path:
        path = Path(cppipe_path)
        if not path.is_file():
            errors.append(f"CellProfiler pipeline file does not exist: {path}")

    if not str(config.cellprofiler_executable).strip():
        errors.append("CellProfiler executable is required.")
    elif find_cellprofiler_executable(config.cellprofiler_executable) is None:
        errors.append("CellProfiler not found. Please select executable.")
    if config.fiji_executable:
        if find_fiji_executable(config.fiji_executable) is None:
            errors.append(f"Fiji executable does not exist: {config.fiji_executable}")
    if config.fiji_macro_path and not Path(config.fiji_macro_path).is_file():
        errors.append(f"Fiji macro does not exist: {config.fiji_macro_path}")

    if input_dir is not None and list(iter_oir_files(input_dir)):
        engine = (config.oir_projection_engine or "python").strip().lower()
        if engine not in {"python", "fiji"}:
            errors.append(
                "OIR projection engine must be 'python' or 'fiji'."
            )
        elif engine == "python" and not python_oir_dependencies_available():
            errors.append(PYTHON_OIR_MISSING_DEPS_MESSAGE)
        elif engine == "fiji" and find_fiji_executable(config.fiji_executable) is None:
            errors.append(fiji_not_found_message())
    return errors


def build_workflow_summary(result: CellProfilerWorkflowResult) -> GuiWorkflowSummary:
    """Build a GUI display summary from a workflow result."""
    measurement_files = sorted(result.measurements_dir.glob("*.csv"))
    qc_preview_files = sorted(result.qc_dir.glob("*.png"))
    mask_files = sorted(result.masks_dir.glob("*.tif"))
    label_files = sorted(result.labels_dir.glob("*.tif"))
    warnings = [
        *list(result.import_warnings or []),
        *list(result.export_warnings or []),
    ]
    return GuiWorkflowSummary(
        results_dir=result.results_dir,
        processed_count=len(result.processed_images),
        measurements_dir=result.measurements_dir,
        masks_dir=result.masks_dir,
        labels_dir=result.labels_dir,
        qc_dir=result.qc_dir,
        logs_dir=result.logs_dir,
        measurement_files=measurement_files,
        qc_preview_files=qc_preview_files,
        mask_files=mask_files,
        label_files=label_files,
        timing=dict(result.timing or {}),
        export_engine=result.export_engine,
        export_mode=result.export_mode,
        warnings=warnings,
    )


def read_log_tail(path: str | Path, *, max_lines: int = 80) -> str:
    """Read the last lines of a log file for display."""
    log_path = Path(path)
    if not log_path.is_file():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def load_measurements_preview(path: str | Path, *, max_rows: int = 20) -> pd.DataFrame:
    """Load a small CSV preview for the GUI measurements panel."""
    return pd.read_csv(path, nrows=max_rows)


def load_preview_image(path: str | Path, *, max_size: tuple[int, int] = (360, 360)):
    """Load an image file as an 8-bit, thumbnail-sized PIL image for preview.

    Handles grayscale, RGB(A), and multi-dimensional (z/channel/time) TIFF
    stacks by reducing extra axes and rescaling non-8-bit data for display.
    """
    import numpy as np
    from PIL import Image

    image_path = Path(path)
    suffix = image_path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        import tifffile

        array = np.asarray(tifffile.imread(str(image_path)))
        while array.ndim > 3:
            array = array[0]
        if array.ndim == 3 and array.shape[-1] not in (3, 4):
            array = array[0]
        if array.dtype != np.uint8:
            floats = array.astype("float32")
            low = float(floats.min())
            high = float(floats.max())
            if high > low:
                floats = (floats - low) / (high - low) * 255.0
            else:
                floats = np.zeros_like(floats)
            array = floats.astype("uint8")
        image = Image.fromarray(array)
    else:
        image = Image.open(image_path)
        image.load()
        if image.mode not in {"RGB", "RGBA", "L"}:
            image = image.convert("RGB")

    image.thumbnail(max_size)
    return image


def open_path(path: str | Path) -> None:
    """Open a file or folder with the operating system default handler."""
    target = Path(path)
    if sys.platform.startswith("win"):
        os.startfile(target)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])


def run_gui_workflow(
    config: GuiWorkflowConfig,
    *,
    runner: Callable[..., CellProfilerWorkflowResult] | None = None,
) -> GuiWorkflowSummary:
    """Validate and run the headless CP/Fiji workflow for the GUI."""
    errors = validate_workflow_config(config)
    if errors:
        raise ValueError("\n".join(errors))

    if runner is None:
        from bioimage_pipeline.analysis import run_cellprofiler_workflow

        runner = run_cellprofiler_workflow

    result = runner(
        config.input_dir,
        config.output_dir,
        config.cppipe_path,
        cellprofiler_executable=config.cellprofiler_executable,
        fiji_executable=config.fiji_executable,
        fiji_macro_path=config.fiji_macro_path,
        export_fiji_tiffs=config.export_fiji_tiffs,
        generate_qc=config.generate_qc,
        oir_projection_engine=config.oir_projection_engine,
    )
    return build_workflow_summary(result)


def _browse_folder(entry: Any) -> None:
    from tkinter import filedialog

    selected = filedialog.askdirectory()
    if selected:
        entry.delete(0, "end")
        entry.insert(0, selected)


def _browse_file(entry: Any, filetypes: list[tuple[str, str]] | None = None) -> None:
    from tkinter import filedialog

    selected = filedialog.askopenfilename(filetypes=filetypes or [("All files", "*.*")])
    if selected:
        entry.delete(0, "end")
        entry.insert(0, selected)


def launch_workflow_shell() -> None:
    """Launch the Phase 15.3 editor-first pipeline workspace."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.geometry("1024x600")
    root.minsize(800, 500)

    editor_session = EditorSession()
    builder_state: dict[str, PipelineBuilderState | None] = {"state": None}
    run_entries: dict[str, tk.Entry] = {}
    executable_cache = build_cached_run_executables()
    sync_discovered_executables_to_settings(executable_cache)
    startup_warnings = [
        *executable_cache.cellprofiler.warnings,
        *executable_cache.fiji.warnings,
    ]

    def pipeline_text() -> str:
        state = builder_state["state"]
        return state.pipeline.to_text() if state is not None else ""

    def refresh_title() -> None:
        editor_session.sync_dirty(pipeline_text())
        root.title(window_title(path=editor_session.path, dirty=editor_session.dirty))

    def mark_modified() -> None:
        editor_session.mark_dirty()
        refresh_title()

    def confirm_discard() -> bool:
        refresh_title()
        if not editor_session.dirty:
            return True
        return messagebox.askyesno(
            "Unsaved changes",
            "Discard unsaved pipeline changes?",
        )

    def apply_builder_state(state: PipelineBuilderState) -> None:
        builder_state["state"] = state
        refresh_pipeline_list()
        refresh_title()

    def working_pipeline_path() -> Path:
        working_dir = Path(tempfile.gettempdir()) / "bioimage_pipeline"
        working_dir.mkdir(parents=True, exist_ok=True)
        return working_dir / "working_pipeline.cppipe"

    def materialize_current_pipeline() -> Path:
        state = builder_state["state"]
        if state is None:
            raise ValueError("Build or load a pipeline before running.")
        saved = save_pipeline_builder_state(state, working_pipeline_path())
        return saved

    # --- Menu bar (Phase C) -------------------------------------------------
    menubar = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=0)
    tools_menu = tk.Menu(menubar, tearoff=0)
    help_menu = tk.Menu(menubar, tearoff=0)

    def file_new() -> None:
        if not confirm_discard():
            return
        state = create_default_pipeline_builder_state()
        builder_state["state"] = state
        editor_session.path = None
        editor_session.baseline_text = state.pipeline.to_text()
        editor_session.dirty = False
        refresh_pipeline_list()
        status.set("New pipeline with required setup modules.")
        refresh_title()

    def file_open() -> None:
        if not confirm_discard():
            return
        selected = filedialog.askopenfilename(
            filetypes=[("CellProfiler pipeline", "*.cppipe"), ("All files", "*.*")],
        )
        if not selected:
            return
        try:
            state = load_pipeline_builder_state(selected)
        except Exception as exc:
            messagebox.showerror("Open pipeline", str(exc))
            return
        builder_state["state"] = state
        editor_session.mark_saved(Path(selected), state.pipeline.to_text())
        refresh_pipeline_list()
        status.set(f"Opened {selected}")
        refresh_title()

    def file_save() -> None:
        state = builder_state["state"]
        if state is None:
            messagebox.showerror("Save pipeline", "No pipeline to save.")
            return
        target = editor_session.path
        if target is None or not str(target):
            file_save_as()
            return
        try:
            save_pipeline_builder_state(state, target)
        except Exception as exc:
            messagebox.showerror("Save pipeline", str(exc))
            return
        editor_session.mark_saved(target, state.pipeline.to_text())
        status.set(f"Saved {target}")
        refresh_title()

    def file_save_as() -> None:
        state = builder_state["state"]
        if state is None:
            messagebox.showerror("Save pipeline", "No pipeline to save.")
            return
        selected = filedialog.asksaveasfilename(
            defaultextension=".cppipe",
            filetypes=[("CellProfiler pipeline", "*.cppipe"), ("All files", "*.*")],
        )
        if not selected:
            return
        try:
            saved = save_pipeline_builder_state(state, selected)
        except Exception as exc:
            messagebox.showerror("Save pipeline", str(exc))
            return
        editor_session.mark_saved(saved, state.pipeline.to_text())
        status.set(f"Saved {saved}")
        refresh_title()

    def tools_open_in_cellprofiler() -> None:
        state = builder_state["state"]
        if state is None:
            messagebox.showerror("CellProfiler", "No pipeline to open.")
            return
        try:
            path = materialize_current_pipeline()
            launch_cellprofiler_gui(
                path,
                cellprofiler_executable=run_entries["cellprofiler_executable"].get().strip()
                or "cellprofiler",
            )
        except Exception as exc:
            messagebox.showerror("CellProfiler", str(exc))

    def help_module() -> None:
        state = builder_state["state"]
        selected = state.selected_module if state is not None else None
        definition = state.selected_definition if state is not None else None
        if selected is None:
            messagebox.showinfo(
                "Module help",
                "Select a module in the pipeline panel to view its help.",
            )
            return
        if definition is not None and definition.description:
            body = definition.description
        else:
            body = "No catalog help is available for this module."
        messagebox.showinfo(f"Help: {selected.name}", body)

    def help_about() -> None:
        messagebox.showinfo(
            "About Bioimage Pipeline",
            "Bioimage Pipeline\n\n"
            "A CellProfiler-style pipeline editor for building, running, and\n"
            "exporting headless CellProfiler/Fiji image-analysis workflows.\n\n"
            "Use Tools > Open in CellProfiler to author advanced pipelines in\n"
            "the native CellProfiler application.",
        )

    file_menu.add_command(label="New", accelerator="Ctrl+N", command=file_new)
    file_menu.add_command(label="Open...", accelerator="Ctrl+O", command=file_open)
    file_menu.add_separator()
    file_menu.add_command(label="Save", accelerator="Ctrl+S", command=file_save)
    file_menu.add_command(label="Save As...", command=file_save_as)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.destroy)
    tools_menu.add_command(label="Open in CellProfiler...", command=tools_open_in_cellprofiler)
    help_menu.add_command(label="Module Help", command=help_module)
    help_menu.add_separator()
    help_menu.add_command(label="About Bioimage Pipeline", command=help_about)
    menubar.add_cascade(label="File", menu=file_menu)
    menubar.add_cascade(label="Tools", menu=tools_menu)
    menubar.add_cascade(label="Help", menu=help_menu)
    root.config(menu=menubar)
    root.bind("<Control-n>", lambda _e: file_new())
    root.bind("<Control-o>", lambda _e: file_open())
    root.bind("<Control-s>", lambda _e: file_save())

    # --- Layout (Phase F: editor-first) -------------------------------------
    main = ttk.Frame(root, padding=8)
    main.pack(fill="both", expand=True)
    main.rowconfigure(0, weight=1)
    main.columnconfigure(0, weight=1)

    editor = ttk.Frame(main)
    editor.grid(row=0, column=0, sticky="nsew")
    editor.columnconfigure(0, weight=1)
    editor.rowconfigure(0, weight=1)

    # CellProfiler uses a vertical splitter: pipeline list (left), file list +
    # module settings (right).
    editor_paned = ttk.Panedwindow(editor, orient="horizontal")
    editor_paned.grid(row=0, column=0, sticky="nsew")

    left_panel = ttk.Frame(editor_paned, padding=4)
    right_panel = ttk.Frame(editor_paned, padding=4)
    editor_paned.add(left_panel, weight=1)
    editor_paned.add(right_panel, weight=2)
    left_panel.columnconfigure(0, weight=1)
    left_panel.rowconfigure(1, weight=1)
    right_panel.columnconfigure(0, weight=1)
    right_panel.rowconfigure(1, weight=1)

    input_frame = ttk.LabelFrame(left_panel, text="Input", padding=4)
    input_frame.grid(row=0, column=0, sticky="ew")
    input_frame.columnconfigure(0, weight=1)
    input_list = tk.Listbox(
        input_frame, height=4, exportselection=False,
        background="#eef3f8", activestyle="none", highlightthickness=1,
    )
    input_list.grid(row=0, column=0, sticky="ew")

    modules_frame = ttk.LabelFrame(left_panel, text="Modules", padding=4)
    modules_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
    modules_frame.columnconfigure(0, weight=1)
    modules_frame.rowconfigure(0, weight=1)
    pipeline_list = tk.Listbox(modules_frame, height=14, exportselection=False)
    pipeline_list.grid(row=0, column=0, sticky="nsew")
    pipeline_scroll = ttk.Scrollbar(modules_frame, orient="vertical", command=pipeline_list.yview)
    pipeline_scroll.grid(row=0, column=1, sticky="ns")
    pipeline_list.configure(yscrollcommand=pipeline_scroll.set)

    module_controls = ttk.Frame(left_panel)
    module_controls.grid(row=2, column=0, sticky="ew", pady=(8, 0))

    outputs_row = ttk.Frame(left_panel)
    outputs_row.grid(row=3, column=0, sticky="ew", pady=(8, 0))

    module_ui_frame = ttk.Frame(right_panel)
    module_ui_frame.grid(row=0, column=0, sticky="nsew")
    module_ui_frame.columnconfigure(0, weight=1)
    module_ui_frame.rowconfigure(2, weight=1)

    notes_panel = ttk.LabelFrame(module_ui_frame, text="Module notes", padding=4)
    notes_panel.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    notes_panel.columnconfigure(0, weight=1)
    module_notes_text = tk.Text(notes_panel, height=3, wrap="word", state="disabled")
    module_notes_text.grid(row=0, column=0, sticky="ew")

    module_settings_title = tk.StringVar(value="Module settings")
    file_list_panel = ttk.LabelFrame(module_ui_frame, text="File list", padding=4)
    file_list_panel.grid(row=1, column=0, sticky="nsew", pady=(0, 6))
    file_list_panel.columnconfigure(0, weight=1)
    file_list_panel.rowconfigure(1, weight=1)
    file_list_tools = ttk.Frame(file_list_panel)
    file_list_tools.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
    input_folder_var = tk.StringVar()
    input_folder_entry = ttk.Entry(file_list_tools, textvariable=input_folder_var)
    input_folder_entry.pack(side="left", fill="x", expand=True)
    file_list_tree = ttk.Treeview(file_list_panel, height=5, show="tree", selectmode="browse")
    file_list_tree.grid(row=1, column=0, sticky="nsew")
    file_list_scroll = ttk.Scrollbar(file_list_panel, orient="vertical", command=file_list_tree.yview)
    file_list_scroll.grid(row=1, column=1, sticky="ns")
    file_list_tree.configure(yscrollcommand=file_list_scroll.set)
    file_list_footer = ttk.Frame(file_list_panel)
    file_list_footer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
    show_excluded_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        file_list_footer, text="Show files excluded by filters", variable=show_excluded_var,
    ).pack(side="left")
    file_list_status = tk.StringVar(value="Drop files and folders here")
    ttk.Label(file_list_footer, textvariable=file_list_status).pack(side="right")
    file_list_panel.grid_remove()

    module_settings_box = ttk.LabelFrame(
        module_ui_frame, text=module_settings_title.get(), padding=4,
    )
    module_settings_box.grid(row=2, column=0, sticky="nsew")

    def _sync_module_settings_box_title(*_args: Any) -> None:
        module_settings_box.configure(text=module_settings_title.get())

    module_settings_title.trace_add("write", _sync_module_settings_box_title)
    module_settings_box.rowconfigure(0, weight=1)
    module_settings_box.columnconfigure(0, weight=1)

    settings_canvas = tk.Canvas(module_settings_box, highlightthickness=0)
    settings_canvas.grid(row=0, column=0, sticky="nsew")
    settings_scroll = ttk.Scrollbar(
        module_settings_box, orient="vertical", command=settings_canvas.yview,
    )
    settings_scroll.grid(row=0, column=1, sticky="ns")
    settings_canvas.configure(yscrollcommand=settings_scroll.set)
    settings_frame = ttk.Frame(settings_canvas)
    settings_window = settings_canvas.create_window((0, 0), window=settings_frame, anchor="nw")

    def _sync_settings_scroll_region(_event: Any) -> None:
        settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))

    def _sync_settings_width(event: Any) -> None:
        settings_canvas.itemconfigure(settings_window, width=event.width)

    settings_frame.bind("<Configure>", _sync_settings_scroll_region)
    settings_canvas.bind("<Configure>", _sync_settings_width)

    imageset_panel = ttk.LabelFrame(module_ui_frame, text="Image sets", padding=4)
    imageset_panel.grid(row=3, column=0, sticky="ew", pady=(6, 0))
    imageset_panel.columnconfigure(0, weight=1)
    imageset_tree = ttk.Treeview(imageset_panel, height=5, show="headings", selectmode="browse")
    imageset_tree.grid(row=0, column=0, sticky="ew")
    imageset_scroll = ttk.Scrollbar(imageset_panel, orient="vertical", command=imageset_tree.yview)
    imageset_scroll.grid(row=0, column=1, sticky="ns")
    imageset_tree.configure(yscrollcommand=imageset_scroll.set)
    imageset_panel.grid_remove()

    output_settings_frame = ttk.Frame(right_panel, padding=4)
    output_settings_frame.columnconfigure(1, weight=1)
    output_settings_frame.grid_remove()

    status = tk.StringVar(
        value=(
            startup_warnings[0]
            if len(startup_warnings) == 1
            else (
                f"{len(startup_warnings)} startup warning(s). See output after run."
                if startup_warnings
                else "Your pipeline is ready. Select your images and press Analyze Images."
            )
        ),
    )
    status_frame = ttk.Frame(main)
    status_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))
    ttk.Label(status_frame, textvariable=status, relief="sunken", anchor="w").pack(
        fill="x", expand=True,
    )

    # --- Output settings (CellProfiler preferences panel) -------------------
    run_panel = output_settings_frame
    run_panel.columnconfigure(1, weight=1)

    def add_run_row(row: int, label: str, key: str, *, browse: str = "file") -> None:
        ttk.Label(run_panel, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(run_panel)
        entry.grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=2)
        run_entries[key] = entry
        if browse == "folder":
            cmd = lambda e=entry: _browse_folder(e)
        else:
            cmd = lambda e=entry: _browse_file(e)
        ttk.Button(run_panel, text="Browse", command=cmd).grid(row=row, column=2, pady=2)

    ttk.Label(
        run_panel, text="Output Settings", font=("Segoe UI", 10, "bold"),
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
    add_run_row(1, "Default Input Folder", "input_dir", browse="folder")
    add_run_row(2, "Default Output Folder", "output_dir", browse="folder")
    add_run_row(3, "CellProfiler executable", "cellprofiler_executable")
    run_entries["cellprofiler_executable"].insert(
        0,
        executable_cache.cellprofiler.display_value,
    )
    add_run_row(4, "Fiji executable", "fiji_executable")
    if executable_cache.fiji.display_value:
        run_entries["fiji_executable"].insert(0, executable_cache.fiji.display_value)
    add_run_row(5, "Fiji macro", "fiji_macro_path")

    ttk.Label(run_panel, text="OIR projection engine").grid(
        row=6, column=0, sticky="w", pady=2,
    )
    oir_projection_engine = tk.StringVar(
        value=default_oir_projection_engine_choice(
            fiji_executable=run_entries["fiji_executable"].get().strip() or None,
        ),
    )
    oir_engine_combo = ttk.Combobox(
        run_panel,
        textvariable=oir_projection_engine,
        values=("python", "fiji"),
        state="readonly",
        width=12,
    )
    oir_engine_combo.grid(row=6, column=1, sticky="w", padx=(8, 8), pady=2)

    export_fiji = tk.BooleanVar(value=True)
    generate_qc = tk.BooleanVar(value=True)
    opts = ttk.Frame(run_panel)
    opts.grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 0))
    ttk.Checkbutton(opts, text="Run Fiji headless export", variable=export_fiji).pack(side="left")
    ttk.Checkbutton(opts, text="Generate QC overlays", variable=generate_qc).pack(
        side="left", padx=(12, 0),
    )

    output_text = tk.Text(run_panel, height=6, wrap="word")
    output_text.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(4, 0))

    results_preview = ttk.Label(run_panel, anchor="w", justify="left")
    results_preview.grid(row=9, column=0, columnspan=3, sticky="w", pady=(6, 0))

    run_buttons = ttk.Frame(run_panel)
    run_buttons.grid(row=10, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def write_output(text: str) -> None:
        output_text.delete("1.0", "end")
        output_text.insert("end", text)

    def show_module_ui(show: bool = True) -> None:
        if show:
            output_settings_frame.grid_remove()
            module_ui_frame.grid(row=0, column=0, sticky="nsew")
        else:
            module_ui_frame.grid_remove()

    def show_output_settings(show: bool = True) -> None:
        if show:
            module_ui_frame.grid_remove()
            output_settings_frame.grid(row=0, column=0, sticky="nsew")
            state = builder_state["state"]
            if state is not None:
                run_entries["input_dir"].delete(0, "end")
                run_entries["input_dir"].insert(0, get_images_input_folder(state))
        else:
            output_settings_frame.grid_remove()

    def sync_input_folder_fields(folder: str) -> None:
        if input_folder_var.get() != folder:
            input_folder_var.set(folder)
        if "input_dir" in run_entries:
            entry = run_entries["input_dir"]
            if entry.get().strip() != folder:
                entry.delete(0, "end")
                entry.insert(0, folder)

    def sync_path_list_visibility(module_name: str | None) -> None:
        """Show the file list only for the Images module, like CellProfiler."""
        if should_show_path_list(module_name):
            file_list_panel.grid()
        else:
            file_list_panel.grid_remove()

    def sync_imageset_visibility(module_name: str | None) -> None:
        if should_show_imageset(module_name):
            imageset_panel.grid()
        else:
            imageset_panel.grid_remove()

    def refresh_notes_panel() -> None:
        state = builder_state["state"]
        module_notes_text.configure(state="normal")
        module_notes_text.delete("1.0", "end")
        if state is None or state.selected_module is None:
            module_notes_text.configure(state="disabled")
            return
        selected = state.selected_module
        notes = parse_module_notes(selected)
        if not notes and state.selected_definition is not None:
            notes = state.selected_definition.description
        module_notes_text.insert("1.0", notes)
        module_notes_text.configure(state="disabled")

    def refresh_imageset_panel() -> None:
        imageset_tree.delete(*imageset_tree.get_children())
        state = builder_state["state"]
        if state is None:
            return
        folder = get_images_input_folder(state)
        columns, rows = build_imageset_rows(state.pipeline, folder)
        imageset_tree.configure(columns=["cycle", *columns])
        imageset_tree.heading("cycle", text="#")
        imageset_tree.column("cycle", width=40, anchor="center")
        for column in columns:
            imageset_tree.heading(column, text=column)
            imageset_tree.column(column, width=140, anchor="w")
        for cycle, values in rows:
            imageset_tree.insert("", "end", values=(cycle, *[values.get(col, "") for col in columns]))

    def refresh_file_list_panel() -> None:
        state = builder_state["state"]
        folder = get_images_input_folder(state) if state is not None else ""
        sync_input_folder_fields(folder)
        file_list_tree.delete(*file_list_tree.get_children())
        if not folder:
            file_list_tree.insert("", "end", text="Drop files and folders here")
            file_list_status.set("Browse for a folder to populate the file list.")
            return
        if show_excluded_var.get():
            files = scan_folder_files(folder)
        else:
            files = scan_detected_images(folder)
        root_id = file_list_tree.insert("", "end", text=Path(folder).name, open=True)
        if files:
            for path in files[:500]:
                file_list_tree.insert(root_id, "end", text=path.name)
            file_list_status.set(f"{len(files)} file(s) in {folder}")
        else:
            file_list_tree.insert(root_id, "end", text="(no files found)")
            file_list_status.set(f"0 file(s) in {folder}")

    def apply_input_folder(*_args: Any) -> None:
        state = builder_state["state"]
        if state is None:
            return
        folder = input_folder_var.get().strip()
        builder_state["state"] = set_images_input_folder(state, folder)
        mark_modified()
        sync_input_folder_fields(folder)
        refresh_file_list_panel()
        refresh_imageset_panel()

    def browse_input_folder() -> None:
        selected = filedialog.askdirectory()
        if not selected:
            return
        input_folder_var.set(selected)
        apply_input_folder()

    def clear_file_list() -> None:
        input_folder_var.set("")
        apply_input_folder()

    ttk.Button(file_list_tools, text="Browse...", command=browse_input_folder).pack(
        side="left", padx=(6, 0),
    )
    ttk.Button(file_list_tools, text="Clear", command=clear_file_list).pack(
        side="left", padx=(6, 0),
    )
    input_folder_entry.bind("<KeyRelease>", apply_input_folder)
    show_excluded_var.trace_add("write", lambda *_args: refresh_file_list_panel())

    def bind_default_input_folder() -> None:
        def sync_from_output_settings(*_args: Any) -> None:
            folder = run_entries["input_dir"].get().strip()
            input_folder_var.set(folder)
            apply_input_folder()

        run_entries["input_dir"].bind("<KeyRelease>", sync_from_output_settings)

    bind_default_input_folder()

    def refresh_settings_panel() -> None:
        for child in settings_frame.winfo_children():
            child.destroy()
        state = builder_state["state"]
        if state is None or state.selected_module is None:
            module_settings_title.set("Module settings")
            sync_path_list_visibility(None)
            sync_imageset_visibility(None)
            refresh_notes_panel()
            ttk.Label(
                settings_frame,
                text="Select a pipeline module to edit settings.",
                columnspan=2,
            ).grid(row=0, column=0, sticky="w", pady=4)
            return

        selected_module = state.selected_module
        values = state.selected_setting_values
        module_settings_title.set(
            module_settings_label(selected_module.name, selected_module.module_num),
        )
        sync_path_list_visibility(selected_module.name)
        sync_imageset_visibility(selected_module.name)
        refresh_notes_panel()
        if should_show_path_list(selected_module.name):
            refresh_file_list_panel()
        if should_show_imageset(selected_module.name):
            refresh_imageset_panel()

        next_row = 0
        parameters = state.selected_visible_parameters
        for parameter in parameters:
            current_value = values.get(parameter.label, parameter.default)
            ttk.Label(
                settings_frame, text=parameter.label, wraplength=240, justify="right",
            ).grid(row=next_row, column=0, sticky="e", pady=3, padx=(0, 8))
            value_var = tk.StringVar(value=current_value)

            def apply_value(
                *_args: Any,
                label: str = parameter.label,
                variable: tk.StringVar = value_var,
            ) -> None:
                st = builder_state["state"]
                if st is None or st.selected_module_index is None:
                    return
                builder_state["state"] = update_pipeline_module_setting(
                    st, st.selected_module_index, label, variable.get(),
                )
                mark_modified()
                status.set("Setting updated.")
                refresh_settings_panel()

            if parameter.choices:
                widget = ttk.Combobox(
                    settings_frame, textvariable=value_var,
                    values=list(parameter.choices), state="readonly",
                )
                widget.bind("<<ComboboxSelected>>", apply_value)
                widget.grid(row=next_row, column=1, sticky="ew", pady=3)
            else:
                widget = ttk.Entry(settings_frame, textvariable=value_var)
                widget.bind("<KeyRelease>", apply_value)
                widget.grid(row=next_row, column=1, sticky="ew", pady=3)
            next_row += 1

        if (
            selected_module.name == "Groups"
            and groups_wants_grouping(state.pipeline)
        ):
            ttk.Label(
                settings_frame,
                text=GROUPS_GROUPING_HELP,
                wraplength=420,
                justify="left",
            ).grid(row=next_row, column=0, columnspan=2, sticky="w", pady=(8, 4))
            next_row += 1

            metadata_choices = list_metadata_category_choices(state.pipeline)
            categories = list_groups_metadata_categories(state.pipeline)

            for category_index, category_value in enumerate(categories):
                ttk.Label(
                    settings_frame,
                    text=GROUPS_METADATA_CATEGORY,
                    wraplength=240,
                    justify="right",
                ).grid(row=next_row, column=0, sticky="e", pady=3, padx=(0, 8))
                category_var = tk.StringVar(value=category_value)

                def apply_category(
                    *_args: Any,
                    index: int = category_index,
                    variable: tk.StringVar = category_var,
                ) -> None:
                    st = builder_state["state"]
                    if st is None or st.selected_module_index is None:
                        return
                    updated_categories = list_groups_metadata_categories(st.pipeline)
                    if index >= len(updated_categories):
                        return
                    updated_categories[index] = variable.get()
                    pipeline = update_groups_metadata_categories(
                        st.pipeline,
                        st.selected_module_index,
                        updated_categories,
                    )
                    builder_state["state"] = PipelineBuilderState(
                        pipeline=pipeline,
                        catalog_modules=list(st.catalog_modules),
                        selected_module_index=st.selected_module_index,
                    )
                    mark_modified()
                    status.set("Grouping metadata updated.")
                    refresh_settings_panel()

                category_widget = ttk.Combobox(
                    settings_frame,
                    textvariable=category_var,
                    values=metadata_choices,
                    state="readonly",
                )
                category_widget.bind("<<ComboboxSelected>>", apply_category)
                category_widget.grid(row=next_row, column=1, sticky="ew", pady=3)

                if category_index > 0:
                    def remove_category(index: int = category_index) -> None:
                        st = builder_state["state"]
                        if st is None or st.selected_module_index is None:
                            return
                        updated_categories = list_groups_metadata_categories(st.pipeline)
                        if len(updated_categories) <= 1 or index >= len(updated_categories):
                            return
                        del updated_categories[index]
                        pipeline = update_groups_metadata_categories(
                            st.pipeline,
                            st.selected_module_index,
                            updated_categories,
                        )
                        builder_state["state"] = PipelineBuilderState(
                            pipeline=pipeline,
                            catalog_modules=list(st.catalog_modules),
                            selected_module_index=st.selected_module_index,
                        )
                        mark_modified()
                        status.set("Grouping metadata removed.")
                        refresh_settings_panel()

                    ttk.Button(
                        settings_frame,
                        text="Remove this metadata item",
                        command=remove_category,
                    ).grid(row=next_row, column=2, sticky="w", padx=(8, 0), pady=3)
                next_row += 1

            def add_metadata_category() -> None:
                st = builder_state["state"]
                if st is None or st.selected_module_index is None:
                    return
                updated_categories = list_groups_metadata_categories(st.pipeline)
                default_category = metadata_choices[0]
                updated_categories.append(default_category)
                pipeline = update_groups_metadata_categories(
                    st.pipeline,
                    st.selected_module_index,
                    updated_categories,
                )
                builder_state["state"] = PipelineBuilderState(
                    pipeline=pipeline,
                    catalog_modules=list(st.catalog_modules),
                    selected_module_index=st.selected_module_index,
                )
                mark_modified()
                status.set("Grouping metadata added.")
                refresh_settings_panel()

            ttk.Button(
                settings_frame,
                text="Add another metadata item",
                command=add_metadata_category,
            ).grid(row=next_row, column=0, columnspan=2, sticky="w", pady=(4, 0))
            next_row += 1

        if not parameters:
            ttk.Label(
                settings_frame, text="No cataloged settings for this module.", columnspan=2,
            ).grid(row=next_row, column=0, sticky="w", pady=4)

        settings_frame.columnconfigure(0, weight=1)
        settings_frame.columnconfigure(1, weight=2)

    def refresh_pipeline_list() -> None:
        input_list.delete(0, "end")
        pipeline_list.delete(0, "end")
        state = builder_state["state"]
        if state is None:
            refresh_settings_panel()
            refresh_file_list_panel()
            return
        input_rows, analysis_rows, protected = split_pipeline_rows(state.pipeline.modules)
        for row in input_rows:
            input_list.insert("end", row)
        for row in analysis_rows:
            pipeline_list.insert("end", row)
        input_list.selection_clear(0, "end")
        pipeline_list.selection_clear(0, "end")
        selected = state.selected_module_index
        if selected is not None:
            if selected < protected:
                input_list.selection_set(selected)
                input_list.see(selected)
            else:
                analysis_index = selected - protected
                pipeline_list.selection_set(analysis_index)
                pipeline_list.see(analysis_index)
        refresh_settings_panel()
        refresh_file_list_panel()
        show_module_ui(True)

    def add_module(module_name: str) -> None:
        state = builder_state["state"]
        if state is None:
            messagebox.showerror("Pipeline builder", "Start a pipeline first.")
            return
        try:
            builder_state["state"] = add_named_module_to_pipeline(state, module_name)
        except Exception as exc:
            messagebox.showerror("Pipeline builder", str(exc))
            return
        mark_modified()
        show_module_ui(True)
        refresh_pipeline_list()
        status.set(f"Added {module_name}.")

    def show_add_module_dialog() -> None:
        def on_help(definition: ModuleDefinition) -> None:
            messagebox.showinfo(f"Help: {definition.name}", definition.description)

        open_add_module_dialog(root, on_add=add_module, on_help=on_help)

    def protected_count() -> int:
        state = builder_state["state"]
        return leading_protected_count(state.pipeline.modules) if state is not None else 0

    def selected_pipeline_index() -> int | None:
        """Return the global module index for the active list selection."""
        analysis_selection = pipeline_list.curselection()
        if analysis_selection:
            return protected_count() + analysis_selection[0]
        input_selection = input_list.curselection()
        if input_selection:
            return input_selection[0]
        return None

    def select_module_from_list(_event: Any | None = None) -> None:
        state = builder_state["state"]
        if state is None:
            return
        selection = pipeline_list.curselection()
        if not selection:
            return
        input_list.selection_clear(0, "end")
        builder_state["state"] = select_pipeline_module(
            state, protected_count() + selection[0],
        )
        show_module_ui(True)
        refresh_settings_panel()

    def select_input_module_from_list(_event: Any | None = None) -> None:
        state = builder_state["state"]
        if state is None:
            return
        selection = input_list.curselection()
        if not selection:
            return
        pipeline_list.selection_clear(0, "end")
        builder_state["state"] = select_pipeline_module(state, selection[0])
        show_module_ui(True)
        refresh_settings_panel()

    def delete_selected_module(_event: Any | None = None) -> None:
        state = builder_state["state"]
        index = selected_pipeline_index()
        if state is None or index is None:
            messagebox.showerror("Pipeline builder", "Select a pipeline module first.")
            return
        module = state.pipeline.modules[index]
        if is_protected_module_name(module.name):
            messagebox.showinfo(
                "Protected module",
                f"{module.name} is a required setup module and cannot be removed.",
            )
            return
        builder_state["state"] = remove_pipeline_module(state, index)
        mark_modified()
        refresh_pipeline_list()
        status.set(f"Removed {module.name}.")

    def move_selected_module_by(delta: int) -> None:
        state = builder_state["state"]
        index = selected_pipeline_index()
        if state is None or index is None:
            messagebox.showerror("Pipeline builder", "Select a pipeline module first.")
            return
        module = state.pipeline.modules[index]
        if is_protected_module_name(module.name):
            status.set("Required setup modules keep a fixed position.")
            return
        floor = leading_protected_count(state.pipeline.modules)
        target = max(floor, min(index + delta, len(state.pipeline.modules) - 1))
        if target == index:
            return
        builder_state["state"] = move_pipeline_module(state, index, target)
        mark_modified()
        refresh_pipeline_list()
        status.set("Module order updated.")

    def current_config() -> GuiWorkflowConfig:
        state = builder_state["state"]

        def optional_value(key: str) -> str | None:
            value = run_entries[key].get().strip()
            return value or None

        input_dir = get_images_input_folder(state) if state is not None else ""
        return GuiWorkflowConfig(
            input_dir=input_dir,
            output_dir=run_entries["output_dir"].get().strip(),
            cppipe_path="",
            cellprofiler_executable=run_entries["cellprofiler_executable"].get().strip()
            or "cellprofiler",
            fiji_executable=optional_value("fiji_executable"),
            fiji_macro_path=optional_value("fiji_macro_path"),
            export_fiji_tiffs=export_fiji.get(),
            generate_qc=generate_qc.get(),
            oir_projection_engine=oir_projection_engine.get().strip() or "python",
        )

    def run_async() -> None:
        state = builder_state["state"]
        if state is None or not state.pipeline.modules:
            messagebox.showerror("Run pipeline", "Build or load a pipeline first.")
            return
        try:
            resolve_workflow_input_dir(state)
            cppipe_path = materialize_current_pipeline()
        except ValueError as exc:
            messagebox.showerror("Invalid pipeline", str(exc))
            return

        config = current_config()
        config = GuiWorkflowConfig(
            input_dir=str(resolve_workflow_input_dir(state)),
            output_dir=config.output_dir,
            cppipe_path=str(cppipe_path),
            cellprofiler_executable=config.cellprofiler_executable,
            fiji_executable=config.fiji_executable,
            fiji_macro_path=config.fiji_macro_path,
            export_fiji_tiffs=config.export_fiji_tiffs,
            generate_qc=config.generate_qc,
            oir_projection_engine=config.oir_projection_engine,
        )
        errors = validate_workflow_config(config)
        if errors:
            messagebox.showerror("Invalid workflow settings", "\n".join(errors))
            return

        save_gui_run_settings(
            collect_run_settings_from_values(
                cellprofiler_executable=config.cellprofiler_executable,
                fiji_executable=str(config.fiji_executable or ""),
            )
        )

        run_button.configure(state="disabled")
        status.set("Running headless CellProfiler/Fiji workflow...")
        write_output("Workflow started.\n")

        def worker() -> None:
            try:
                summary = run_gui_workflow(config)
            except Exception as exc:
                root.after(0, lambda exc=exc: _finish_error(exc))
                return
            root.after(0, lambda summary=summary: _finish_success(summary))

        threading.Thread(target=worker, daemon=True).start()

    def _show_results_preview(summary: GuiWorkflowSummary) -> None:
        from PIL import ImageTk

        candidates = [
            *summary.qc_preview_files,
            *summary.mask_files,
            *summary.label_files,
        ]
        if not candidates:
            results_preview.configure(image="", text="")
            results_preview.image = None  # type: ignore[attr-defined]
            return
        target = candidates[0]
        try:
            photo = ImageTk.PhotoImage(load_preview_image(target))
        except Exception as exc:  # noqa: BLE001 - show any decode error inline
            results_preview.configure(image="", text=f"Output preview unavailable: {exc}")
            results_preview.image = None  # type: ignore[attr-defined]
            return
        results_preview.configure(
            image=photo, text=f"Output preview: {target.name}", compound="top",
        )
        results_preview.image = photo  # type: ignore[attr-defined]  # keep a reference

    def _finish_success(summary: GuiWorkflowSummary) -> None:
        status.set("Workflow complete.")
        write_output("\n".join(summary.to_display_lines()))
        run_button.configure(state="normal")
        _show_results_preview(summary)

    def _finish_error(exc: Exception) -> None:
        import traceback

        status.set("Workflow failed.")
        write_output("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        run_button.configure(state="normal")

    def open_results() -> None:
        output_dir = run_entries["output_dir"].get().strip()
        if output_dir:
            open_path(output_dir)

    run_button = ttk.Button(run_buttons, text="Analyze Images", command=run_async)
    run_button.pack(side="left")
    ttk.Button(run_buttons, text="Open Results Folder", command=open_results).pack(
        side="left", padx=(8, 0),
    )

    pipeline_list.bind("<<ListboxSelect>>", select_module_from_list)
    input_list.bind("<<ListboxSelect>>", select_input_module_from_list)
    pipeline_menu = tk.Menu(root, tearoff=0)
    pipeline_menu.add_command(label="Delete module", command=delete_selected_module)

    def show_pipeline_menu(event: Any) -> None:
        nearest = pipeline_list.nearest(event.y)
        if nearest >= 0:
            pipeline_list.selection_clear(0, "end")
            pipeline_list.selection_set(nearest)
            pipeline_list.activate(nearest)
            select_module_from_list()
        st = builder_state["state"]
        global_index = protected_count() + nearest if nearest >= 0 else -1
        protected = (
            st is not None
            and 0 <= global_index < len(st.pipeline.modules)
            and is_protected_module_name(st.pipeline.modules[global_index].name)
        )
        pipeline_menu.entryconfigure("Delete module", state="disabled" if protected else "normal")
        pipeline_menu.tk_popup(event.x_root, event.y_root)

    pipeline_list.bind("<Button-3>", show_pipeline_menu)
    pipeline_list.bind("<Delete>", delete_selected_module)

    ttk.Button(module_controls, text="?", width=3, command=help_module).pack(side="left")
    ttk.Label(module_controls, text="Adjust modules:").pack(side="left", padx=(8, 4))
    add_module_button = ttk.Button(
        module_controls, text="+", width=3, command=show_add_module_dialog,
    )
    add_module_button.pack(side="left")
    ttk.Button(module_controls, text="-", width=3, command=delete_selected_module).pack(
        side="left", padx=(4, 0),
    )
    ttk.Button(
        module_controls, text="^", width=3, command=lambda: move_selected_module_by(-1),
    ).pack(side="left", padx=(4, 0))
    ttk.Button(
        module_controls, text="v", width=3, command=lambda: move_selected_module_by(1),
    ).pack(side="left", padx=(4, 0))

    ttk.Button(outputs_row, text="Output Settings", command=lambda: show_output_settings(True)).pack(
        fill="x",
    )

    if startup_warnings:
        write_output("\n".join(startup_warnings))

    def persist_current_run_settings() -> None:
        save_gui_run_settings(
            collect_run_settings_from_values(
                cellprofiler_executable=run_entries["cellprofiler_executable"].get(),
                fiji_executable=run_entries["fiji_executable"].get(),
            )
        )

    def on_close() -> None:
        persist_current_run_settings()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    file_new()
    refresh_title()
    root.mainloop()
