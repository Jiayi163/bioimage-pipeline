"""Phase 15 import-only GUI workflow shell.

Users author pipelines in native CellProfiler, import a ``.cppipe`` path, and run
headless CellProfiler/Fiji orchestration via
:func:`bioimage_pipeline.analysis.run_cellprofiler_workflow`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from bioimage_pipeline.analysis import CellProfilerWorkflowResult, resolve_workflow_output_dir
from bioimage_pipeline.cellprofiler_runner import find_cellprofiler_executable
from bioimage_pipeline.fiji_runner import find_fiji_executable, fiji_not_found_message
from bioimage_pipeline.gui.run_settings import (
    OIR_PROJECTION_METHOD_KEY,
    build_cached_run_executables,
    collect_run_settings_from_values,
    load_gui_run_settings,
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
    advise_pipeline_for_run,
    append_module,
    create_pipeline_from_catalog,
    load_and_validate_imported_pipeline,
    load_cppipe,
    move_module,
    remove_module,
    save_cppipe,
    summarize_modules,
    update_module_setting,
    validate_cppipe,
)
from bioimage_pipeline.pipeline_catalog import ModuleDefinition, list_modules, search_modules
from bioimage_pipeline.workflow_timing import format_timing_breakdown
from bioimage_pipeline.gui.workflow_editor import (
    launch_cellprofiler_gui,
    list_module_output_lines_for_pipeline,
    resolve_workflow_input_dir_from_string,
    scan_detected_images,
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
    oir_projection_engine: str = "fiji"
    oir_projection_method: str = "max"
    force_oir_reproject: bool = False


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
            lines.append("")
            lines.extend(format_timing_breakdown(self.timing).splitlines())
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        return lines


@dataclass
class ImportedPipelineState:
    """Imported CellProfiler pipeline loaded read-only from disk."""

    path: Path
    pipeline: CppipePipeline


def load_imported_pipeline(cppipe_path: str | Path) -> ImportedPipelineState:
    """Load and validate an imported ``.cppipe`` file for the GUI shell."""
    path = Path(cppipe_path).resolve()
    pipeline = load_and_validate_imported_pipeline(path)
    return ImportedPipelineState(path=path, pipeline=pipeline)


def resolve_imported_pipeline_path(cppipe_path: str | Path) -> Path:
    """Return a validated imported pipeline path."""
    return load_imported_pipeline(cppipe_path).path


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
    """Validate and save the builder pipeline without rewriting module settings."""
    errors = validate_cppipe(state.pipeline)
    if errors:
        raise ValueError("\n".join(errors))
    return save_cppipe(state.pipeline, path)


def default_oir_projection_engine_choice(
    *,
    fiji_executable: str | Path | None = None,
) -> str:
    """Return the GUI default OIR projection engine for this machine."""
    if find_fiji_executable(fiji_executable) is not None:
        return "fiji"
    if python_oir_dependencies_available():
        return "python"
    return "fiji"


def validate_workflow_config(config: GuiWorkflowConfig) -> list[str]:
    """Return validation errors for a GUI workflow config."""
    errors: list[str] = []
    if not str(config.input_dir).strip():
        errors.append("Select an input folder before running.")
    if not str(config.output_dir).strip():
        errors.append("Output folder is required.")

    input_dir = Path(config.input_dir) if str(config.input_dir).strip() else None
    output_dir = (
        Path(config.output_dir).expanduser().resolve()
        if str(config.output_dir).strip()
        else None
    )

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
        engine = (config.oir_projection_engine or "fiji").strip().lower()
        if engine not in {"python", "fiji", "auto"}:
            errors.append(
                "OIR projection engine must be 'python', 'fiji', or 'auto'."
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

    resolved_output_dir = resolve_workflow_output_dir(config.output_dir)
    result = runner(
        config.input_dir,
        str(resolved_output_dir),
        config.cppipe_path,
        cellprofiler_executable=config.cellprofiler_executable,
        fiji_executable=config.fiji_executable,
        fiji_macro_path=config.fiji_macro_path,
        export_fiji_tiffs=config.export_fiji_tiffs,
        generate_qc=config.generate_qc,
        oir_projection_engine=config.oir_projection_engine,
        oir_projection_method=config.oir_projection_method,
        force_oir_reproject=config.force_oir_reproject,
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
    """Launch the import-only CellProfiler workflow shell (GUI-1..4)."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from bioimage_pipeline.gui.panels.main_workflow_panel import build_main_workflow_panel
    from bioimage_pipeline.gui.panels.preprocessing_panel import build_preprocessing_panel
    from bioimage_pipeline.gui.panels.results_polish_panel import (
        build_results_polish_panel,
        schedule_log_polling,
    )
    from bioimage_pipeline.gui.run_settings import (
        CPPIPE_PATH_KEY,
        INPUT_DIR_KEY,
        OUTPUT_DIR_KEY,
        collect_run_settings_from_values,
        extract_recent_workflow_paths,
        load_gui_run_settings,
        save_gui_run_settings,
        sync_discovered_executables_to_settings,
    )
    from bioimage_pipeline.gui.threshold_recommender_window import (
        ThresholdRecommenderLaunchContext,
        launch_threshold_recommender_window,
        reset_open_threshold_recommender_windows,
    )
    from bioimage_pipeline.gui.workflow_controller import (
        WorkflowFormValues,
        prepare_workflow_run,
    )
    from bioimage_pipeline.gui.workflow_session_reset import (
        clear_experiment_paths_from_settings,
        needs_reset_confirmation,
        workflow_has_experiment_fields,
        workflow_run_actions_enabled,
    )

    root = tk.Tk()
    root.geometry("1040x640")
    root.minsize(820, 520)

    imported_state: dict[str, ImportedPipelineState | None] = {"state": None}
    pipeline_path_var = tk.StringVar()
    run_state: dict[str, bool] = {"running": False}
    saved_settings = load_gui_run_settings()
    sync_discovered_executables_to_settings(build_cached_run_executables(saved_settings))
    saved_settings = load_gui_run_settings()
    startup_warnings = [
        *build_cached_run_executables(saved_settings).cellprofiler.warnings,
        *build_cached_run_executables(saved_settings).fiji.warnings,
    ]

    def refresh_title() -> None:
        state = imported_state["state"]
        root.title(window_title(path=state.path if state else None, dirty=False))

    def refresh_module_list(module_list: tk.Listbox) -> None:
        module_list.delete(0, "end")
        state = imported_state["state"]
        if state is None:
            return
        for row in summarize_modules(state.pipeline.modules):
            module_list.insert("end", row)
        outputs = list_module_output_lines_for_pipeline(state.pipeline)
        if outputs:
            module_list.insert("end", "")
            module_list.insert("end", "Outputs:")
            for line in outputs:
                module_list.insert("end", f"  {line}")

    def set_imported_pipeline(path: str | Path) -> None:
        state = load_imported_pipeline(path)
        imported_state["state"] = state
        pipeline_path_var.set(str(state.path))
        refresh_title()

    def file_open(module_list: tk.Listbox) -> None:
        selected = filedialog.askopenfilename(
            filetypes=[("CellProfiler pipeline", "*.cppipe"), ("All files", "*.*")],
        )
        if not selected:
            return
        try:
            set_imported_pipeline(selected)
        except Exception as exc:
            messagebox.showerror("Open pipeline", str(exc))
            return
        refresh_module_list(module_list)
        status.set(f"Imported {selected}")
        persist_settings()
        update_workflow_action_states()

    def browse_pipeline(module_list: tk.Listbox) -> None:
        file_open(module_list)

    def tools_open_in_cellprofiler(
        cellprofiler_executable_entry: ttk.Entry,
    ) -> None:
        path = pipeline_path_var.get().strip()
        if not path:
            messagebox.showerror("CellProfiler", "Import a pipeline file first.")
            return
        try:
            launch_cellprofiler_gui(
                resolve_imported_pipeline_path(path),
                cellprofiler_executable=cellprofiler_executable_entry.get().strip()
                or "cellprofiler",
            )
        except Exception as exc:
            messagebox.showerror("CellProfiler", str(exc))

    def help_about() -> None:
        messagebox.showinfo(
            "About Bioimage Pipeline",
            "Bioimage Pipeline\n\n"
            "Import a CellProfiler .cppipe file, set input/output folders, and run "
            "headless CellProfiler/Fiji orchestration with logs and QC.\n\n"
            "Author pipelines in CellProfiler (Tools > Open in CellProfiler).",
        )

    menubar = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=0)
    tools_menu = tk.Menu(menubar, tearoff=0)
    help_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Open Pipeline...", accelerator="Ctrl+O")
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.destroy)
    help_menu.add_command(label="About Bioimage Pipeline", command=help_about)
    menubar.add_cascade(label="File", menu=file_menu)
    menubar.add_cascade(label="Tools", menu=tools_menu)
    menubar.add_cascade(label="Help", menu=help_menu)
    root.config(menu=menubar)

    main = ttk.Frame(root, padding=8)
    main.pack(fill="both", expand=True)
    main.rowconfigure(0, weight=1)
    main.columnconfigure(0, weight=1)

    paned = ttk.Panedwindow(main, orient="horizontal")
    paned.grid(row=0, column=0, sticky="nsew")

    left = ttk.Frame(paned, padding=4)
    right = ttk.Frame(paned, padding=4)
    paned.add(left, weight=1)
    paned.add(right, weight=2)
    left.columnconfigure(0, weight=1)
    left.rowconfigure(2, weight=1)
    right.columnconfigure(0, weight=1)
    right.rowconfigure(1, weight=1)

    pipeline_frame = ttk.LabelFrame(left, text="CellProfiler pipeline", padding=4)
    pipeline_frame.grid(row=0, column=0, sticky="ew")
    pipeline_frame.columnconfigure(0, weight=1)
    ttk.Entry(pipeline_frame, textvariable=pipeline_path_var).grid(
        row=0, column=0, sticky="ew",
    )
    pipeline_tools = ttk.Frame(pipeline_frame)
    pipeline_tools.grid(row=1, column=0, sticky="ew", pady=(4, 0))
    browse_pipeline_button = ttk.Button(pipeline_tools, text="Browse...")
    browse_pipeline_button.pack(side="left")
    open_in_cp_button = ttk.Button(pipeline_tools, text="Open in CellProfiler")
    open_in_cp_button.pack(side="left", padx=(8, 0))

    modules_frame = ttk.LabelFrame(left, text="Modules (read-only)", padding=4)
    modules_frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
    modules_frame.columnconfigure(0, weight=1)
    modules_frame.rowconfigure(0, weight=1)
    module_list = tk.Listbox(modules_frame, height=16, exportselection=False)
    module_list.grid(row=0, column=0, sticky="nsew")
    module_scroll = ttk.Scrollbar(modules_frame, orient="vertical", command=module_list.yview)
    module_scroll.grid(row=0, column=1, sticky="ns")
    module_list.configure(yscrollcommand=module_scroll.set)

    ttk.Label(
        left,
        text="Edit the pipeline in CellProfiler, then re-import the saved .cppipe file.",
        wraplength=320,
        justify="left",
    ).grid(row=3, column=0, sticky="w", pady=(8, 0))

    workflow_frame = ttk.LabelFrame(right, text="Workflow", padding=4)
    workflow_frame.grid(row=0, column=0, sticky="ew")
    workflow_frame.columnconfigure(1, weight=1)

    workflow_panel = build_main_workflow_panel(
        workflow_frame,
        saved_settings=saved_settings,
        browse_folder=_browse_folder,
        browse_file=_browse_file,
    )

    preprocessing_frame = ttk.LabelFrame(
        right,
        text="Fiji / OIR preprocessing",
        padding=4,
    )
    preprocessing_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
    preprocessing_frame.columnconfigure(1, weight=1)
    preprocessing_panel = build_preprocessing_panel(
        preprocessing_frame,
        saved_settings=saved_settings,
        fiji_executable=workflow_panel.fiji_executable_entry.get().strip(),
    )

    results_frame = ttk.LabelFrame(right, text="Results", padding=4)
    results_frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
    results_frame.columnconfigure(0, weight=1)
    results_frame.rowconfigure(0, weight=1)
    right.rowconfigure(2, weight=1)
    polish_panel = build_results_polish_panel(results_frame)

    status = tk.StringVar(
        value=(
            startup_warnings[0]
            if len(startup_warnings) == 1
            else (
                f"{len(startup_warnings)} startup warning(s). See Summary tab after run."
                if startup_warnings
                else "Import a CellProfiler pipeline and press Analyze Images."
            )
        ),
    )
    ttk.Label(main, textvariable=status, relief="sunken", anchor="w").grid(
        row=1, column=0, sticky="ew", pady=(4, 0),
    )

    browse_pipeline_button.configure(command=lambda: browse_pipeline(module_list))
    file_menu.entryconfigure("Open Pipeline...", command=lambda: file_open(module_list))
    root.bind("<Control-o>", lambda _e: file_open(module_list))
    open_in_cp_button.configure(
        command=lambda: tools_open_in_cellprofiler(
            workflow_panel.cellprofiler_executable_entry,
        ),
    )
    tools_menu.add_command(
        label="Open in CellProfiler...",
        command=lambda: tools_open_in_cellprofiler(
            workflow_panel.cellprofiler_executable_entry,
        ),
    )
    help_menu.add_command(label="About Bioimage Pipeline", command=help_about)

    def current_form_values(*, cppipe_path: str | None = None) -> WorkflowFormValues:
        return WorkflowFormValues(
            input_dir=workflow_panel.input_dir_entry.get().strip(),
            output_dir=workflow_panel.output_dir_entry.get().strip(),
            cppipe_path=cppipe_path or pipeline_path_var.get().strip(),
            cellprofiler_executable=workflow_panel.cellprofiler_executable_entry.get().strip(),
            fiji_executable=workflow_panel.fiji_executable_entry.get().strip(),
            fiji_macro_path=workflow_panel.fiji_macro_entry.get().strip(),
            export_fiji_tiffs=workflow_panel.export_fiji_var.get(),
            generate_qc=workflow_panel.generate_qc_var.get(),
            oir_projection_engine=preprocessing_panel.oir_projection_engine_var.get().strip(),
            oir_projection_method=preprocessing_panel.oir_projection_method_var.get().strip(),
            force_oir_reproject=preprocessing_panel.force_oir_reproject_var.get(),
        )

    def persist_settings(*, oir_projection_method: str | None = None) -> None:
        values = current_form_values()
        payload = load_gui_run_settings()
        payload.update(
            collect_run_settings_from_values(
                cellprofiler_executable=values.cellprofiler_executable,
                fiji_executable=values.fiji_executable,
                oir_projection_method=oir_projection_method or values.oir_projection_method,
                cppipe_path=values.cppipe_path,
                input_dir=values.input_dir,
                output_dir=values.output_dir,
                fiji_macro_path=values.fiji_macro_path,
                oir_projection_engine=values.oir_projection_engine,
                export_fiji_tiffs=values.export_fiji_tiffs,
                generate_qc=values.generate_qc,
                force_oir_reproject=values.force_oir_reproject,
            )
        )
        save_gui_run_settings(payload)

    def results_panel_has_display() -> bool:
        return polish_panel._last_summary is not None

    def update_workflow_action_states() -> None:
        enabled = workflow_run_actions_enabled(
            cppipe_path=pipeline_path_var.get(),
            input_dir=workflow_panel.input_dir_entry.get(),
            output_dir=workflow_panel.output_dir_entry.get(),
            running=run_state["running"],
        )
        run_state_value = "normal" if enabled else "disabled"
        workflow_panel.run_button.configure(state=run_state_value)
        workflow_panel.threshold_recommender_button.configure(state=run_state_value)
        output_dir = workflow_panel.output_dir_entry.get().strip()
        open_results_state = "normal" if output_dir and not run_state["running"] else "disabled"
        workflow_panel.open_results_button.configure(state=open_results_state)
        clear_state = "disabled" if run_state["running"] else "normal"
        workflow_panel.clear_session_button.configure(state=clear_state)

    def apply_cleared_experiment_state(*, status_message: str) -> None:
        pipeline_path_var.set("")
        imported_state["state"] = None
        refresh_module_list(module_list)
        refresh_title()

        workflow_panel.input_dir_entry.delete(0, "end")
        workflow_panel.output_dir_entry.delete(0, "end")

        polish_panel.clear_session()
        reset_open_threshold_recommender_windows()

        payload = clear_experiment_paths_from_settings(load_gui_run_settings())
        payload.update(
            collect_run_settings_from_values(
                cellprofiler_executable=workflow_panel.cellprofiler_executable_entry.get().strip(),
                fiji_executable=workflow_panel.fiji_executable_entry.get().strip(),
                oir_projection_method=preprocessing_panel.oir_projection_method_var.get().strip(),
                fiji_macro_path=workflow_panel.fiji_macro_entry.get().strip(),
                oir_projection_engine=preprocessing_panel.oir_projection_engine_var.get().strip(),
                export_fiji_tiffs=workflow_panel.export_fiji_var.get(),
                generate_qc=workflow_panel.generate_qc_var.get(),
                force_oir_reproject=preprocessing_panel.force_oir_reproject_var.get(),
            )
        )
        save_gui_run_settings(payload)
        status.set(status_message)
        update_workflow_action_states()

    def clear_experiment_session() -> None:
        if run_state["running"]:
            messagebox.showwarning(
                "Clear session",
                "A workflow run is in progress. Wait for it to finish before clearing.",
            )
            return

        values = current_form_values()
        has_fields = workflow_has_experiment_fields(
            cppipe_path=values.cppipe_path,
            input_dir=values.input_dir,
            output_dir=values.output_dir,
            pipeline_loaded=imported_state["state"] is not None,
        )
        if not has_fields and not results_panel_has_display():
            status.set("Session is already clear.")
            return

        if needs_reset_confirmation(
            cppipe_path=values.cppipe_path,
            input_dir=values.input_dir,
            output_dir=values.output_dir,
            pipeline_loaded=imported_state["state"] is not None,
            has_result_display=results_panel_has_display(),
            run_in_progress=run_state["running"],
        ):
            proceed = messagebox.askyesno(
                "Clear session",
                "Clear the current pipeline, input/output paths, and displayed results?\n\n"
                "Saved files on disk are not deleted.",
            )
            if not proceed:
                return

        apply_cleared_experiment_state(
            status_message="Session cleared. Import a pipeline and set folders to start a new run.",
        )

    def run_async() -> None:
        path_text = pipeline_path_var.get().strip()
        if not path_text:
            messagebox.showerror("Run pipeline", "Import a CellProfiler pipeline first.")
            return

        try:
            preparation = prepare_workflow_run(current_form_values())
        except ValueError as exc:
            messagebox.showerror("Invalid workflow settings", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("Invalid pipeline", str(exc))
            return

        if preparation.advisories:
            message = "\n".join(f"- {line}" for line in preparation.advisories)
            proceed = messagebox.askyesno(
                "Pipeline advisories",
                message + "\n\nContinue with this run?",
            )
            if not proceed:
                return

        config = preparation.config
        oir_projection_method = config.oir_projection_method
        if preparation.requires_oir_method_dialog:
            from bioimage_pipeline.gui.oir_projection_dialog import ask_oir_projection_method

            saved_method = load_gui_run_settings().get(
                "oir_projection_method",
                oir_projection_method,
            )
            selected_method = ask_oir_projection_method(root, default=saved_method)
            if selected_method is None:
                return
            oir_projection_method = selected_method
            config = GuiWorkflowConfig(
                input_dir=config.input_dir,
                output_dir=config.output_dir,
                cppipe_path=config.cppipe_path,
                cellprofiler_executable=config.cellprofiler_executable,
                fiji_executable=config.fiji_executable,
                fiji_macro_path=config.fiji_macro_path,
                export_fiji_tiffs=config.export_fiji_tiffs,
                generate_qc=config.generate_qc,
                oir_projection_engine=config.oir_projection_engine,
                oir_projection_method=oir_projection_method,
                force_oir_reproject=config.force_oir_reproject,
            )

        workflow_panel.output_dir_entry.delete(0, "end")
        workflow_panel.output_dir_entry.insert(0, str(config.output_dir))
        persist_settings(oir_projection_method=oir_projection_method)

        run_state["running"] = True
        workflow_panel.run_button.configure(state="disabled")
        workflow_panel.threshold_recommender_button.configure(state="disabled")
        workflow_panel.open_results_button.configure(state="disabled")
        workflow_panel.clear_session_button.configure(state="disabled")
        status.set("Running headless CellProfiler/Fiji workflow...")
        polish_panel.set_running(output_dir=str(config.output_dir))
        schedule_log_polling(
            root,
            polish_panel,
            output_dir=str(config.output_dir),
            cancel_event=lambda: not run_state["running"],
        )

        def worker() -> None:
            try:
                summary = run_gui_workflow(config)
            except Exception as exc:
                root.after(0, lambda exc=exc: _finish_error(exc))
                return
            root.after(0, lambda summary=summary: _finish_success(summary))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_success(summary: GuiWorkflowSummary) -> None:
        run_state["running"] = False
        polish_panel.set_idle()
        polish_panel.show_success(summary)
        status.set("Workflow complete.")
        update_workflow_action_states()

    def _finish_error(exc: Exception) -> None:
        import traceback

        run_state["running"] = False
        polish_panel.set_idle()
        polish_panel.show_error("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        status.set("Workflow failed.")
        update_workflow_action_states()

    def open_results() -> None:
        output_dir = workflow_panel.output_dir_entry.get().strip()
        if output_dir:
            open_path(output_dir)

    def open_threshold_recommender() -> None:
        path_text = pipeline_path_var.get().strip()
        input_dir = workflow_panel.input_dir_entry.get().strip()
        output_dir = workflow_panel.output_dir_entry.get().strip()
        if not path_text:
            messagebox.showerror(
                "Threshold Recommender",
                "Import a CellProfiler pipeline first.",
            )
            return
        if not input_dir:
            messagebox.showerror(
                "Threshold Recommender",
                "Select a default input folder first.",
            )
            return
        if not output_dir:
            messagebox.showerror(
                "Threshold Recommender",
                "Select a default output folder first.",
            )
            return
        try:
            cppipe_path = str(resolve_imported_pipeline_path(path_text))
            resolved_output_dir = str(resolve_workflow_output_dir(output_dir))
        except ValueError as exc:
            messagebox.showerror("Threshold Recommender", str(exc))
            return

        launch_threshold_recommender_window(
            root,
            ThresholdRecommenderLaunchContext(
                cppipe_path=cppipe_path,
                input_dir=input_dir,
                output_dir=resolved_output_dir,
                cellprofiler_executable=workflow_panel.cellprofiler_executable_entry.get().strip()
                or "cellprofiler",
            ),
        )

    def load_recent_paths(module_list: tk.Listbox) -> None:
        recent = extract_recent_workflow_paths(load_gui_run_settings())
        input_path = recent.get(INPUT_DIR_KEY, "")
        output_path = recent.get(OUTPUT_DIR_KEY, "")
        cppipe_path = recent.get(CPPIPE_PATH_KEY, "")
        if not any((input_path, output_path, cppipe_path)):
            messagebox.showinfo(
                "Load recent paths",
                "No recent input, output, or pipeline paths are saved yet.",
            )
            return

        workflow_panel.input_dir_entry.delete(0, "end")
        if input_path:
            workflow_panel.input_dir_entry.insert(0, input_path)
        workflow_panel.output_dir_entry.delete(0, "end")
        if output_path:
            workflow_panel.output_dir_entry.insert(0, output_path)

        if cppipe_path:
            try:
                set_imported_pipeline(cppipe_path)
            except Exception as exc:
                pipeline_path_var.set(cppipe_path)
                messagebox.showwarning(
                    "Load recent paths",
                    f"Restored pipeline path, but import validation failed:\n{exc}",
                )
        else:
            pipeline_path_var.set("")
            imported_state["state"] = None

        refresh_module_list(module_list)
        refresh_title()
        status.set("Loaded recent workflow paths.")
        update_workflow_action_states()

    workflow_panel.load_recent_paths_button.configure(
        command=lambda: load_recent_paths(module_list),
    )
    workflow_panel.clear_session_button.configure(command=clear_experiment_session)
    workflow_panel.run_button.configure(command=run_async)
    workflow_panel.open_results_button.configure(command=open_results)
    workflow_panel.threshold_recommender_button.configure(command=open_threshold_recommender)

    for entry in (
        workflow_panel.input_dir_entry,
        workflow_panel.output_dir_entry,
    ):
        entry.bind("<KeyRelease>", lambda _event: update_workflow_action_states())
    pipeline_path_var.trace_add("write", lambda *_args: update_workflow_action_states())

    if startup_warnings:
        polish_panel.summary_text.insert("end", "\n".join(startup_warnings))

    def on_close() -> None:
        persist_settings()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh_title()
    update_workflow_action_states()
    root.mainloop()
