"""Phase 15.1 GUI workflow shell.

The shell is a thin front-end over the existing headless orchestration APIs. It
does not open Fiji or CellProfiler UI windows; it calls their CLI/headless paths
through :func:`bioimage_pipeline.analysis.run_cellprofiler_workflow`.
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

from bioimage_pipeline.analysis import CellProfilerWorkflowResult
from bioimage_pipeline.cppipe_io import (
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
    errors = validate_cppipe(state.pipeline)
    if errors:
        raise ValueError("\n".join(errors))
    return save_cppipe(state.pipeline, path)


def validate_workflow_config(config: GuiWorkflowConfig) -> list[str]:
    """Return validation errors for a GUI workflow config."""
    errors: list[str] = []
    if not str(config.input_dir).strip():
        errors.append("Input folder is required.")
    if not str(config.output_dir).strip():
        errors.append("Output folder is required.")
    if not str(config.cppipe_path).strip():
        errors.append("CellProfiler pipeline file is required.")

    input_dir = Path(config.input_dir)
    output_dir = Path(config.output_dir)
    cppipe_path = Path(config.cppipe_path)

    if not input_dir.is_dir():
        errors.append(f"Input folder does not exist: {input_dir}")
    elif not any(path.is_file() for path in input_dir.iterdir()):
        errors.append(f"Input folder is empty: {input_dir}")
    if not cppipe_path.is_file():
        errors.append(f"CellProfiler pipeline file does not exist: {cppipe_path}")
    if output_dir.exists() and not output_dir.is_dir():
        errors.append(f"Output path exists but is not a folder: {output_dir}")
    if not str(config.cellprofiler_executable).strip():
        errors.append("CellProfiler executable is required.")
    if config.fiji_executable and not Path(config.fiji_executable).is_file():
        errors.append(f"Fiji executable does not exist: {config.fiji_executable}")
    if config.fiji_macro_path and not Path(config.fiji_macro_path).is_file():
        errors.append(f"Fiji macro does not exist: {config.fiji_macro_path}")
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
    """Launch the Phase 15.1 desktop workflow shell."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("Bioimage Pipeline Workflow Shell")
    root.geometry("1120x760")

    main = ttk.Frame(root, padding=12)
    main.pack(fill="both", expand=True)
    main.columnconfigure(1, weight=1)
    main.rowconfigure(9, weight=1)

    entries: dict[str, tk.Entry] = {}

    def add_path_row(
        row: int,
        label: str,
        key: str,
        *,
        browse: str = "file",
        filetypes: list[tuple[str, str]] | None = None,
    ) -> None:
        ttk.Label(main, text=label).grid(row=row, column=0, sticky="w", pady=3)
        entry = ttk.Entry(main)
        entry.grid(row=row, column=1, sticky="ew", pady=3, padx=(8, 8))
        entries[key] = entry
        if browse == "folder":
            command = lambda e=entry: _browse_folder(e)
        else:
            command = lambda e=entry, ft=filetypes: _browse_file(e, ft)
        ttk.Button(main, text="Browse", command=command).grid(row=row, column=2, pady=3)

    add_path_row(0, "Input folder", "input_dir", browse="folder")
    add_path_row(1, "Output folder", "output_dir", browse="folder")
    add_path_row(
        2,
        ".cppipe file",
        "cppipe_path",
        filetypes=[("CellProfiler pipeline", "*.cppipe"), ("All files", "*.*")],
    )
    add_path_row(3, "CellProfiler executable", "cellprofiler_executable")
    entries["cellprofiler_executable"].insert(0, "cellprofiler")
    add_path_row(4, "Fiji executable", "fiji_executable")
    add_path_row(
        5,
        "Fiji macro",
        "fiji_macro_path",
        filetypes=[("ImageJ macro", "*.ijm"), ("All files", "*.*")],
    )

    export_fiji = tk.BooleanVar(value=True)
    generate_qc = tk.BooleanVar(value=True)
    ttk.Checkbutton(main, text="Run Fiji headless export", variable=export_fiji).grid(
        row=6,
        column=1,
        sticky="w",
        pady=4,
    )
    ttk.Checkbutton(main, text="Generate QC overlays", variable=generate_qc).grid(
        row=7,
        column=1,
        sticky="w",
        pady=4,
    )

    status = tk.StringVar(value="Ready.")
    ttk.Label(main, textvariable=status).grid(row=8, column=0, columnspan=3, sticky="w")

    output_text = tk.Text(main, height=20, wrap="word")
    output_text.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=(8, 8))

    builder = ttk.LabelFrame(main, text="Pipeline Builder (15.2)", padding=8)
    builder.grid(row=0, column=3, rowspan=11, sticky="nsew", padx=(12, 0))
    builder.columnconfigure(0, weight=1)
    builder.columnconfigure(1, weight=2)
    builder.rowconfigure(1, weight=1)

    ttk.Label(builder, text="Selected pipeline modules").grid(row=0, column=0, sticky="w")
    ttk.Label(builder, text="Settings for selected module").grid(
        row=0,
        column=1,
        sticky="w",
        padx=(10, 0),
    )

    left_panel = ttk.Frame(builder)
    left_panel.grid(row=1, column=0, sticky="nsew")
    left_panel.rowconfigure(0, weight=1)
    left_panel.columnconfigure(0, weight=1)

    pipeline_list = tk.Listbox(left_panel, height=14, exportselection=False)
    pipeline_list.grid(row=0, column=0, sticky="nsew")
    pipeline_scroll = ttk.Scrollbar(left_panel, orient="vertical", command=pipeline_list.yview)
    pipeline_scroll.grid(row=0, column=1, sticky="ns")
    pipeline_list.configure(yscrollcommand=pipeline_scroll.set)

    module_controls = ttk.Frame(left_panel)
    module_controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

    settings_canvas = tk.Canvas(builder, height=240, highlightthickness=0)
    settings_canvas.grid(row=1, column=1, sticky="nsew", padx=(10, 0))
    settings_scroll = ttk.Scrollbar(builder, orient="vertical", command=settings_canvas.yview)
    settings_scroll.grid(row=1, column=2, sticky="ns")
    settings_canvas.configure(yscrollcommand=settings_scroll.set)
    settings_frame = ttk.Frame(settings_canvas)
    settings_window = settings_canvas.create_window((0, 0), window=settings_frame, anchor="nw")

    def _sync_settings_scroll_region(event: Any) -> None:
        settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))

    def _sync_settings_width(event: Any) -> None:
        settings_canvas.itemconfigure(settings_window, width=event.width)

    settings_frame.bind("<Configure>", _sync_settings_scroll_region)
    settings_canvas.bind("<Configure>", _sync_settings_width)

    catalog_panel = ttk.LabelFrame(builder, text="Add module from catalog", padding=6)
    catalog_panel.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
    catalog_panel.columnconfigure(0, weight=1)
    catalog_query = ttk.Entry(catalog_panel)
    catalog_query.grid(row=0, column=0, sticky="ew")
    catalog_list = tk.Listbox(catalog_panel, height=5, exportselection=False)
    catalog_list.grid(row=1, column=0, sticky="ew", pady=(4, 0))

    builder_status = tk.StringVar(value="Create or load a .cppipe to edit.")
    ttk.Label(builder, textvariable=builder_status, wraplength=420).grid(
        row=4,
        column=0,
        columnspan=3,
        sticky="ew",
        pady=(8, 0),
    )

    button_frame = ttk.Frame(main)
    button_frame.grid(row=10, column=0, columnspan=3, sticky="ew")

    builder_state: dict[str, PipelineBuilderState | None] = {"state": None}

    def write_output(text: str) -> None:
        output_text.delete("1.0", "end")
        output_text.insert("end", text)

    def refresh_catalog(modules: list[ModuleDefinition] | None = None) -> None:
        catalog_list.delete(0, "end")
        for module in modules or list_modules():
            catalog_list.insert("end", f"{module.name} ({module.category})")

    def refresh_pipeline_list() -> None:
        pipeline_list.delete(0, "end")
        state = builder_state["state"]
        if state is None:
            refresh_settings_panel()
            return
        for row in state.module_rows:
            pipeline_list.insert("end", row)
        if state.selected_module_index is not None:
            pipeline_list.selection_clear(0, "end")
            pipeline_list.selection_set(state.selected_module_index)
            pipeline_list.see(state.selected_module_index)
        refresh_settings_panel()

    def refresh_settings_panel() -> None:
        for child in settings_frame.winfo_children():
            child.destroy()
        state = builder_state["state"]
        if state is None or state.selected_module is None:
            ttk.Label(settings_frame, text="Select a pipeline module to edit settings.").grid(
                row=0,
                column=0,
                sticky="w",
                pady=4,
            )
            return

        selected_module = state.selected_module
        definition = state.selected_definition
        values = state.selected_setting_values
        title = selected_module.display_name
        if definition is not None:
            title = f"{selected_module.display_name} - {definition.category}"
        ttk.Label(settings_frame, text=title).grid(row=0, column=0, columnspan=2, sticky="w")
        if definition is not None and definition.description:
            ttk.Label(
                settings_frame,
                text=definition.description,
                wraplength=360,
            ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 8))

        parameters = state.selected_visible_parameters
        if not parameters:
            ttk.Label(settings_frame, text="No cataloged settings for this module.").grid(
                row=2,
                column=0,
                sticky="w",
                pady=4,
            )
            return

        for row, parameter in enumerate(parameters, start=2):
            current_value = values.get(parameter.label, parameter.default)
            ttk.Label(settings_frame, text=parameter.label, wraplength=180).grid(
                row=row,
                column=0,
                sticky="nw",
                pady=3,
                padx=(0, 8),
            )
            value_var = tk.StringVar(value=current_value)

            def apply_value(
                *_args: Any,
                label: str = parameter.label,
                variable: tk.StringVar = value_var,
            ) -> None:
                state = builder_state["state"]
                if state is None or state.selected_module_index is None:
                    return
                builder_state["state"] = update_pipeline_module_setting(
                    state,
                    state.selected_module_index,
                    label,
                    variable.get(),
                )
                builder_status.set("Setting updated in the internal pipeline model.")
                refresh_settings_panel()

            if parameter.choices:
                widget = ttk.Combobox(
                    settings_frame,
                    textvariable=value_var,
                    values=list(parameter.choices),
                    state="readonly",
                )
                widget.bind("<<ComboboxSelected>>", apply_value)
            else:
                widget = ttk.Entry(settings_frame, textvariable=value_var)
                widget.bind("<KeyRelease>", apply_value)
            widget.grid(row=row, column=1, sticky="ew", pady=3)
            if parameter.description:
                widget.configure(width=30)
        settings_frame.columnconfigure(1, weight=1)

    def load_builder_pipeline() -> None:
        cppipe_path = entries["cppipe_path"].get().strip()
        if not cppipe_path:
            messagebox.showerror("Pipeline builder", "Choose a .cppipe file first.")
            return
        try:
            state = load_pipeline_builder_state(
                cppipe_path,
                query=catalog_query.get().strip(),
            )
        except Exception as exc:
            messagebox.showerror("Pipeline builder", str(exc))
            return
        builder_state["state"] = state
        refresh_catalog(state.catalog_modules)
        refresh_pipeline_list()
        builder_status.set(f"Loaded {len(state.pipeline.modules)} module(s).")

    def create_builder_pipeline() -> None:
        state = create_default_pipeline_builder_state(query=catalog_query.get().strip())
        builder_state["state"] = state
        refresh_catalog(state.catalog_modules)
        refresh_pipeline_list()
        builder_status.set("Created minimal Phase 15.2 pipeline.")

    def search_catalog() -> None:
        modules = search_modules(catalog_query.get().strip())
        state = builder_state["state"]
        if state is not None:
            builder_state["state"] = PipelineBuilderState(
                pipeline=state.pipeline,
                catalog_modules=modules,
                selected_module_index=state.selected_module_index,
            )
        refresh_catalog(modules)

    def add_selected_module() -> None:
        state = builder_state["state"]
        selection = catalog_list.curselection()
        if state is None:
            messagebox.showerror("Pipeline builder", "Load a .cppipe file first.")
            return
        if not selection:
            messagebox.showerror("Pipeline builder", "Select a catalog module first.")
            return
        try:
            builder_state["state"] = add_catalog_module_to_pipeline(state, selection[0])
        except Exception as exc:
            messagebox.showerror("Pipeline builder", str(exc))
            return
        refresh_pipeline_list()
        builder_status.set("Module appended. Save the pipeline before running.")

    def selected_pipeline_index() -> int | None:
        selection = pipeline_list.curselection()
        if not selection:
            return None
        return selection[0]

    def select_module_from_list(_event: Any | None = None) -> None:
        state = builder_state["state"]
        if state is None:
            return
        builder_state["state"] = select_pipeline_module(state, selected_pipeline_index())
        refresh_settings_panel()

    def remove_selected_module() -> None:
        state = builder_state["state"]
        index = selected_pipeline_index()
        if state is None or index is None:
            messagebox.showerror("Pipeline builder", "Select a pipeline module first.")
            return
        builder_state["state"] = remove_pipeline_module(state, index)
        refresh_pipeline_list()
        builder_status.set("Module removed from the internal pipeline model.")

    def move_selected_module_by(delta: int) -> None:
        state = builder_state["state"]
        index = selected_pipeline_index()
        if state is None or index is None:
            messagebox.showerror("Pipeline builder", "Select a pipeline module first.")
            return
        target = index + delta
        if not 0 <= target < len(state.pipeline.modules):
            return
        builder_state["state"] = move_pipeline_module(state, index, target)
        refresh_pipeline_list()
        builder_status.set("Module order updated in the internal pipeline model.")

    def save_builder_pipeline() -> None:
        from tkinter import filedialog

        state = builder_state["state"]
        if state is None:
            messagebox.showerror("Pipeline builder", "Load a .cppipe file first.")
            return
        selected = filedialog.asksaveasfilename(
            defaultextension=".cppipe",
            filetypes=[("CellProfiler pipeline", "*.cppipe"), ("All files", "*.*")],
        )
        if not selected:
            return
        try:
            saved_path = save_pipeline_builder_state(state, selected)
        except Exception as exc:
            messagebox.showerror("Pipeline builder", str(exc))
            return
        entries["cppipe_path"].delete(0, "end")
        entries["cppipe_path"].insert(0, str(saved_path))
        builder_status.set(f"Saved: {saved_path}")

    def current_config() -> GuiWorkflowConfig:
        def optional_value(key: str) -> str | None:
            value = entries[key].get().strip()
            return value or None

        return GuiWorkflowConfig(
            input_dir=entries["input_dir"].get().strip(),
            output_dir=entries["output_dir"].get().strip(),
            cppipe_path=entries["cppipe_path"].get().strip(),
            cellprofiler_executable=entries["cellprofiler_executable"].get().strip()
            or "cellprofiler",
            fiji_executable=optional_value("fiji_executable"),
            fiji_macro_path=optional_value("fiji_macro_path"),
            export_fiji_tiffs=export_fiji.get(),
            generate_qc=generate_qc.get(),
        )

    def run_async() -> None:
        config = current_config()
        errors = validate_workflow_config(config)
        if errors:
            messagebox.showerror("Invalid workflow settings", "\n".join(errors))
            return

        run_button.configure(state="disabled")
        status.set("Running headless CellProfiler/Fiji workflow...")
        write_output("Workflow started. External tools run headlessly.\n")

        def worker() -> None:
            try:
                summary = run_gui_workflow(config)
            except Exception as exc:  # GUI boundary: show concise error to user.
                root.after(0, lambda: _finish_error(exc))
                return
            root.after(0, lambda: _finish_success(summary))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_success(summary: GuiWorkflowSummary) -> None:
        status.set("Workflow complete.")
        write_output("\n".join(summary.to_display_lines()))
        run_button.configure(state="normal")

    def _finish_error(exc: Exception) -> None:
        status.set("Workflow failed.")
        write_output(str(exc))
        run_button.configure(state="normal")

    def open_results() -> None:
        output_dir = entries["output_dir"].get().strip()
        if output_dir:
            open_path(output_dir)

    run_button = ttk.Button(button_frame, text="Run Workflow", command=run_async)
    run_button.pack(side="left")
    ttk.Button(button_frame, text="Open Results Folder", command=open_results).pack(
        side="left",
        padx=(8, 0),
    )

    pipeline_list.bind("<<ListboxSelect>>", select_module_from_list)

    ttk.Button(module_controls, text="Remove", command=remove_selected_module).pack(
        side="left",
    )
    ttk.Button(module_controls, text="Up", command=lambda: move_selected_module_by(-1)).pack(
        side="left",
        padx=(6, 0),
    )
    ttk.Button(module_controls, text="Down", command=lambda: move_selected_module_by(1)).pack(
        side="left",
        padx=(6, 0),
    )

    catalog_buttons = ttk.Frame(catalog_panel)
    catalog_buttons.grid(row=2, column=0, sticky="ew", pady=(6, 0))
    ttk.Button(catalog_buttons, text="Search", command=search_catalog).pack(side="left")
    ttk.Button(catalog_buttons, text="Add Module", command=add_selected_module).pack(
        side="left",
        padx=(8, 0),
    )

    builder_buttons = ttk.Frame(builder)
    builder_buttons.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
    ttk.Button(builder_buttons, text="New Minimal Pipeline", command=create_builder_pipeline).pack(
        side="left",
    )
    ttk.Button(builder_buttons, text="Load .cppipe", command=load_builder_pipeline).pack(
        side="left",
        padx=(8, 0),
    )
    ttk.Button(builder_buttons, text="Save As", command=save_builder_pipeline).pack(
        side="left",
        padx=(8, 0),
    )
    refresh_catalog()
    refresh_settings_panel()

    root.mainloop()
