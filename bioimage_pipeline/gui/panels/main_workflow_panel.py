"""GUI-1: minimal workflow path controls and run actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from bioimage_pipeline.gui.run_settings import (
    CELLPROFILER_SETTINGS_KEY,
    EXPORT_FIJI_TIFFS_KEY,
    FIJI_MACRO_PATH_KEY,
    FIJI_SETTINGS_KEY,
    GENERATE_QC_KEY,
    INPUT_DIR_KEY,
    OUTPUT_DIR_KEY,
    build_cached_run_executables,
    parse_bool_setting,
)


@dataclass
class MainWorkflowPanel:
    """Widgets for import/run/open-output workflow controls."""

    input_dir_entry: Any
    output_dir_entry: Any
    cellprofiler_executable_entry: Any
    fiji_executable_entry: Any
    fiji_macro_entry: Any
    export_fiji_var: Any
    generate_qc_var: Any
    run_button: Any
    open_results_button: Any
    threshold_recommender_button: Any


def build_main_workflow_panel(
    parent: Any,
    *,
    saved_settings: dict[str, str],
    browse_folder: Callable[[Any], None],
    browse_file: Callable[[Any], None],
) -> MainWorkflowPanel:
    """Create GUI-1 path fields and primary workflow action buttons."""
    import tkinter as tk
    from tkinter import ttk

    executable_cache = build_cached_run_executables(saved_settings)

    ttk.Label(parent, text="Default Input Folder").grid(row=0, column=0, sticky="w", pady=2)
    input_dir_entry = ttk.Entry(parent)
    input_dir_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8), pady=2)
    input_dir_entry.insert(0, saved_settings.get(INPUT_DIR_KEY, ""))
    ttk.Button(
        parent,
        text="Browse",
        command=lambda: browse_folder(input_dir_entry),
    ).grid(row=0, column=2, pady=2)

    ttk.Label(parent, text="Default Output Folder").grid(row=1, column=0, sticky="w", pady=2)
    output_dir_entry = ttk.Entry(parent)
    output_dir_entry.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=2)
    output_dir_entry.insert(0, saved_settings.get(OUTPUT_DIR_KEY, ""))
    ttk.Button(
        parent,
        text="Browse",
        command=lambda: browse_folder(output_dir_entry),
    ).grid(row=1, column=2, pady=2)

    ttk.Label(parent, text="CellProfiler executable").grid(row=2, column=0, sticky="w", pady=2)
    cellprofiler_executable_entry = ttk.Entry(parent)
    cellprofiler_executable_entry.grid(row=2, column=1, sticky="ew", padx=(8, 8), pady=2)
    cellprofiler_executable_entry.insert(
        0,
        saved_settings.get(CELLPROFILER_SETTINGS_KEY)
        or executable_cache.cellprofiler.display_value,
    )
    ttk.Button(
        parent,
        text="Browse",
        command=lambda: browse_file(cellprofiler_executable_entry),
    ).grid(row=2, column=2, pady=2)

    ttk.Label(parent, text="Fiji executable").grid(row=3, column=0, sticky="w", pady=2)
    fiji_executable_entry = ttk.Entry(parent)
    fiji_executable_entry.grid(row=3, column=1, sticky="ew", padx=(8, 8), pady=2)
    fiji_value = saved_settings.get(FIJI_SETTINGS_KEY) or executable_cache.fiji.display_value
    if fiji_value:
        fiji_executable_entry.insert(0, fiji_value)
    ttk.Button(
        parent,
        text="Browse",
        command=lambda: browse_file(fiji_executable_entry),
    ).grid(row=3, column=2, pady=2)

    ttk.Label(parent, text="Fiji export macro").grid(row=4, column=0, sticky="w", pady=2)
    fiji_macro_entry = ttk.Entry(parent)
    fiji_macro_entry.grid(row=4, column=1, sticky="ew", padx=(8, 8), pady=2)
    fiji_macro_entry.insert(0, saved_settings.get(FIJI_MACRO_PATH_KEY, ""))
    ttk.Button(
        parent,
        text="Browse",
        command=lambda: browse_file(fiji_macro_entry),
    ).grid(row=4, column=2, pady=2)

    export_fiji_var = tk.BooleanVar(
        value=parse_bool_setting(saved_settings.get(EXPORT_FIJI_TIFFS_KEY), default=True),
    )
    generate_qc_var = tk.BooleanVar(
        value=parse_bool_setting(saved_settings.get(GENERATE_QC_KEY), default=True),
    )
    opts = ttk.Frame(parent)
    opts.grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))
    ttk.Checkbutton(opts, text="Run Fiji headless export", variable=export_fiji_var).pack(
        side="left",
    )
    ttk.Checkbutton(opts, text="Generate QC overlays", variable=generate_qc_var).pack(
        side="left",
        padx=(12, 0),
    )

    run_buttons = ttk.Frame(parent)
    run_buttons.grid(row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))
    run_button = ttk.Button(run_buttons, text="Analyze Images")
    run_button.pack(side="left")
    open_results_button = ttk.Button(run_buttons, text="Open Results Folder")
    open_results_button.pack(side="left", padx=(8, 0))
    threshold_recommender_button = ttk.Button(
        run_buttons,
        text="Threshold Recommender…",
    )
    threshold_recommender_button.pack(side="left", padx=(8, 0))

    return MainWorkflowPanel(
        input_dir_entry=input_dir_entry,
        output_dir_entry=output_dir_entry,
        cellprofiler_executable_entry=cellprofiler_executable_entry,
        fiji_executable_entry=fiji_executable_entry,
        fiji_macro_entry=fiji_macro_entry,
        export_fiji_var=export_fiji_var,
        generate_qc_var=generate_qc_var,
        run_button=run_button,
        open_results_button=open_results_button,
        threshold_recommender_button=threshold_recommender_button,
    )
