"""GUI-3: Fiji/OIR preprocessing controls (thin wrapper over library options)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bioimage_pipeline.gui.run_settings import (
    FORCE_OIR_REPROJECT_KEY,
    OIR_PROJECTION_ENGINE_KEY,
    OIR_PROJECTION_METHOD_KEY,
    parse_bool_setting,
)
from bioimage_pipeline.gui.workflow_shell import default_oir_projection_engine_choice


@dataclass
class PreprocessingPanel:
    """Widgets for OIR/Fiji preprocessing options."""

    oir_projection_engine_var: Any
    oir_projection_method_var: Any
    force_oir_reproject_var: Any
    info_label: Any


def build_preprocessing_panel(
    parent: Any,
    *,
    saved_settings: dict[str, str],
    fiji_executable: str,
) -> PreprocessingPanel:
    """Create GUI-3 preprocessing controls."""
    import tkinter as tk
    from tkinter import ttk

    default_engine = default_oir_projection_engine_choice(
        fiji_executable=fiji_executable or None,
    )
    oir_projection_engine_var = tk.StringVar(
        value=saved_settings.get(OIR_PROJECTION_ENGINE_KEY, default_engine),
    )
    oir_projection_method_var = tk.StringVar(
        value=saved_settings.get(OIR_PROJECTION_METHOD_KEY, "max"),
    )
    force_oir_reproject_var = tk.BooleanVar(
        value=parse_bool_setting(
            saved_settings.get(FORCE_OIR_REPROJECT_KEY),
            default=False,
        ),
    )

    ttk.Label(parent, text="OIR projection engine").grid(row=0, column=0, sticky="w", pady=2)
    ttk.Combobox(
        parent,
        textvariable=oir_projection_engine_var,
        values=("fiji", "python", "auto"),
        state="readonly",
        width=12,
    ).grid(row=0, column=1, sticky="w", padx=(8, 8), pady=2)

    ttk.Label(parent, text="Default Z projection").grid(row=1, column=0, sticky="w", pady=2)
    ttk.Combobox(
        parent,
        textvariable=oir_projection_method_var,
        values=("max", "min", "mean", "sum", "median", "std"),
        state="readonly",
        width=12,
    ).grid(row=1, column=1, sticky="w", padx=(8, 8), pady=2)

    ttk.Checkbutton(
        parent,
        text="Force OIR reprojection (ignore cache)",
        variable=force_oir_reproject_var,
    ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

    info_label = ttk.Label(
        parent,
        text=(
            "Preprocessing runs automatically when the input folder contains .oir files. "
            "Fiji export macro above applies after CellProfiler analysis."
        ),
        wraplength=520,
        justify="left",
    )
    info_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

    return PreprocessingPanel(
        oir_projection_engine_var=oir_projection_engine_var,
        oir_projection_method_var=oir_projection_method_var,
        force_oir_reproject_var=force_oir_reproject_var,
        info_label=info_label,
    )
