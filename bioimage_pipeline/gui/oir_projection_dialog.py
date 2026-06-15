"""Modal dialog for choosing Fiji Z projection method before OIR batch runs."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from bioimage_pipeline.z_projection import (
    DEFAULT_Z_PROJECTION_METHOD,
    Z_PROJECTION_GUI_LABELS,
    Z_PROJECTION_METHODS,
    ZProjectionMethod,
    normalize_projection_method,
)


def ask_oir_projection_method(
    parent,
    *,
    default: str | ZProjectionMethod = DEFAULT_Z_PROJECTION_METHOD,
) -> str | None:
    """Prompt for a Z projection method and return the slug, or ``None`` if cancelled."""
    try:
        initial = normalize_projection_method(default)
    except ValueError:
        initial = DEFAULT_Z_PROJECTION_METHOD

    result: dict[str, str | None] = {"method": None}

    dialog = tk.Toplevel(parent)
    dialog.title("Z Projection")
    dialog.geometry("360x320")
    dialog.minsize(320, 280)
    dialog.transient(parent)
    dialog.grab_set()
    dialog.resizable(False, False)

    body = ttk.Frame(dialog, padding=12)
    body.pack(fill="both", expand=True)
    ttk.Label(
        body,
        text=(
            "Choose the Z projection method for .oir files "
            "(Fiji Z Project when Fiji engine is selected):"
        ),
        wraplength=320,
    ).pack(anchor="w")

    selected = tk.StringVar(value=initial)
    options = ttk.Frame(body)
    options.pack(fill="both", expand=True, pady=(12, 0))
    for method in Z_PROJECTION_METHODS:
        ttk.Radiobutton(
            options,
            text=Z_PROJECTION_GUI_LABELS[method],
            value=method,
            variable=selected,
        ).pack(anchor="w", pady=2)

    controls = ttk.Frame(dialog, padding=(12, 0, 12, 12))
    controls.pack(fill="x")

    def accept() -> None:
        result["method"] = selected.get()
        dialog.grab_release()
        dialog.destroy()

    def cancel() -> None:
        result["method"] = None
        dialog.grab_release()
        dialog.destroy()

    ttk.Button(controls, text="Cancel", command=cancel).pack(side="right")
    ttk.Button(controls, text="OK", command=accept).pack(side="right", padx=(0, 8))

    dialog.protocol("WM_DELETE_WINDOW", cancel)
    dialog.bind("<Return>", lambda _event: accept())
    dialog.bind("<Escape>", lambda _event: cancel())
    dialog.wait_window()

    method = result["method"]
    if method is None:
        return None
    return normalize_projection_method(method)
