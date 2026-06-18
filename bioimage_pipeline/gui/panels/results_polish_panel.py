"""GUI-4: progress, logs, measurements preview, and output shortcuts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from bioimage_pipeline.gui.workflow_controller import (
    format_measurements_preview_text,
    format_workflow_log_tail,
    output_shortcut_targets,
)
from bioimage_pipeline.gui.workflow_shell import GuiWorkflowSummary, open_path


@dataclass
class ResultsPolishPanel:
    """Widgets for run feedback and output navigation."""

    progress_bar: Any
    notebook: Any
    summary_text: Any
    logs_text: Any
    measurements_text: Any
    preview_label: Any
    shortcut_buttons: dict[str, Any]
    _active_output_dir: str | None = None
    _last_summary: GuiWorkflowSummary | None = None

    def set_running(self, *, output_dir: str) -> None:
        """Show indeterminate progress and clear panels for a new run."""
        self._active_output_dir = output_dir
        self._last_summary = None
        self.progress_bar.grid()
        self.progress_bar.start(12)
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("end", "Workflow started.\n")
        self.logs_text.delete("1.0", "end")
        self.measurements_text.delete("1.0", "end")
        self.preview_label.configure(image="", text="")
        self.preview_label.image = None  # type: ignore[attr-defined]

    def set_idle(self) -> None:
        """Stop the progress indicator."""
        self.progress_bar.stop()
        self.progress_bar.grid_remove()

    def update_logs(self, output_dir: str | None = None) -> None:
        """Refresh the live log panel from workflow log files."""
        target = output_dir or self._active_output_dir
        if not target:
            return
        content = format_workflow_log_tail(target)
        self.logs_text.delete("1.0", "end")
        self.logs_text.insert("end", content or "(No log output yet.)")

    def show_success(self, summary: GuiWorkflowSummary) -> None:
        """Populate summary, logs, measurements, and preview after a run."""
        self._last_summary = summary
        self._active_output_dir = str(summary.results_dir)
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("end", "\n".join(summary.to_display_lines()))
        self.update_logs(str(summary.results_dir))
        self.measurements_text.delete("1.0", "end")
        self.measurements_text.insert(
            "end",
            format_measurements_preview_text(summary.measurement_files),
        )
        self._show_preview(summary)

    def show_error(self, message: str) -> None:
        """Display a workflow failure in the summary panel."""
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("end", message)
        self.update_logs()

    def open_shortcut(self, key: str) -> None:
        """Open a named output folder from the last successful run."""
        if self._last_summary is None:
            if self._active_output_dir:
                open_path(self._active_output_dir)
            return
        targets = output_shortcut_targets(self._last_summary)
        target = targets.get(key)
        if target is not None:
            open_path(target)

    def _show_preview(self, summary: GuiWorkflowSummary) -> None:
        from PIL import ImageTk

        from bioimage_pipeline.gui.workflow_shell import load_preview_image

        candidates = [
            *summary.qc_preview_files,
            *summary.mask_files,
            *summary.label_files,
        ]
        if not candidates:
            self.preview_label.configure(image="", text="No preview image available.")
            self.preview_label.image = None  # type: ignore[attr-defined]
            return
        target = candidates[0]
        try:
            photo = ImageTk.PhotoImage(load_preview_image(target))
        except Exception as exc:  # noqa: BLE001
            self.preview_label.configure(image="", text=f"Preview unavailable: {exc}")
            self.preview_label.image = None  # type: ignore[attr-defined]
            return
        self.preview_label.configure(
            image=photo,
            text=f"Output preview: {target.name}",
            compound="top",
        )
        self.preview_label.image = photo  # type: ignore[attr-defined]


def build_results_polish_panel(parent: Any) -> ResultsPolishPanel:
    """Create GUI-4 notebook tabs, progress bar, and output shortcuts."""
    from tkinter import ttk

    progress_bar = ttk.Progressbar(parent, mode="indeterminate")
    progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    progress_bar.grid_remove()

    notebook = ttk.Notebook(parent)
    notebook.grid(row=1, column=0, sticky="nsew")
    parent.rowconfigure(1, weight=1)
    parent.columnconfigure(0, weight=1)

    summary_frame = ttk.Frame(notebook, padding=4)
    logs_frame = ttk.Frame(notebook, padding=4)
    measurements_frame = ttk.Frame(notebook, padding=4)
    preview_frame = ttk.Frame(notebook, padding=4)
    notebook.add(summary_frame, text="Summary")
    notebook.add(logs_frame, text="Logs")
    notebook.add(measurements_frame, text="Measurements")
    notebook.add(preview_frame, text="QC Preview")

    summary_text = _make_text_widget(summary_frame)
    logs_text = _make_text_widget(logs_frame)
    measurements_text = _make_text_widget(measurements_frame)

    preview_label = ttk.Label(preview_frame, anchor="center", justify="center")
    preview_label.pack(fill="both", expand=True)

    shortcuts = ttk.Frame(parent)
    shortcuts.grid(row=2, column=0, sticky="w", pady=(6, 0))
    panel = ResultsPolishPanel(
        progress_bar=progress_bar,
        notebook=notebook,
        summary_text=summary_text,
        logs_text=logs_text,
        measurements_text=measurements_text,
        preview_label=preview_label,
        shortcut_buttons={},
    )

    shortcut_specs = (
        ("results", "Results"),
        ("measurements", "Measurements"),
        ("qc", "QC"),
        ("logs", "Logs"),
    )
    for key, label in shortcut_specs:
        button = ttk.Button(
            shortcuts,
            text=f"Open {label}",
            command=lambda shortcut=key: panel.open_shortcut(shortcut),
        )
        button.pack(side="left", padx=(0, 8))
        panel.shortcut_buttons[key] = button

    return panel


def _make_text_widget(parent: Any) -> Any:
    from tkinter import ttk

    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True)
    text = _TextWidget(frame)
    text.pack(side="left", fill="both", expand=True)
    scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    scroll.pack(side="right", fill="y")
    text.configure(yscrollcommand=scroll.set)
    return text


def _TextWidget(parent: Any) -> Any:
    import tkinter as tk

    return tk.Text(parent, height=12, wrap="word")


def schedule_log_polling(
    root: Any,
    panel: ResultsPolishPanel,
    *,
    output_dir: str,
    interval_ms: int = 1000,
    cancel_event: Callable[[], bool] | None = None,
) -> None:
    """Poll workflow logs until the run completes or the shell closes."""

    def _poll() -> None:
        if cancel_event is not None and cancel_event():
            return
        panel.update_logs(output_dir)
        root.after(interval_ms, _poll)

    root.after(interval_ms, _poll)
