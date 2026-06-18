"""Separate Tk window for subset-first threshold recommender trials."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bioimage_pipeline.gui.threshold_recommender_helpers import (
    build_recommender_config,
    selected_manual_subset_names,
)
from bioimage_pipeline.gui.workflow_editor import scan_detected_images
from bioimage_pipeline.gui.workflow_shell import load_preview_image, open_path
from bioimage_pipeline.threshold_recommender import (
    ThresholdRecommenderTrialResult,
    apply_confirmed_threshold_variant,
    load_trial_result_from_session,
    run_threshold_recommender_trial,
)


@dataclass(frozen=True)
class ThresholdRecommenderLaunchContext:
    """Prefilled values when opening the recommender from the workflow shell."""

    cppipe_path: str
    input_dir: str
    output_dir: str
    cellprofiler_executable: str = "cellprofiler"


def launch_threshold_recommender_window(
    parent: Any,
    context: ThresholdRecommenderLaunchContext,
) -> None:
    """Open the threshold recommender window."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    window = tk.Toplevel(parent)
    window.title("Threshold Recommender")
    window.geometry("980x640")
    window.minsize(820, 520)

    state: dict[str, ThresholdRecommenderTrialResult | None] = {"trial": None}
    image_names: list[str] = []

    main = ttk.Frame(window, padding=8)
    main.pack(fill="both", expand=True)
    main.columnconfigure(0, weight=1)
    main.rowconfigure(3, weight=1)

    paths_frame = ttk.LabelFrame(main, text="Workflow paths", padding=4)
    paths_frame.grid(row=0, column=0, sticky="ew")
    for row, (label, value) in enumerate(
        (
            ("Pipeline", context.cppipe_path),
            ("Full input folder", context.input_dir),
            ("Output folder", context.output_dir),
            ("CellProfiler", context.cellprofiler_executable),
        )
    ):
        ttk.Label(paths_frame, text=label).grid(row=row, column=0, sticky="w")
        ttk.Label(paths_frame, text=value, wraplength=760, justify="left").grid(
            row=row,
            column=1,
            sticky="w",
            padx=(8, 0),
        )

    subset_frame = ttk.LabelFrame(main, text="Subset trial", padding=4)
    subset_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
    subset_frame.columnconfigure(1, weight=1)

    ttk.Label(subset_frame, text="Subset count").grid(row=0, column=0, sticky="w")
    subset_count_var = tk.IntVar(value=5)
    ttk.Spinbox(
        subset_frame,
        from_=1,
        to=100,
        textvariable=subset_count_var,
        width=8,
    ).grid(row=0, column=1, sticky="w", padx=(8, 0))

    ttk.Label(subset_frame, text="Sample method").grid(row=0, column=2, sticky="w", padx=(16, 0))
    subset_method_var = tk.StringVar(value="even")
    ttk.Combobox(
        subset_frame,
        textvariable=subset_method_var,
        values=("even", "first", "random"),
        state="readonly",
        width=10,
    ).grid(row=0, column=3, sticky="w", padx=(8, 0))

    ttk.Label(
        subset_frame,
        text="Select specific images below for a manual subset, or leave unselected for auto sampling.",
        wraplength=900,
        justify="left",
    ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 4))

    fast_optimistic_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        subset_frame,
        text="Fast optimistic mode (try one Otsu adaptive candidate first; fall back to full search if QC fails)",
        variable=fast_optimistic_var,
    ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(0, 4))

    image_list = tk.Listbox(
        subset_frame,
        selectmode="extended",
        height=6,
        exportselection=False,
    )
    image_list.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=(0, 4))
    subset_frame.rowconfigure(3, weight=1)
    image_scroll = ttk.Scrollbar(subset_frame, orient="vertical", command=image_list.yview)
    image_scroll.grid(row=3, column=3, sticky="ns")
    image_list.configure(yscrollcommand=image_scroll.set)

    def refresh_image_list() -> None:
        nonlocal image_names
        image_list.delete(0, "end")
        image_names = [
            path.name for path in scan_detected_images(context.input_dir)
        ]
        for name in image_names:
            image_list.insert("end", name)
        status_var.set(f"Detected {len(image_names)} image(s) in input folder.")

    ttk.Button(subset_frame, text="Refresh images", command=refresh_image_list).grid(
        row=4,
        column=0,
        sticky="w",
    )

    content = ttk.Panedwindow(main, orient="vertical")
    content.grid(row=3, column=0, sticky="nsew", pady=(8, 0))

    ranking_frame = ttk.LabelFrame(content, text="Ranking", padding=4)
    preview_frame = ttk.LabelFrame(content, text="QC preview", padding=4)
    content.add(ranking_frame, weight=3)
    content.add(preview_frame, weight=2)

    columns = (
        "rank",
        "variant_id",
        "name",
        "score",
        "object_count",
        "tiny_frac",
        "huge_frac",
        "reason",
    )
    ranking_tree = ttk.Treeview(
        ranking_frame,
        columns=columns,
        show="headings",
        height=10,
    )
    headings = {
        "rank": "Rank",
        "variant_id": "Variant",
        "name": "Name",
        "score": "Score",
        "object_count": "Objects",
        "tiny_frac": "Tiny",
        "huge_frac": "Huge",
        "reason": "Reason",
    }
    for column in columns:
        ranking_tree.heading(column, text=headings[column])
        ranking_tree.column(column, width=90 if column != "reason" else 220, stretch=True)
    ranking_tree.pack(fill="both", expand=True, side="left")
    ranking_scroll = ttk.Scrollbar(
        ranking_frame,
        orient="vertical",
        command=ranking_tree.yview,
    )
    ranking_scroll.pack(side="right", fill="y")
    ranking_tree.configure(yscrollcommand=ranking_scroll.set)

    preview_label = ttk.Label(preview_frame, anchor="center", justify="center")
    preview_label.pack(fill="both", expand=True)

    actions = ttk.Frame(main)
    actions.grid(row=4, column=0, sticky="ew", pady=(8, 0))

    status_var = tk.StringVar(value="Ready.")
    ttk.Label(main, textvariable=status_var, relief="sunken", anchor="w").grid(
        row=5,
        column=0,
        sticky="ew",
        pady=(8, 0),
    )

    def populate_ranking(trial_result: ThresholdRecommenderTrialResult) -> None:
        ranking_tree.delete(*ranking_tree.get_children())
        for score in trial_result.ranked_scores:
            ranking_tree.insert(
                "",
                "end",
                iid=score.variant_id,
                values=(
                    score.rank,
                    score.variant_id,
                    score.display_name,
                    f"{score.score:.2f}",
                    score.object_count if score.object_count is not None else "",
                    f"{score.tiny_frac:.2f}" if score.tiny_frac is not None else "",
                    f"{score.huge_frac:.2f}" if score.huge_frac is not None else "",
                    score.reason,
                ),
            )

    def show_preview_for_variant(variant_id: str) -> None:
        from PIL import ImageTk

        trial_result = state["trial"]
        if trial_result is None:
            preview_label.configure(image="", text="Run a subset trial to preview variants.")
            preview_label.image = None  # type: ignore[attr-defined]
            return
        preview = trial_result.preview_index.get(variant_id)
        if preview is None or not preview.qc_preview_paths:
            preview_label.configure(
                image="",
                text=f"No QC preview available for {variant_id}.",
            )
            preview_label.image = None  # type: ignore[attr-defined]
            return
        target = preview.qc_preview_paths[0]
        try:
            photo = ImageTk.PhotoImage(load_preview_image(target))
        except Exception as exc:  # noqa: BLE001
            preview_label.configure(image="", text=f"Preview unavailable: {exc}")
            preview_label.image = None  # type: ignore[attr-defined]
            return
        preview_label.configure(
            image=photo,
            text=target.name,
            compound="top",
        )
        preview_label.image = photo  # type: ignore[attr-defined]

    def on_ranking_select(_event: object = None) -> None:
        selection = ranking_tree.selection()
        if not selection:
            return
        show_preview_for_variant(selection[0])

    ranking_tree.bind("<<TreeviewSelect>>", on_ranking_select)

    def _selected_variant_id() -> str | None:
        selection = ranking_tree.selection()
        if selection:
            return selection[0]
        trial_result = state["trial"]
        if trial_result and trial_result.ranked_scores:
            return trial_result.ranked_scores[0].variant_id
        return None

    def open_variant_folder() -> None:
        variant_id = _selected_variant_id()
        trial_result = state["trial"]
        if variant_id is None or trial_result is None:
            messagebox.showinfo("Open folder", "Run a subset trial and select a variant.")
            return
        preview = trial_result.preview_index.get(variant_id)
        if preview is None:
            messagebox.showinfo("Open folder", f"No folder recorded for {variant_id}.")
            return
        open_path(preview.variant_dir)

    def open_ranking_csv() -> None:
        trial_result = state["trial"]
        if trial_result is None:
            messagebox.showinfo("Open ranking", "Run a subset trial first.")
            return
        open_path(trial_result.ranking_paths["csv"])

    def run_trial_async() -> None:
        if not Path(context.cppipe_path).is_file():
            messagebox.showerror("Threshold Recommender", "Import a pipeline first.")
            return
        if not Path(context.input_dir).is_dir():
            messagebox.showerror("Threshold Recommender", "Input folder not found.")
            return

        selected_indices = list(image_list.curselection())
        manual_names = selected_manual_subset_names(image_names, selected_indices)
        config = build_recommender_config(
            imported_cppipe_path=context.cppipe_path,
            input_dir=context.input_dir,
            output_dir=context.output_dir,
            cellprofiler_executable=context.cellprofiler_executable,
            subset_count=subset_count_var.get(),
            subset_method=subset_method_var.get(),  # type: ignore[arg-type]
            manual_subset_image_names=manual_names,
            fast_optimistic=fast_optimistic_var.get(),
        )

        trial_button.configure(state="disabled")
        apply_button.configure(state="disabled")
        status_var.set("Running subset trial...")

        def worker() -> None:
            try:
                trial_result = run_threshold_recommender_trial(config)
            except Exception as exc:
                window.after(0, lambda exc=exc: _finish_trial_error(exc))
                return
            window.after(0, lambda trial_result=trial_result: _finish_trial_success(trial_result))

        threading.Thread(target=worker, daemon=True).start()

    def _trial_status_message(trial_result: ThresholdRecommenderTrialResult) -> str:
        base = (
            f"Subset trial complete: {len(trial_result.subset_manifest.image_names)} "
            f"image(s), ranking saved to {trial_result.ranking_paths['csv']}"
        )
        if trial_result.trial_mode == "optimistic" and trial_result.optimistic_qc is not None:
            return (
                f"{base} — optimistic candidate passed QC; you can apply it to the full dataset."
            )
        if trial_result.fell_back_to_full_search:
            reasons = ""
            if trial_result.optimistic_qc is not None and trial_result.optimistic_qc.reasons:
                reasons = f" Fallback reason: {trial_result.optimistic_qc.reasons[0]}"
            return f"{base} — optimistic QC failed; ran full variant search.{reasons}"
        return base

    def _finish_trial_success(trial_result: ThresholdRecommenderTrialResult) -> None:
        state["trial"] = trial_result
        populate_ranking(trial_result)
        if trial_result.ranked_scores:
            top_id = trial_result.ranked_scores[0].variant_id
            ranking_tree.selection_set(top_id)
            ranking_tree.focus(top_id)
            show_preview_for_variant(top_id)
        status_var.set(_trial_status_message(trial_result))
        trial_button.configure(state="normal")
        apply_button.configure(state="normal")
        if trial_result.trial_mode == "optimistic":
            messagebox.showinfo(
                "Optimistic candidate passed QC",
                (
                    "The fast optimistic Otsu adaptive candidate passed basic QC on the "
                    "subset.\n\nReview the preview and metrics, then apply to the full "
                    "dataset if it looks reasonable."
                ),
            )
        elif trial_result.fell_back_to_full_search:
            messagebox.showwarning(
                "Fell back to full variant search",
                (
                    "The optimistic candidate did not pass basic QC on the subset.\n\n"
                    "The recommender ran the full multi-variant threshold search instead."
                ),
            )

    def _finish_trial_error(exc: Exception) -> None:
        messagebox.showerror("Subset trial failed", str(exc))
        status_var.set("Subset trial failed.")
        trial_button.configure(state="normal")

    def apply_selected_async() -> None:
        variant_id = _selected_variant_id()
        if variant_id is None:
            messagebox.showinfo(
                "Apply variant",
                "Run a subset trial and select a variant to apply.",
            )
            return

        trial_result = state["trial"]
        display_name = variant_id
        if trial_result is not None:
            for score in trial_result.ranked_scores:
                if score.variant_id == variant_id:
                    display_name = score.display_name
                    break

        total_images = len(scan_detected_images(context.input_dir))
        proceed = messagebox.askyesno(
            "Apply to full dataset",
            (
                f"Apply variant {variant_id} ({display_name}) to the full input folder?\n\n"
                f"Full dataset: {total_images} image(s) in\n{context.input_dir}\n\n"
                "The imported .cppipe file will not be modified."
            ),
        )
        if not proceed:
            return

        config = build_recommender_config(
            imported_cppipe_path=context.cppipe_path,
            input_dir=context.input_dir,
            output_dir=context.output_dir,
            cellprofiler_executable=context.cellprofiler_executable,
            subset_count=subset_count_var.get(),
            subset_method=subset_method_var.get(),  # type: ignore[arg-type]
            fast_optimistic=fast_optimistic_var.get(),
        )

        apply_button.configure(state="disabled")
        status_var.set(f"Applying {variant_id} to full dataset...")

        def worker() -> None:
            try:
                apply_result = apply_confirmed_threshold_variant(
                    config,
                    variant_id,
                    confirmed=True,
                )
            except Exception as exc:
                window.after(0, lambda exc=exc: _finish_apply_error(exc))
                return
            window.after(0, lambda apply_result=apply_result: _finish_apply_success(apply_result))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_apply_success(apply_result: object) -> None:
        from bioimage_pipeline.threshold_recommender import ThresholdRecommenderApplyResult

        assert isinstance(apply_result, ThresholdRecommenderApplyResult)
        if apply_result.run_result.success:
            messagebox.showinfo(
                "Full dataset apply complete",
                f"Outputs written to:\n{apply_result.confirmed_run_dir}",
            )
            status_var.set(f"Confirmed full run complete for {apply_result.variant_id}.")
        else:
            messagebox.showerror(
                "Full dataset apply failed",
                apply_result.run_result.error_message or "Unknown error",
            )
            status_var.set("Confirmed full run failed.")
        apply_button.configure(state="normal")

    def _finish_apply_error(exc: Exception) -> None:
        messagebox.showerror("Apply failed", str(exc))
        status_var.set("Apply failed.")
        apply_button.configure(state="normal")

    def reload_existing_session() -> None:
        try:
            trial_result = load_trial_result_from_session(context.output_dir)
        except FileNotFoundError:
            return
        state["trial"] = trial_result
        populate_ranking(trial_result)
        status_var.set("Loaded existing recommender session.")

    trial_button = ttk.Button(actions, text="Run subset trial", command=run_trial_async)
    trial_button.pack(side="left")
    apply_button = ttk.Button(
        actions,
        text="Apply selected variant to full dataset…",
        command=apply_selected_async,
        state="disabled",
    )
    apply_button.pack(side="left", padx=(8, 0))
    ttk.Button(actions, text="Open variant folder", command=open_variant_folder).pack(
        side="left",
        padx=(8, 0),
    )
    ttk.Button(actions, text="Open ranking CSV", command=open_ranking_csv).pack(
        side="left",
        padx=(8, 0),
    )

    refresh_image_list()
    reload_existing_session()
