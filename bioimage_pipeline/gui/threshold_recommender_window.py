"""Separate Tk window for CellProfiler threshold variant QC assistant trials."""

from __future__ import annotations

import threading
from collections.abc import Callable
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
from bioimage_pipeline.cellprofiler_runner import RESULTS_RAW_DIR
from bioimage_pipeline.ground_truth import load_ground_truth_manifest, load_reference_mask
from bioimage_pipeline.threshold_recommender_previews import resolve_compare_preview_path
from bioimage_pipeline.threshold_variant_gt_scoring import resolve_predicted_mask_path


@dataclass(frozen=True)
class ThresholdRecommenderLaunchContext:
    """Prefilled values when opening the recommender from the workflow shell."""

    cppipe_path: str
    input_dir: str
    output_dir: str
    cellprofiler_executable: str = "cellprofiler"


_RESET_HANDLERS: list[Callable[[], None]] = []


def reset_open_threshold_recommender_windows() -> None:
    """Clear trial display and reference-mask state in open recommender windows."""
    for handler in list(_RESET_HANDLERS):
        handler()


def _register_threshold_recommender_reset_handler(
    handler: Callable[[], None],
) -> Callable[[], None]:
    """Register a reset handler and return an unregister callback."""
    _RESET_HANDLERS.append(handler)

    def unregister() -> None:
        try:
            _RESET_HANDLERS.remove(handler)
        except ValueError:
            pass

    return unregister


def launch_threshold_recommender_window(
    parent: Any,
    context: ThresholdRecommenderLaunchContext,
) -> None:
    """Open the threshold recommender window."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    window = tk.Toplevel(parent)
    window.title("CellProfiler Threshold Variant QC Assistant")
    window.geometry("1100x780")
    window.minsize(900, 620)

    state: dict[str, ThresholdRecommenderTrialResult | None] = {"trial": None}
    image_names: list[str] = []

    def window_is_open() -> bool:
        try:
            return bool(window.winfo_exists())
        except tk.TclError:
            return False

    main = ttk.Frame(window, padding=8)
    main.pack(fill="both", expand=True)
    main.columnconfigure(0, weight=1)
    main.rowconfigure(4, weight=1)

    disclaimer = ttk.Label(
        main,
        text=(
            "Rankings are heuristic screening aids, not biological ground truth. "
            "When reference masks are provided, GT metrics reflect agreement with "
            "lab annotations. Review previews and per-image QC, then choose a "
            "candidate before applying."
        ),
        wraplength=1040,
        justify="left",
    )
    disclaimer.grid(row=0, column=0, sticky="ew", pady=(0, 4))

    paths_frame = ttk.LabelFrame(main, text="Workflow paths", padding=4)
    paths_frame.grid(row=1, column=0, sticky="ew")
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
    subset_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
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

    force_full_search_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        subset_frame,
        text="Always run full variant search (recommended; do not accept optimistic candidate even if screening passes)",
        variable=force_full_search_var,
    ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(0, 4))

    ttk.Label(subset_frame, text="Reference masks folder").grid(
        row=4, column=0, sticky="w"
    )
    reference_masks_var = tk.StringVar(value="")
    ttk.Entry(subset_frame, textvariable=reference_masks_var).grid(
        row=4,
        column=1,
        sticky="ew",
        padx=(8, 8),
        pady=(0, 4),
    )

    def browse_reference_masks_folder() -> None:
        from tkinter import filedialog

        selected = filedialog.askdirectory()
        if selected:
            reference_masks_var.set(selected)

    ttk.Button(
        subset_frame,
        text="Browse",
        command=browse_reference_masks_folder,
    ).grid(row=4, column=2, sticky="w", pady=(0, 4))

    image_list = tk.Listbox(
        subset_frame,
        selectmode="extended",
        height=6,
        exportselection=False,
    )
    image_list.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(0, 4))
    subset_frame.rowconfigure(5, weight=1)
    image_scroll = ttk.Scrollbar(subset_frame, orient="vertical", command=image_list.yview)
    image_scroll.grid(row=5, column=3, sticky="ns")
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
        row=6,
        column=0,
        sticky="w",
    )

    content = ttk.Panedwindow(main, orient="vertical")
    content.grid(row=4, column=0, sticky="nsew", pady=(8, 0))

    upper_content = ttk.Panedwindow(content, orient="horizontal")
    content.add(upper_content, weight=3)

    screening_frame = ttk.LabelFrame(upper_content, text="Candidate screening", padding=4)
    per_image_frame = ttk.LabelFrame(upper_content, text="Per-image QC", padding=4)
    upper_content.add(screening_frame, weight=3)
    upper_content.add(per_image_frame, weight=2)

    compare_frame = ttk.LabelFrame(content, text="Compare candidates", padding=4)
    preview_frame = ttk.LabelFrame(content, text="QC preview", padding=4)
    content.add(compare_frame, weight=1)
    content.add(preview_frame, weight=2)

    columns = (
        "rank",
        "screening_label",
        "gt_label",
        "gt_f1",
        "gt_dice",
        "gt_count_error",
        "variant_id",
        "name",
        "score",
        "object_count",
        "count_ratio",
        "tiny_frac",
        "huge_frac",
        "reason",
    )
    ranking_tree = ttk.Treeview(
        screening_frame,
        columns=columns,
        show="headings",
        height=8,
    )
    headings = {
        "rank": "Rank",
        "screening_label": "Screen",
        "gt_label": "GT",
        "gt_f1": "GT F1",
        "gt_dice": "GT Dice",
        "gt_count_error": "GT Δcount",
        "variant_id": "Variant",
        "name": "Name",
        "score": "Score",
        "object_count": "Objects",
        "count_ratio": "Count/baseline",
        "tiny_frac": "Tiny",
        "huge_frac": "Huge",
        "reason": "Reason",
    }
    for column in columns:
        ranking_tree.heading(column, text=headings[column])
        ranking_tree.column(column, width=90 if column != "reason" else 220, stretch=True)
    ranking_tree.pack(fill="both", expand=True, side="left")
    ranking_scroll = ttk.Scrollbar(
        screening_frame,
        orient="vertical",
        command=ranking_tree.yview,
    )
    ranking_scroll.pack(side="right", fill="y")
    ranking_tree.configure(yscrollcommand=ranking_scroll.set)

    per_image_columns = (
        "image_name",
        "object_count",
        "count_ratio",
        "foreground_coverage",
        "warnings",
    )
    per_image_tree = ttk.Treeview(
        per_image_frame,
        columns=per_image_columns,
        show="headings",
        height=8,
    )
    for column, label in (
        ("image_name", "Image"),
        ("object_count", "Objects"),
        ("count_ratio", "Count/baseline"),
        ("foreground_coverage", "Coverage"),
        ("warnings", "Warnings"),
    ):
        per_image_tree.heading(column, text=label)
        per_image_tree.column(column, width=120 if column != "warnings" else 220, stretch=True)
    per_image_tree.pack(fill="both", expand=True, side="left")
    per_image_scroll = ttk.Scrollbar(
        per_image_frame,
        orient="vertical",
        command=per_image_tree.yview,
    )
    per_image_scroll.pack(side="right", fill="y")
    per_image_tree.configure(yscrollcommand=per_image_scroll.set)

    compare_controls = ttk.Frame(compare_frame)
    compare_controls.pack(fill="x")
    compare_image_var = tk.StringVar(value="")
    compare_variant_a_var = tk.StringVar(value="")
    compare_variant_b_var = tk.StringVar(value="")
    ttk.Label(compare_controls, text="Image").grid(row=0, column=0, sticky="w")
    compare_image_combo = ttk.Combobox(
        compare_controls,
        textvariable=compare_image_var,
        state="readonly",
        width=36,
    )
    compare_image_combo.grid(row=0, column=1, sticky="w", padx=(8, 16))
    ttk.Label(compare_controls, text="Variant A").grid(row=0, column=2, sticky="w")
    compare_variant_a_combo = ttk.Combobox(
        compare_controls,
        textvariable=compare_variant_a_var,
        state="readonly",
        width=28,
    )
    compare_variant_a_combo.grid(row=0, column=3, sticky="w", padx=(8, 16))
    ttk.Label(compare_controls, text="Variant B").grid(row=0, column=4, sticky="w")
    compare_variant_b_combo = ttk.Combobox(
        compare_controls,
        textvariable=compare_variant_b_var,
        state="readonly",
        width=28,
    )
    compare_variant_b_combo.grid(row=0, column=5, sticky="w", padx=(8, 0))

    compare_grid = ttk.Frame(compare_frame)
    compare_grid.pack(fill="both", expand=True, pady=(8, 0))
    compare_labels: dict[str, ttk.Label] = {}
    compare_titles = (
        "Original",
        "Variant A overlay",
        "Variant B overlay",
        "Reference (GT)",
    )
    for column, title in enumerate(compare_titles):
        ttk.Label(compare_grid, text=title).grid(row=0, column=column, pady=(0, 4))
        label = ttk.Label(compare_grid, anchor="center", justify="center")
        label.grid(row=1, column=column, sticky="nsew", padx=4)
        compare_labels[title] = label
    for column in range(len(compare_titles)):
        compare_grid.columnconfigure(column, weight=1)

    preview_label = ttk.Label(preview_frame, anchor="center", justify="center")
    preview_label.pack(fill="both", expand=True)

    actions = ttk.Frame(main)
    actions.grid(row=5, column=0, sticky="ew", pady=(8, 0))

    status_var = tk.StringVar(value="Ready.")
    ttk.Label(main, textvariable=status_var, relief="sunken", anchor="w").grid(
        row=6,
        column=0,
        sticky="ew",
        pady=(8, 0),
    )

    def populate_ranking(trial_result: ThresholdRecommenderTrialResult) -> None:
        ranking_tree.delete(*ranking_tree.get_children())
        gt_by_variant = {
            score.variant_id: score for score in trial_result.gt_ranked_scores
        }
        for score in trial_result.ranked_scores:
            gt_score = gt_by_variant.get(score.variant_id)
            ranking_tree.insert(
                "",
                "end",
                iid=score.variant_id,
                values=(
                    score.rank,
                    score.screening_label,
                    gt_score.gt_label if gt_score is not None else "",
                    f"{gt_score.mean_f1:.2f}" if gt_score and gt_score.mean_f1 is not None else "",
                    f"{gt_score.mean_dice:.2f}" if gt_score and gt_score.mean_dice is not None else "",
                    (
                        f"{gt_score.mean_count_error:.0f}"
                        if gt_score and gt_score.mean_count_error is not None
                        else ""
                    ),
                    score.variant_id,
                    score.display_name,
                    f"{score.score:.2f}",
                    score.object_count if score.object_count is not None else "",
                    (
                        f"{score.object_count_ratio_vs_baseline:.1f}x"
                        if score.object_count_ratio_vs_baseline is not None
                        else ""
                    ),
                    f"{score.tiny_frac:.2f}" if score.tiny_frac is not None else "",
                    f"{score.huge_frac:.2f}" if score.huge_frac is not None else "",
                    score.reason,
                ),
            )

        image_options = list(trial_result.subset_manifest.image_names)
        variant_options = [score.variant_id for score in trial_result.ranked_scores]
        compare_image_combo.configure(values=image_options)
        compare_variant_a_combo.configure(values=variant_options)
        compare_variant_b_combo.configure(values=variant_options)
        if image_options:
            compare_image_var.set(image_options[0])
        if variant_options:
            compare_variant_a_var.set(
                next(
                    (item for item in variant_options if "baseline" in item),
                    variant_options[0],
                )
            )
            compare_variant_b_var.set(
                variant_options[1] if len(variant_options) > 1 else variant_options[0]
            )
        if trial_result.gt_ranked_scores:
            top_gt = trial_result.gt_ranked_scores[0]
            if top_gt.variant_id in variant_options:
                compare_variant_a_var.set(top_gt.variant_id)

    def populate_per_image_for_variant(variant_id: str) -> None:
        per_image_tree.delete(*per_image_tree.get_children())
        trial_result = state["trial"]
        if trial_result is None:
            return
        for row in trial_result.per_image_summaries:
            if row.variant_id != variant_id:
                continue
            per_image_tree.insert(
                "",
                "end",
                values=(
                    row.image_name or f"image {row.image_number}",
                    row.object_count if row.object_count is not None else "",
                    (
                        f"{row.object_count_ratio_vs_baseline:.1f}x"
                        if row.object_count_ratio_vs_baseline is not None
                        else ""
                    ),
                    (
                        f"{row.foreground_coverage:.1%}"
                        if row.foreground_coverage is not None
                        else ""
                    ),
                    "; ".join(row.warnings),
                ),
            )

    def show_compare_previews() -> None:
        from PIL import Image, ImageTk

        from bioimage_pipeline.qc import create_fp_fn_overlay, create_mask_overlay, normalize_for_display
        from bioimage_pipeline.io import read_tiff

        trial_result = state["trial"]
        if trial_result is None:
            return

        image_name = compare_image_var.get()
        variant_a = compare_variant_a_var.get()
        variant_b = compare_variant_b_var.get()
        if not image_name or not variant_a or not variant_b:
            return

        image_path = trial_result.subset_dir / image_name
        panels: dict[str, Path | None | str] = {
            "Original": image_path if image_path.is_file() else None,
            "Variant A overlay": None,
            "Variant B overlay": None,
            "Reference (GT)": None,
        }
        variant_dirs: dict[str, Path] = {}
        for variant_id, title in (
            (variant_a, "Variant A overlay"),
            (variant_b, "Variant B overlay"),
        ):
            preview = trial_result.preview_index.get(variant_id)
            if preview is None:
                continue
            variant_dirs[variant_id] = preview.variant_dir
            panels[title] = resolve_compare_preview_path(
                preview.variant_dir,
                image_name,
                prefer="mask",
            )

        reference_mask = None
        if trial_result.ground_truth_manifest_path is not None:
            try:
                manifest = load_ground_truth_manifest(trial_result.ground_truth_manifest_path)
                for entry in manifest.entries:
                    if entry.image_name == image_name:
                        reference_mask = load_reference_mask(entry.reference_mask_path)
                        break
            except (OSError, ValueError):
                reference_mask = None

        for title, label in compare_labels.items():
            target = panels.get(title)
            if title == "Reference (GT)" and reference_mask is not None and image_path.is_file():
                try:
                    image_array = read_tiff(image_path)
                    preview_a = trial_result.preview_index.get(variant_a)
                    predicted_mask = None
                    if preview_a is not None:
                        predicted_path = resolve_predicted_mask_path(
                            preview_a.variant_dir / RESULTS_RAW_DIR,
                            image_name,
                        )
                        if predicted_path is not None:
                            predicted_mask = load_reference_mask(predicted_path)
                    if predicted_mask is not None:
                        overlay = create_fp_fn_overlay(
                            image_array,
                            predicted_mask,
                            reference_mask,
                        )
                    else:
                        overlay = create_mask_overlay(
                            image_array,
                            reference_mask,
                            color=(0, 220, 0),
                        )
                    photo = ImageTk.PhotoImage(Image.fromarray(overlay))
                    label.configure(image=photo, text=image_name, compound="top")
                    label.image = photo  # type: ignore[attr-defined]
                except Exception as exc:  # noqa: BLE001
                    label.configure(image="", text=f"Reference preview unavailable: {exc}")
                    label.image = None  # type: ignore[attr-defined]
                continue

            if target is None or not isinstance(target, Path) or not target.is_file():
                label.configure(image="", text=f"No preview for {title}.")
                label.image = None  # type: ignore[attr-defined]
                continue
            try:
                if title == "Original":
                    array = normalize_for_display(read_tiff(target))
                    photo = ImageTk.PhotoImage(Image.fromarray(array))
                else:
                    photo = ImageTk.PhotoImage(load_preview_image(target))
            except Exception as exc:  # noqa: BLE001
                label.configure(image="", text=f"Preview unavailable: {exc}")
                label.image = None  # type: ignore[attr-defined]
                continue
            label.configure(image=photo, text=target.name, compound="top")
            label.image = photo  # type: ignore[attr-defined]

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
                text=(
                    f"No QC preview available for {variant_id}.\n\n"
                    "Ensure the pipeline exports mask/label images (SaveImages) "
                    "so overlay previews can be generated."
                ),
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
        variant_id = selection[0]
        show_preview_for_variant(variant_id)
        populate_per_image_for_variant(variant_id)

    ranking_tree.bind("<<TreeviewSelect>>", on_ranking_select)

    def on_compare_selection_changed(_event: object = None) -> None:
        show_compare_previews()

    compare_image_combo.bind("<<ComboboxSelected>>", on_compare_selection_changed)
    compare_variant_a_combo.bind("<<ComboboxSelected>>", on_compare_selection_changed)
    compare_variant_b_combo.bind("<<ComboboxSelected>>", on_compare_selection_changed)

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
            messagebox.showerror("Threshold Variant QC Assistant", "Import a pipeline first.")
            return
        if not Path(context.input_dir).is_dir():
            messagebox.showerror("Threshold Variant QC Assistant", "Input folder not found.")
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
            force_full_search=force_full_search_var.get(),
            reference_mask_dir=reference_masks_var.get().strip() or None,
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
            ratio_note = ""
            optimistic_qc = trial_result.optimistic_qc
            if optimistic_qc.object_count_ratio_vs_baseline is not None:
                ratio_note = (
                    f" (object_count {optimistic_qc.object_count_ratio_vs_baseline:.1f}x baseline)"
                )
            return (
                f"{base} — optimistic candidate passed basic screening{ratio_note}; "
                "review previews before applying to the full dataset."
            )
        if trial_result.forced_full_search:
            return (
                f"{base} — full variant search was forced despite optimistic QC pass."
            )
        if trial_result.fell_back_to_full_search:
            reasons = ""
            if trial_result.optimistic_qc is not None and trial_result.optimistic_qc.reasons:
                reasons = f" Fallback reason: {trial_result.optimistic_qc.reasons[0]}"
            return f"{base} — optimistic screening failed; ran full variant search.{reasons}"
        return base

    def _finish_trial_success(trial_result: ThresholdRecommenderTrialResult) -> None:
        if not window_is_open():
            return
        state["trial"] = trial_result
        populate_ranking(trial_result)
        if trial_result.ranked_scores:
            top_id = trial_result.ranked_scores[0].variant_id
            ranking_tree.selection_set(top_id)
            ranking_tree.focus(top_id)
            show_preview_for_variant(top_id)
            populate_per_image_for_variant(top_id)
            show_compare_previews()
        char_note = ""
        if trial_result.subset_characterization_paths.get("csv"):
            char_note = (
                f"\n\nSubset characterization: "
                f"{trial_result.subset_characterization_paths['csv']}"
            )
        if trial_result.gt_ranked_scores:
            char_note += (
                f"\n\nGround-truth ranking: "
                f"{trial_result.ground_truth_comparison_paths.get('ranking_csv', '')}"
            )
        status_var.set(_trial_status_message(trial_result))
        trial_button.configure(state="normal")
        apply_button.configure(state="normal")
        if trial_result.trial_mode == "optimistic":
            optimistic_qc = trial_result.optimistic_qc
            warning_lines = ""
            if optimistic_qc is not None and optimistic_qc.warnings:
                warning_lines = "\n\nWarnings:\n" + "\n".join(
                    f"- {line}" for line in optimistic_qc.warnings
                )
            messagebox.showinfo(
                "Optimistic candidate passed basic screening",
                (
                    "The fast optimistic Otsu adaptive candidate passed basic heuristic "
                    "screening on the subset.\n\nReview previews, per-image QC, and "
                    "compare candidates before applying to the full dataset."
                    f"{warning_lines}{char_note}"
                ),
            )
        elif trial_result.forced_full_search:
            messagebox.showinfo(
                "Full variant search completed",
                (
                    "The optimistic candidate passed basic screening, but full variant "
                    "search was requested.\n\nReview the full candidate table and "
                    "compare previews before applying a variant."
                    f"{char_note}"
                ),
            )
        elif trial_result.fell_back_to_full_search:
            messagebox.showwarning(
                "Fell back to full variant search",
                (
                    "The optimistic candidate did not pass basic screening on the subset.\n\n"
                    "The assistant ran the full multi-variant threshold search instead."
                    f"{char_note}"
                ),
            )

    def _finish_trial_error(exc: Exception) -> None:
        if not window_is_open():
            return
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
            force_full_search=force_full_search_var.get(),
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
        if not window_is_open():
            return
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
        if not window_is_open():
            return
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

    def clear_session_state() -> None:
        if not window_is_open():
            return
        state["trial"] = None
        ranking_tree.delete(*ranking_tree.get_children())
        per_image_tree.delete(*per_image_tree.get_children())
        reference_masks_var.set("")
        compare_image_var.set("")
        compare_variant_a_var.set("")
        compare_variant_b_var.set("")
        compare_image_combo.configure(values=[])
        compare_variant_a_combo.configure(values=[])
        compare_variant_b_combo.configure(values=[])
        for label in compare_labels.values():
            label.configure(image="", text="")
            label.image = None  # type: ignore[attr-defined]
        preview_label.configure(
            image="",
            text="Run a subset trial to preview variants.",
        )
        preview_label.image = None  # type: ignore[attr-defined]
        apply_button.configure(state="disabled")
        status_var.set(
            "Session cleared from main workflow. Close this window and reopen "
            "after configuring new paths.",
        )

    unregister_reset_handler = _register_threshold_recommender_reset_handler(
        clear_session_state,
    )
    window.protocol("WM_DELETE_WINDOW", lambda: (unregister_reset_handler(), window.destroy()))

    refresh_image_list()
    reload_existing_session()
