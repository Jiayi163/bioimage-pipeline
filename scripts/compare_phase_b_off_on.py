#!/usr/bin/env python3
"""Compare Phase B OFF vs ON puncta declump results object-by-object.

Analysis only: reads exported CSVs and reports meaningful final-output changes.
Does not modify fitting behavior or thresholds.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


DEFAULT_OFF_DIR = Path(r"C:\Users\Administrator\Desktop\example1\phase_b_off")
DEFAULT_ON_DIR = Path(r"C:\Users\Administrator\Desktop\example1\phase_b_on_true")
DEFAULT_OUTPUT = Path("phase_b_off_vs_on_comparison.csv")
WATCH_OBJECT_IDS = (260, 337, 574, 642, 1140)


@dataclass(frozen=True)
class ObjectSummary:
    object_id: int
    accepted_count: int
    centers: list[tuple[float, float]]
    best_gmm_r_squared: float | None
    best_gmm_residual: float | None
    selected_model_order: int | None
    model_selection_reason: str
    phase_b_triggered: bool
    residual_split_transition: str


def find_csv(folder: Path, suffix: str) -> Path:
    matches = sorted(folder.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No *{suffix} found in {folder}")
    if len(matches) > 1:
        # Prefer the shortest stem (usually the main export, not cache copies).
        matches.sort(key=lambda path: (len(path.name), path.name))
    return matches[0]


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "t"}


def _parse_selected_model_order(reason: str) -> int | None:
    if not reason:
        return None
    match = re.search(r"selected_gmm_n=(\d+)", reason)
    if match:
        return int(match.group(1))
    lowered = reason.lower()
    if "kept_single" in lowered or "fast_path" in lowered:
        return 1
    return None


def _parse_residual_split_transition(reason: str) -> str:
    match = re.search(r"residual_split_applied_n=(\d+)->(\d+)", reason)
    if match:
        return f"{match.group(1)} -> {match.group(2)}"
    return ""


def _phase_b_triggered(reason: str) -> bool:
    return "residual_split" in reason.lower()


def _format_centers(centers: list[tuple[float, float]]) -> str:
    if not centers:
        return ""
    return "; ".join(f"({row:.2f},{col:.2f})" for row, col in centers)


def load_object_summaries(folder: Path) -> dict[int, ObjectSummary]:
    measurements_path = find_csv(folder, "_measurements.csv")
    diagnostics_path = find_csv(folder, "_object_diagnostics.csv")

    measurements = pd.read_csv(measurements_path)
    diagnostics = pd.read_csv(diagnostics_path)

    required_measurement_cols = {"object_id", "final_row", "final_col", "accepted"}
    missing = required_measurement_cols - set(measurements.columns)
    if missing:
        raise ValueError(f"{measurements_path} missing columns: {sorted(missing)}")

    accepted_rows = measurements[measurements["accepted"].map(_to_bool)].copy()
    accepted_rows = accepted_rows.dropna(subset=["final_row", "final_col"])

    centers_by_object: dict[int, list[tuple[float, float]]] = {}
    for object_id, group in accepted_rows.groupby("object_id"):
        centers = [
            (float(row), float(col))
            for row, col in zip(group["final_row"], group["final_col"], strict=True)
        ]
        centers_by_object[int(object_id)] = centers

    summaries: dict[int, ObjectSummary] = {}
    for _, row in diagnostics.iterrows():
        object_id = int(row["object_id"])
        reason = "" if pd.isna(row.get("model_selection_reason")) else str(row["model_selection_reason"])
        accepted_count = int(row.get("n_accepted_fit_ok", len(centers_by_object.get(object_id, []))) or 0)
        centers = centers_by_object.get(object_id, [])
        if accepted_count != len(centers):
            # Trust accepted measurement rows as the final accepted output.
            accepted_count = len(centers)

        best_r2 = row.get("best_gmm_r_squared")
        best_residual = row.get("best_gmm_residual_relative")
        summaries[object_id] = ObjectSummary(
            object_id=object_id,
            accepted_count=accepted_count,
            centers=centers,
            best_gmm_r_squared=None if pd.isna(best_r2) else float(best_r2),
            best_gmm_residual=None if pd.isna(best_residual) else float(best_residual),
            selected_model_order=_parse_selected_model_order(reason),
            model_selection_reason=reason,
            phase_b_triggered=_phase_b_triggered(reason),
            residual_split_transition=_parse_residual_split_transition(reason),
        )

    # Include objects present only in measurements (should be rare).
    for object_id, centers in centers_by_object.items():
        if object_id not in summaries:
            summaries[object_id] = ObjectSummary(
                object_id=object_id,
                accepted_count=len(centers),
                centers=centers,
                best_gmm_r_squared=None,
                best_gmm_residual=None,
                selected_model_order=None,
                model_selection_reason="",
                phase_b_triggered=False,
                residual_split_transition="",
            )

    return summaries


def match_center_displacements(
    off_centers: list[tuple[float, float]],
    on_centers: list[tuple[float, float]],
) -> tuple[float, float, list[float]]:
    """Return max, mean, and per-match displacements using Hungarian matching."""
    if not off_centers and not on_centers:
        return 0.0, 0.0, []
    if not off_centers or not on_centers:
        unmatched = max(len(off_centers), len(on_centers))
        return float("inf"), float("inf"), [float("inf")] * unmatched

    off = np.asarray(off_centers, dtype=float)
    on = np.asarray(on_centers, dtype=float)
    cost = np.linalg.norm(off[:, None, :] - on[None, :, :], axis=2)
    row_idx, col_idx = linear_sum_assignment(cost)

    matched = [float(cost[r, c]) for r, c in zip(row_idx, col_idx, strict=True)]
    unmatched_penalty = abs(len(off_centers) - len(on_centers))
    if unmatched_penalty:
        matched.extend([float("inf")] * unmatched_penalty)

    finite = [value for value in matched if np.isfinite(value)]
    if not finite:
        return float("inf"), float("inf"), matched

    return max(finite), float(np.mean(finite)), matched


def build_comparison_rows(
    off_summaries: dict[int, ObjectSummary],
    on_summaries: dict[int, ObjectSummary],
) -> pd.DataFrame:
    object_ids = sorted(set(off_summaries) | set(on_summaries))
    rows: list[dict[str, object]] = []

    for object_id in object_ids:
        off = off_summaries.get(object_id)
        on = on_summaries.get(object_id)

        off_count = off.accepted_count if off else 0
        on_count = on.accepted_count if on else 0
        off_centers = off.centers if off else []
        on_centers = on.centers if on else []

        max_disp, mean_disp, _ = match_center_displacements(off_centers, on_centers)
        count_changed = off_count != on_count

        off_r2 = off.best_gmm_r_squared if off else None
        on_r2 = on.best_gmm_r_squared if on else None
        off_res = off.best_gmm_residual if off else None
        on_res = on.best_gmm_residual if on else None

        r2_delta = None
        residual_delta = None
        metrics_improved = False
        if off_r2 is not None and on_r2 is not None:
            r2_delta = on_r2 - off_r2
        if off_res is not None and on_res is not None:
            residual_delta = off_res - on_res
        if r2_delta is not None and residual_delta is not None:
            metrics_improved = (r2_delta > 1e-6) or (residual_delta > 1e-6)

        final_output_changed = (
            count_changed
            or (np.isfinite(max_disp) and max_disp > 1e-6)
            or _format_centers(off_centers) != _format_centers(on_centers)
        )
        metrics_improved_no_output_change = metrics_improved and not final_output_changed

        interest_score = 0
        if count_changed:
            interest_score += 1000
        if np.isfinite(max_disp) and max_disp > 1.0:
            interest_score += 500
        elif np.isfinite(max_disp) and max_disp > 0.5:
            interest_score += 200
        if r2_delta is not None and r2_delta > 0.05:
            interest_score += 100
        if residual_delta is not None and residual_delta > 0.05:
            interest_score += 100
        if on and on.phase_b_triggered:
            interest_score += 50

        rows.append(
            {
                "object_id": object_id,
                "off_accepted_component_count": off_count,
                "on_accepted_component_count": on_count,
                "accepted_component_count_changed": count_changed,
                "off_final_accepted_centers": _format_centers(off_centers),
                "on_final_accepted_centers": _format_centers(on_centers),
                "nearest_center_displacement_px": max_disp,
                "max_matched_center_displacement_px": max_disp,
                "mean_matched_center_displacement_px": mean_disp,
                "off_best_gmm_r_squared": off_r2,
                "on_best_gmm_r_squared": on_r2,
                "delta_best_gmm_r_squared": r2_delta,
                "off_best_gmm_residual": off_res,
                "on_best_gmm_residual": on_res,
                "delta_best_gmm_residual": residual_delta,
                "off_selected_model_order": off.selected_model_order if off else None,
                "on_selected_model_order": on.selected_model_order if on else None,
                "on_phase_b_triggered": on.phase_b_triggered if on else False,
                "on_residual_split_transition": on.residual_split_transition if on else "",
                "off_model_selection_reason": off.model_selection_reason if off else "",
                "on_model_selection_reason": on.model_selection_reason if on else "",
                "final_output_changed": final_output_changed,
                "metrics_improved_no_output_change": metrics_improved_no_output_change,
                "interest_score": interest_score,
            }
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["interest_score", "object_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    return frame


def print_summary(frame: pd.DataFrame) -> None:
    total = len(frame)
    phase_b_triggered = int(frame["on_phase_b_triggered"].sum())
    count_changed = int(frame["accepted_component_count_changed"].sum())
    moved_gt_0_5 = int((frame["nearest_center_displacement_px"] > 0.5).sum())
    moved_gt_1_0 = int((frame["nearest_center_displacement_px"] > 1.0).sum())
    metrics_no_output = int(frame["metrics_improved_no_output_change"].sum())

    print("\n=== Phase B OFF vs ON Summary ===")
    print(f"total objects compared: {total}")
    print(f"objects where Phase B triggered (ON): {phase_b_triggered}")
    print(f"objects where accepted component count changed: {count_changed}")
    print(f"objects where centers moved > 0.5 px: {moved_gt_0_5}")
    print(f"objects where centers moved > 1.0 px: {moved_gt_1_0}")
    print(f"objects where metrics improved but final output unchanged: {metrics_no_output}")

    interesting = frame[frame["interest_score"] > 0].head(20)
    print("\n=== Top objects to inspect visually ===")
    if interesting.empty:
        print("(none)")
        return

    for _, row in interesting.iterrows():
        print(
            f"object {int(row['object_id']):4d} | score={int(row['interest_score']):4d} | "
            f"count {int(row['off_accepted_component_count'])}->{int(row['on_accepted_component_count'])} | "
            f"max_disp={row['nearest_center_displacement_px']:.3f}px | "
            f"phase_b={bool(row['on_phase_b_triggered'])} | "
            f"transition={row['on_residual_split_transition'] or '-'}"
        )


def print_watch_objects(frame: pd.DataFrame, watch_ids: tuple[int, ...]) -> None:
    print("\n=== Watch objects ===")
    for object_id in watch_ids:
        match = frame[frame["object_id"] == object_id]
        if match.empty:
            print(f"object {object_id}: not found in comparison")
            continue
        row = match.iloc[0]
        print(
            f"object {object_id}: "
            f"count {int(row['off_accepted_component_count'])}->{int(row['on_accepted_component_count'])} | "
            f"max_disp={row['nearest_center_displacement_px']:.3f}px | "
            f"phase_b={bool(row['on_phase_b_triggered'])} | "
            f"transition={row['on_residual_split_transition'] or '-'} | "
            f"off_centers={row['off_final_accepted_centers']} | "
            f"on_centers={row['on_final_accepted_centers']}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Phase B OFF vs ON puncta declump outputs object-by-object.",
    )
    parser.add_argument("--off-dir", type=Path, default=DEFAULT_OFF_DIR)
    parser.add_argument("--on-dir", type=Path, default=DEFAULT_ON_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--watch-object-ids",
        type=int,
        nargs="*",
        default=list(WATCH_OBJECT_IDS),
        help="Object IDs to print explicitly.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.off_dir.is_dir():
        print(f"ERROR: OFF results folder not found: {args.off_dir}", file=sys.stderr)
        return 1
    if not args.on_dir.is_dir():
        print(f"ERROR: ON results folder not found: {args.on_dir}", file=sys.stderr)
        return 1

    off_summaries = load_object_summaries(args.off_dir)
    on_summaries = load_object_summaries(args.on_dir)
    comparison = build_comparison_rows(off_summaries, on_summaries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False)

    print(f"Wrote comparison CSV: {args.output.resolve()}")
    print_summary(comparison)
    print_watch_objects(comparison, tuple(args.watch_object_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
