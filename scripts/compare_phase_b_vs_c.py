#!/usr/bin/env python3
"""Compare Phase B vs Phase C puncta declump results object-by-object."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


DEFAULT_B_DIR = Path(r"C:\Users\Administrator\Desktop\example2\phase_b")
DEFAULT_C_DIR = Path(r"C:\Users\Administrator\Desktop\example2\phase_c")
DEFAULT_OUTPUT = Path(r"C:\Users\Administrator\Desktop\example2\phase_b_vs_c_comparison.csv")


@dataclass(frozen=True)
class ObjectSummary:
    object_id: int
    accepted_count: int
    centers: list[tuple[float, float]]
    best_gmm_r_squared: float | None
    best_gmm_residual: float | None
    selected_model_order: int | None
    model_selection_reason: str
    residual_split_triggered: bool
    residual_split_transition: str
    ambiguous: bool
    stop_reason: str
    split_attempts: int | None


def find_csv(folder: Path, suffix: str) -> Path:
    matches = sorted(folder.glob(f"*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No *{suffix} found in {folder}")
    matches.sort(key=lambda path: (len(path.name), path.name))
    return matches[0]


def load_summary(folder: Path) -> dict[str, object]:
    summary_path = next(iter(sorted(folder.glob("*_summary.json"))), None)
    if summary_path is None:
        return {}
    with summary_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("summary", payload)


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "t"}


def _parse_selected_model_order(reason: str) -> int | None:
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


def _parse_split_attempts(reason: str) -> int | None:
    match = re.search(r"attempts=(\d+)", reason)
    return int(match.group(1)) if match else None


def _parse_ambiguous(reason: str) -> bool:
    return "_ambiguous=" in reason.lower()


def _parse_stop_reason(reason: str) -> str:
    ambiguous = re.search(r"_ambiguous=([^;_]+)", reason)
    if ambiguous:
        return ambiguous.group(1)
    stop = re.search(r"_stop=([^;_]+)", reason)
    if stop:
        return stop.group(1)
    return ""


def _residual_split_triggered(reason: str) -> bool:
    return "residual_split" in reason.lower()


def _format_centers(centers: list[tuple[float, float]]) -> str:
    if not centers:
        return ""
    return "; ".join(f"({row:.2f},{col:.2f})" for row, col in centers)


def load_object_summaries(folder: Path) -> dict[int, ObjectSummary]:
    measurements = pd.read_csv(find_csv(folder, "_measurements.csv"))
    diagnostics = pd.read_csv(find_csv(folder, "_object_diagnostics.csv"))

    accepted_rows = measurements[measurements["accepted"].map(_to_bool)].copy()
    accepted_rows = accepted_rows.dropna(subset=["final_row", "final_col"])
    centers_by_object: dict[int, list[tuple[float, float]]] = {}
    for object_id, group in accepted_rows.groupby("object_id"):
        centers_by_object[int(object_id)] = [
            (float(row), float(col))
            for row, col in zip(group["final_row"], group["final_col"], strict=True)
        ]

    summaries: dict[int, ObjectSummary] = {}
    for _, row in diagnostics.iterrows():
        object_id = int(row["object_id"])
        reason = "" if pd.isna(row.get("model_selection_reason")) else str(row["model_selection_reason"])
        centers = centers_by_object.get(object_id, [])
        best_r2 = row.get("best_gmm_r_squared")
        best_residual = row.get("best_gmm_residual_relative")
        summaries[object_id] = ObjectSummary(
            object_id=object_id,
            accepted_count=len(centers),
            centers=centers,
            best_gmm_r_squared=None if pd.isna(best_r2) else float(best_r2),
            best_gmm_residual=None if pd.isna(best_residual) else float(best_residual),
            selected_model_order=_parse_selected_model_order(reason),
            model_selection_reason=reason,
            residual_split_triggered=_residual_split_triggered(reason),
            residual_split_transition=_parse_residual_split_transition(reason),
            ambiguous=_parse_ambiguous(reason),
            stop_reason=_parse_stop_reason(reason),
            split_attempts=_parse_split_attempts(reason),
        )
    return summaries


def match_center_displacements(
    b_centers: list[tuple[float, float]],
    c_centers: list[tuple[float, float]],
) -> tuple[float, float]:
    if not b_centers and not c_centers:
        return 0.0, 0.0
    if not b_centers or not c_centers:
        return float("inf"), float("inf")

    b = np.asarray(b_centers, dtype=float)
    c = np.asarray(c_centers, dtype=float)
    cost = np.linalg.norm(b[:, None, :] - c[None, :, :], axis=2)
    row_idx, col_idx = linear_sum_assignment(cost)
    matched = [float(cost[r, c]) for r, c in zip(row_idx, col_idx, strict=True)]
    unmatched = abs(len(b_centers) - len(c_centers))
    if unmatched:
        matched.extend([float("inf")] * unmatched)
    finite = [value for value in matched if np.isfinite(value)]
    if not finite:
        return float("inf"), float("inf")
    return max(finite), float(np.mean(finite))


def build_comparison(
    b_summaries: dict[int, ObjectSummary],
    c_summaries: dict[int, ObjectSummary],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for object_id in sorted(set(b_summaries) | set(c_summaries)):
        b = b_summaries.get(object_id)
        c = c_summaries.get(object_id)
        b_count = b.accepted_count if b else 0
        c_count = c.accepted_count if c else 0
        max_disp, mean_disp = match_center_displacements(
            b.centers if b else [],
            c.centers if c else [],
        )
        count_changed = b_count != c_count
        continued_beyond_b = c_count > b_count
        grew_3_to_4 = b_count == 3 and c_count == 4

        b_r2 = b.best_gmm_r_squared if b else None
        c_r2 = c.best_gmm_r_squared if c else None
        b_res = b.best_gmm_residual if b else None
        c_res = c.best_gmm_residual if c else None
        r2_delta = (c_r2 - b_r2) if b_r2 is not None and c_r2 is not None else None
        res_delta = (b_res - c_res) if b_res is not None and c_res is not None else None

        interest = 0
        if count_changed:
            interest += 1000
        if continued_beyond_b:
            interest += 800
        if grew_3_to_4:
            interest += 600
        if np.isfinite(max_disp) and max_disp > 1.0:
            interest += 400
        elif np.isfinite(max_disp) and max_disp > 0.5:
            interest += 200
        if c and c.residual_split_triggered and (not b or not b.residual_split_triggered):
            interest += 150
        if c and c.ambiguous:
            interest += 100
        if r2_delta is not None and r2_delta > 0.05:
            interest += 50
        if res_delta is not None and res_delta > 0.05:
            interest += 50

        rows.append(
            {
                "object_id": object_id,
                "phase_b_accepted_count": b_count,
                "phase_c_accepted_count": c_count,
                "accepted_count_changed": count_changed,
                "phase_c_continued_beyond_phase_b": continued_beyond_b,
                "phase_c_grew_3_to_4": grew_3_to_4,
                "phase_b_final_accepted_centers": _format_centers(b.centers if b else []),
                "phase_c_final_accepted_centers": _format_centers(c.centers if c else []),
                "nearest_center_displacement_px": max_disp,
                "mean_matched_center_displacement_px": mean_disp,
                "phase_b_best_gmm_r_squared": b_r2,
                "phase_c_best_gmm_r_squared": c_r2,
                "delta_best_gmm_r_squared": r2_delta,
                "phase_b_best_gmm_residual": b_res,
                "phase_c_best_gmm_residual": c_res,
                "delta_best_gmm_residual": res_delta,
                "phase_b_selected_model_order": b.selected_model_order if b else None,
                "phase_c_selected_model_order": c.selected_model_order if c else None,
                "phase_b_residual_split_triggered": b.residual_split_triggered if b else False,
                "phase_c_residual_split_triggered": c.residual_split_triggered if c else False,
                "phase_b_residual_split_transition": b.residual_split_transition if b else "",
                "phase_c_residual_split_transition": c.residual_split_transition if c else "",
                "phase_b_split_attempts": b.split_attempts if b else None,
                "phase_c_split_attempts": c.split_attempts if c else None,
                "phase_c_ambiguous": c.ambiguous if c else False,
                "phase_c_stop_reason": c.stop_reason if c else "",
                "phase_b_model_selection_reason": b.model_selection_reason if b else "",
                "phase_c_model_selection_reason": c.model_selection_reason if c else "",
                "interest_score": interest,
            }
        )

    frame = pd.DataFrame(rows).sort_values(
        ["interest_score", "object_id"], ascending=[False, True], kind="stable"
    )
    return frame.reset_index(drop=True)


def print_report(frame: pd.DataFrame, b_summary: dict[str, object], c_summary: dict[str, object]) -> None:
    b_runtime = b_summary.get("total_runtime_seconds")
    c_runtime = c_summary.get("total_runtime_seconds")
    print("\n=== Phase B vs Phase C Summary (example2) ===")
    print(f"total objects compared: {len(frame)}")
    print(f"suspicious objects (Phase B / C): {b_summary.get('suspicious_objects')} / {c_summary.get('suspicious_objects')}")
    print(f"GMM tried (Phase B / C): {b_summary.get('gmm_triggered_objects')} / {c_summary.get('gmm_triggered_objects')}")
    print(f"runtime Phase B: {b_runtime:.1f}s" if b_runtime is not None else "runtime Phase B: n/a")
    print(f"runtime Phase C: {c_runtime:.1f}s" if c_runtime is not None else "runtime Phase C: n/a")
    if b_runtime is not None and c_runtime is not None:
        print(f"runtime delta (C - B): {float(c_runtime) - float(b_runtime):+.1f}s")

    print(f"residual splits triggered (Phase B): {int(frame['phase_b_residual_split_triggered'].sum())}")
    print(f"residual splits triggered (Phase C): {int(frame['phase_c_residual_split_triggered'].sum())}")
    print(f"accepted component count differs: {int(frame['accepted_count_changed'].sum())}")
    print(f"Phase C continued beyond Phase B: {int(frame['phase_c_continued_beyond_phase_b'].sum())}")
    print(f"Phase C grew 3 -> 4: {int(frame['phase_c_grew_3_to_4'].sum())}")
    print(f"centers moved > 0.5 px: {int((frame['nearest_center_displacement_px'] > 0.5).sum())}")
    print(f"centers moved > 1.0 px: {int((frame['nearest_center_displacement_px'] > 1.0).sum())}")
    print(f"Phase C ambiguous stops: {int(frame['phase_c_ambiguous'].sum())}")

    ambiguous = frame[frame["phase_c_ambiguous"]]
    if not ambiguous.empty:
        print("\nAmbiguous / stop reasons (Phase C):")
        for reason, count in ambiguous["phase_c_stop_reason"].value_counts().items():
            print(f"  {reason or '(empty)'}: {count}")

    interesting = frame[frame["interest_score"] > 0].head(20)
    print("\n=== Top objects to inspect visually ===")
    if interesting.empty:
        print("(none)")
        return
    for _, row in interesting.iterrows():
        print(
            f"object {int(row['object_id']):4d} | score={int(row['interest_score']):4d} | "
            f"count {int(row['phase_b_accepted_count'])}->{int(row['phase_c_accepted_count'])} | "
            f"disp={row['nearest_center_displacement_px']:.3f}px | "
            f"B_split={bool(row['phase_b_residual_split_triggered'])} | "
            f"C_split={bool(row['phase_c_residual_split_triggered'])} | "
            f"C_transition={row['phase_c_residual_split_transition'] or '-'} | "
            f"ambiguous={bool(row['phase_c_ambiguous'])}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Phase B vs Phase C declump outputs.")
    parser.add_argument("--phase-b-dir", type=Path, default=DEFAULT_B_DIR)
    parser.add_argument("--phase-c-dir", type=Path, default=DEFAULT_C_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    b_summaries = load_object_summaries(args.phase_b_dir)
    c_summaries = load_object_summaries(args.phase_c_dir)
    comparison = build_comparison(b_summaries, c_summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False)

    b_summary = load_summary(args.phase_b_dir)
    c_summary = load_summary(args.phase_c_dir)
    print(f"Wrote comparison CSV: {args.output.resolve()}")
    print_report(comparison, b_summary, c_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
