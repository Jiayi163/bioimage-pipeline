#!/usr/bin/env python3
"""Evaluate synthetic puncta pipeline runs against known ground truth.

Standalone validation utility — reads ground-truth JSON and pipeline CSV outputs.
Does not modify production puncta logic.

Usage (from project root):
    python scripts/evaluate_synthetic_puncta.py --evaluate-basic
    python scripts/evaluate_synthetic_puncta.py --run-name case1_isolated
    python scripts/evaluate_synthetic_puncta.py --separation-benchmark
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

DEFAULT_TOLERANCE_PX = 2.0
EVALUATOR_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Run specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    """Maps a pipeline results folder to a ground-truth case folder."""

    run_name: str
    ground_truth_case: str
    measurements_stem: str
    notes: str = ""


BASIC_RUNS: tuple[RunSpec, ...] = (
    RunSpec("case1_isolated", "case1_isolated", "case1_isolated"),
    RunSpec("case2_separated", "case2_separated", "case2_separated"),
    RunSpec(
        "case3_overlapping_normal",
        "case3_overlapping",
        "case3_overlapping_normal",
        notes="Normal selective routing",
    ),
    RunSpec(
        "case3_overlapping_forced_gmm",
        "case3_overlapping",
        "case3_overlapping_forced_gmm",
        notes="Forced full fitting (--no-selective-routing)",
    ),
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class TrueSpot:
    spot_id: int
    x: float
    y: float
    amplitude: float
    sigma_x: float
    sigma_y: float


@dataclass
class GroundTruth:
    case_name: str
    true_spot_count: int
    spots: list[TrueSpot]
    raw: dict[str, Any]


@dataclass
class PredictedSpot:
    x: float
    y: float
    path: str | None
    fit_status: str | None
    tried_gmm: bool
    n_components_in_model: int | None
    n_filtered_local_maxima: int | None
    under_split_suspect: bool
    rejection_reason: str | None
    object_id: int | None
    candidate_id: int | None


def load_ground_truth(path: Path) -> GroundTruth:
    payload = json.loads(path.read_text(encoding="utf-8"))
    spots = [
        TrueSpot(
            spot_id=int(entry["id"]),
            x=float(entry["x"]),
            y=float(entry["y"]),
            amplitude=float(entry["amplitude"]),
            sigma_x=float(entry["sigma_x"]),
            sigma_y=float(entry.get("sigma_y", entry["sigma_x"])),
        )
        for entry in payload["spots"]
    ]
    return GroundTruth(
        case_name=str(payload.get("case_name", path.parent.name)),
        true_spot_count=int(payload.get("true_spot_count", len(spots))),
        spots=spots,
        raw=payload,
    )


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "t"}


def _parse_optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_predicted_spots(measurements_csv: Path) -> list[PredictedSpot]:
    """Load final accepted puncta from pipeline measurements CSV."""
    if not measurements_csv.is_file():
        return []

    df = pd.read_csv(measurements_csv)
    if df.empty:
        return []

    if "accepted" not in df.columns:
        raise ValueError(f"Missing 'accepted' column in {measurements_csv}")

    accepted = df[df["accepted"].map(_parse_bool)]
    if accepted.empty:
        return []

    if "final_col" not in accepted.columns or "final_row" not in accepted.columns:
        raise ValueError(
            f"Missing final_col/final_row columns in {measurements_csv}"
        )

    spots: list[PredictedSpot] = []
    for _, row in accepted.iterrows():
        if pd.isna(row["final_col"]) or pd.isna(row["final_row"]):
            continue
        spots.append(
            PredictedSpot(
                x=float(row["final_col"]),
                y=float(row["final_row"]),
                path=str(row["path"]) if "path" in row and pd.notna(row["path"]) else None,
                fit_status=str(row["fit_status"])
                if "fit_status" in row and pd.notna(row["fit_status"])
                else None,
                tried_gmm=_parse_bool(row.get("tried_gmm")),
                n_components_in_model=_parse_optional_int(row.get("n_components_in_model")),
                n_filtered_local_maxima=_parse_optional_int(
                    row.get("n_filtered_local_maxima")
                ),
                under_split_suspect=_parse_bool(row.get("under_split_suspect")),
                rejection_reason=str(row["rejection_reason"])
                if "rejection_reason" in row and pd.notna(row["rejection_reason"])
                else None,
                object_id=_parse_optional_int(row.get("object_id")),
                candidate_id=_parse_optional_int(row.get("candidate_id")),
            )
        )
    return spots


def load_summary_json(summary_path: Path) -> dict[str, Any]:
    if not summary_path.is_file():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Matching and metrics
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    true_index: int
    pred_index: int
    distance_px: float
    within_tolerance: bool


@dataclass
class EvaluationMetrics:
    run_name: str
    ground_truth_case: str
    measurements_stem: str
    tolerance_px: float
    true_spot_count: int
    predicted_accepted_count: int
    exact_count_correct: bool
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    mean_localization_error_px: float | None
    median_localization_error_px: float | None
    max_localization_error_px: float | None
    under_split: bool
    over_split: bool
    pass_criterion: bool
    matches: list[MatchResult] = field(default_factory=list)
    localization_errors_px: list[float] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_summary_row(self) -> dict[str, Any]:
        row = {
            "run_name": self.run_name,
            "ground_truth_case": self.ground_truth_case,
            "measurements_stem": self.measurements_stem,
            "tolerance_px": self.tolerance_px,
            "true_spot_count": self.true_spot_count,
            "predicted_accepted_count": self.predicted_accepted_count,
            "exact_count_correct": self.exact_count_correct,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "mean_localization_error_px": self.mean_localization_error_px,
            "median_localization_error_px": self.median_localization_error_px,
            "max_localization_error_px": self.max_localization_error_px,
            "under_split": self.under_split,
            "over_split": self.over_split,
            "pass_criterion": self.pass_criterion,
            "notes": self.notes,
        }
        row.update(self.diagnostics)
        return row


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(math.hypot(x1 - x2, y1 - y2))


def match_centers(
    true_spots: list[TrueSpot],
    predicted: list[PredictedSpot],
    *,
    tolerance_px: float,
) -> tuple[list[MatchResult], list[int], list[int]]:
    """One-to-one Hungarian matching; TP requires distance <= tolerance."""
    n_true = len(true_spots)
    n_pred = len(predicted)
    if n_true == 0 and n_pred == 0:
        return [], [], []
    if n_true == 0:
        return [], [], list(range(n_pred))
    if n_pred == 0:
        return [], list(range(n_true)), []

    true_xy = np.array([[spot.x, spot.y] for spot in true_spots], dtype=float)
    pred_xy = np.array([[spot.x, spot.y] for spot in predicted], dtype=float)

    diff = true_xy[:, None, :] - pred_xy[None, :, :]
    distance_matrix = np.linalg.norm(diff, axis=2)

    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    matches: list[MatchResult] = []
    matched_true: set[int] = set()
    matched_pred: set[int] = set()

    for row, col in zip(row_ind, col_ind):
        dist = float(distance_matrix[row, col])
        within = dist <= tolerance_px
        matches.append(
            MatchResult(
                true_index=int(row),
                pred_index=int(col),
                distance_px=dist,
                within_tolerance=within,
            )
        )
        if within:
            matched_true.add(int(row))
            matched_pred.add(int(col))

    assigned_true = set(int(row) for row in row_ind)
    assigned_pred = set(int(col) for col in col_ind)
    unmatched_true = [index for index in range(n_true) if index not in assigned_true]
    unmatched_pred = [index for index in range(n_pred) if index not in assigned_pred]
    return matches, unmatched_true, unmatched_pred


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalar types for CSV/JSON export."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    return value


def compute_diagnostics(
    predicted: list[PredictedSpot],
    summary: dict[str, Any],
    all_rows_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Aggregate diagnostic fields from accepted predictions and summary JSON."""
    diag: dict[str, Any] = {}

    if predicted:
        paths = [p.path for p in predicted if p.path]
        diag["accepted_path_counts"] = dict(pd.Series(paths).value_counts()) if paths else {}
        diag["accepted_tried_gmm_count"] = sum(1 for p in predicted if p.tried_gmm)
        diag["accepted_under_split_suspect_count"] = sum(
            1 for p in predicted if p.under_split_suspect
        )
        diag["accepted_fit_status_counts"] = dict(
            pd.Series([p.fit_status for p in predicted if p.fit_status]).value_counts()
        )
        component_counts = [
            p.n_components_in_model
            for p in predicted
            if p.n_components_in_model is not None
        ]
        diag["accepted_n_components_in_model_max"] = (
            max(component_counts) if component_counts else None
        )
        filtered_maxima = [
            p.n_filtered_local_maxima
            for p in predicted
            if p.n_filtered_local_maxima is not None
        ]
        diag["accepted_n_filtered_local_maxima_max"] = (
            max(filtered_maxima) if filtered_maxima else None
        )

    if summary:
        run_summary = summary.get("summary", {})
        diag["pipeline_total_accepted"] = run_summary.get("total_accepted")
        diag["pipeline_gmm_triggered_objects"] = run_summary.get("gmm_triggered_objects")
        diag["pipeline_gmm_accepted_objects"] = run_summary.get("gmm_accepted_objects")
        diag["pipeline_fast_path_objects"] = run_summary.get("fast_path_objects")
        diag["pipeline_suspicious_objects"] = run_summary.get("suspicious_objects")
        diag["pipeline_fit_failed_fallback_count"] = run_summary.get(
            "fit_failed_fallback_count"
        )
        diag["pipeline_total_rejected"] = run_summary.get("total_rejected")
        diag["pipeline_under_split_suspect_objects"] = run_summary.get(
            "under_split_suspect_objects"
        )

    if all_rows_df is not None and not all_rows_df.empty:
        if "fit_status" in all_rows_df.columns:
            diag["all_fit_status_counts"] = dict(
                all_rows_df["fit_status"].value_counts(dropna=False)
            )
        if "rejection_reason" in all_rows_df.columns:
            rejected = all_rows_df[~all_rows_df["accepted"].map(_parse_bool)]
            reasons = rejected["rejection_reason"].dropna()
            if not reasons.empty:
                diag["rejection_reason_counts"] = dict(reasons.value_counts())

    return _json_safe(diag)


def evaluate_run(
    *,
    run_name: str,
    ground_truth_case: str,
    measurements_stem: str,
    data_root: Path,
    tolerance_px: float = DEFAULT_TOLERANCE_PX,
    notes: str = "",
) -> EvaluationMetrics:
    gt_path = data_root / "ground_truth" / ground_truth_case / "synthetic_ground_truth.json"
    results_dir = data_root / "results" / run_name
    measurements_csv = results_dir / f"{measurements_stem}_measurements.csv"
    summary_json = results_dir / f"{measurements_stem}_summary.json"

    ground_truth = load_ground_truth(gt_path)
    predicted = load_predicted_spots(measurements_csv)
    summary = load_summary_json(summary_json)

    all_rows_df = pd.read_csv(measurements_csv) if measurements_csv.is_file() else None

    matches, unmatched_true, unmatched_pred = match_centers(
        ground_truth.spots,
        predicted,
        tolerance_px=tolerance_px,
    )

    tp_matches = [match for match in matches if match.within_tolerance]
    localization_errors = [match.distance_px for match in tp_matches]

    tp = len(tp_matches)
    fp = len(unmatched_pred) + sum(1 for match in matches if not match.within_tolerance)
    fn = len(unmatched_true) + sum(1 for match in matches if not match.within_tolerance)

    predicted_count = len(predicted)
    true_count = ground_truth.true_spot_count
    exact_count = predicted_count == true_count
    under_split = predicted_count < true_count
    over_split = predicted_count > true_count

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tp == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    pass_criterion = (
        exact_count
        and fp == 0
        and fn == 0
        and (not localization_errors or max(localization_errors) <= tolerance_px)
    )

    diagnostics = compute_diagnostics(predicted, summary, all_rows_df)
    diagnostics["ground_truth_path"] = str(gt_path)
    diagnostics["measurements_csv"] = str(measurements_csv)
    diagnostics["summary_json"] = str(summary_json)

    return EvaluationMetrics(
        run_name=run_name,
        ground_truth_case=ground_truth_case,
        measurements_stem=measurements_stem,
        tolerance_px=tolerance_px,
        true_spot_count=true_count,
        predicted_accepted_count=predicted_count,
        exact_count_correct=exact_count,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        mean_localization_error_px=float(np.mean(localization_errors))
        if localization_errors
        else None,
        median_localization_error_px=float(np.median(localization_errors))
        if localization_errors
        else None,
        max_localization_error_px=max(localization_errors) if localization_errors else None,
        under_split=under_split,
        over_split=over_split,
        pass_criterion=pass_criterion,
        matches=matches,
        localization_errors_px=localization_errors,
        diagnostics=diagnostics,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_console_report(metrics: EvaluationMetrics) -> str:
    lines = [
        metrics.run_name,
        f"True count: {metrics.true_spot_count}",
        f"Predicted count: {metrics.predicted_accepted_count}",
        f"Count correct: {'Yes' if metrics.exact_count_correct else 'No'}",
        (
            f"TP: {metrics.true_positives}, FP: {metrics.false_positives}, "
            f"FN: {metrics.false_negatives}"
        ),
    ]
    if metrics.median_localization_error_px is not None:
        lines.append(
            f"Median localization error: {metrics.median_localization_error_px:.2f} px"
        )
    else:
        lines.append("Median localization error: n/a")
    lines.append(f"Result: {'PASS' if metrics.pass_criterion else 'FAIL'}")
    if metrics.notes:
        lines.append(f"Notes: {metrics.notes}")
    return "\n".join(lines)


def write_summary_outputs(
    metrics_list: list[EvaluationMetrics],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [metrics.to_summary_row() for metrics in metrics_list]
    csv_path = output_dir / "synthetic_validation_summary.csv"
    json_path = output_dir / "synthetic_validation_summary.json"

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    payload = {
        "evaluator_version": EVALUATOR_VERSION,
        "tolerance_px": metrics_list[0].tolerance_px if metrics_list else DEFAULT_TOLERANCE_PX,
        "runs": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return csv_path, json_path


def summarize_separation_benchmark(
    metrics_list: list[EvaluationMetrics],
) -> pd.DataFrame:
    """Aggregate separation benchmark pass rates by center separation."""
    import re

    pattern = re.compile(r"^sep_benchmark_sep(\d+)_seed\d+$")
    rows: list[dict[str, Any]] = []
    for metrics in metrics_list:
        match = pattern.match(metrics.run_name)
        if not match:
            continue
        separation = int(match.group(1))
        rows.append(
            {
                "separation_px": separation,
                "run_name": metrics.run_name,
                "predicted_count": metrics.predicted_accepted_count,
                "exact_count_correct": metrics.exact_count_correct,
                "pass_criterion": metrics.pass_criterion,
            }
        )
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("separation_px")
        .agg(
            runs=("run_name", "count"),
            fraction_exact_count=("exact_count_correct", "mean"),
            fraction_pass=("pass_criterion", "mean"),
            mean_predicted_count=("predicted_count", "mean"),
        )
        .reset_index()
        .sort_values("separation_px")
    )
    return summary


# ---------------------------------------------------------------------------
# Pipeline helpers (optional)
# ---------------------------------------------------------------------------


def run_pipeline_for_case(
    *,
    image_case: str,
    run_name: str,
    stem: str,
    data_root: Path,
    project_root: Path,
    force_gmm: bool = False,
) -> None:
    image = data_root / "images" / image_case / "synthetic_noisy.tif"
    mask = data_root / "masks" / image_case / "synthetic_mask.tif"
    output_dir = data_root / "results" / run_name

    cmd = [
        sys.executable,
        str(project_root / "examples" / "run_puncta_declump.py"),
        "--input",
        str(image),
        "--mask",
        str(mask),
        "--output-dir",
        str(output_dir),
        "--stem",
        stem,
        "--candidate-detector",
        "python_log",
        "--diagnostic-mode",
        "summary",
        "--no-fiji-tiffs",
    ]
    if force_gmm:
        cmd.append("--no-selective-routing")

    subprocess.run(cmd, check=True, cwd=project_root)


def discover_separation_benchmark_cases(data_root: Path) -> list[str]:
    """Find separation benchmark case names from generated image folders."""
    images_root = data_root / "images"
    if not images_root.is_dir():
        return []
    return sorted(
        path.name
        for path in images_root.iterdir()
        if path.is_dir() and path.name.startswith("sep_benchmark_")
    )


def discover_separation_benchmark_runs(data_root: Path) -> list[RunSpec]:
    results_root = data_root / "results"
    specs: list[RunSpec] = []
    case_names = discover_separation_benchmark_cases(data_root)
    for name in case_names:
        if (results_root / name).is_dir():
            specs.append(
                RunSpec(
                    run_name=name,
                    ground_truth_case=name,
                    measurements_stem=name,
                    notes="Separation benchmark (forced GMM)",
                )
            )
    return specs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic puncta pipeline runs against ground truth.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("synthetic_test_data"),
        help="Synthetic data root (default: synthetic_test_data).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_PX,
        help="Maximum matching distance in pixels (default: 2.0).",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        dest="run_names",
        help="Evaluate one results folder by name.",
    )
    parser.add_argument(
        "--ground-truth-case",
        action="append",
        dest="ground_truth_cases",
        help="Ground-truth case folder (paired with --run-name).",
    )
    parser.add_argument(
        "--measurements-stem",
        action="append",
        dest="measurement_stems",
        help="Measurements CSV stem (paired with --run-name).",
    )
    parser.add_argument(
        "--evaluate-basic",
        action="store_true",
        help="Evaluate the four basic validation runs.",
    )
    parser.add_argument(
        "--separation-benchmark",
        action="store_true",
        help="Evaluate separation benchmark runs under results/sep_benchmark_*.",
    )
    parser.add_argument(
        "--run-pipelines",
        action="store_true",
        help="Run the puncta pipeline for selected cases before evaluation.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root for subprocess pipeline calls.",
    )
    return parser.parse_args()


def build_run_specs(args: argparse.Namespace, data_root: Path) -> list[RunSpec]:
    specs: list[RunSpec] = []

    if args.evaluate_basic:
        specs.extend(BASIC_RUNS)

    if args.separation_benchmark:
        specs.extend(
            RunSpec(
                run_name=name,
                ground_truth_case=name,
                measurements_stem=name,
                notes="Separation benchmark (forced GMM)",
            )
            for name in discover_separation_benchmark_cases(data_root)
        )

    if args.run_names:
        gt_cases = args.ground_truth_cases or []
        stems = args.measurement_stems or []
        for index, run_name in enumerate(args.run_names):
            gt_case = gt_cases[index] if index < len(gt_cases) else run_name
            stem = stems[index] if index < len(stems) else run_name
            specs.append(RunSpec(run_name, gt_case, stem))

    if not specs:
        specs.extend(BASIC_RUNS)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[RunSpec] = []
    for spec in specs:
        if spec.run_name in seen:
            continue
        seen.add(spec.run_name)
        unique.append(spec)
    return unique


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    project_root = args.project_root.resolve()

    if args.run_pipelines:
        for spec in BASIC_RUNS:
            image_case = spec.ground_truth_case
            run_pipeline_for_case(
                image_case=image_case,
                run_name=spec.run_name,
                stem=spec.measurements_stem,
                data_root=data_root,
                project_root=project_root,
                force_gmm=spec.run_name == "case3_overlapping_forced_gmm",
            )
        for spec in discover_separation_benchmark_runs(data_root):
            run_pipeline_for_case(
                image_case=spec.ground_truth_case,
                run_name=spec.run_name,
                stem=spec.measurements_stem,
                data_root=data_root,
                project_root=project_root,
                force_gmm=True,
            )
        pending_sep = [
            name
            for name in discover_separation_benchmark_cases(data_root)
            if not (data_root / "results" / name).is_dir()
            or not (data_root / "results" / name / f"{name}_measurements.csv").is_file()
        ]
        for name in pending_sep:
            run_pipeline_for_case(
                image_case=name,
                run_name=name,
                stem=name,
                data_root=data_root,
                project_root=project_root,
                force_gmm=True,
            )

    run_specs = build_run_specs(args, data_root)
    metrics_list: list[EvaluationMetrics] = []

    for spec in run_specs:
        metrics = evaluate_run(
            run_name=spec.run_name,
            ground_truth_case=spec.ground_truth_case,
            measurements_stem=spec.measurements_stem,
            data_root=data_root,
            tolerance_px=args.tolerance,
            notes=spec.notes,
        )
        metrics_list.append(metrics)
        print(format_console_report(metrics))
        print()

    summary_dir = data_root / "results"
    csv_path, json_path = write_summary_outputs(metrics_list, summary_dir)
    print(f"Wrote summary CSV: {csv_path}")
    print(f"Wrote summary JSON: {json_path}")

    sep_summary = summarize_separation_benchmark(metrics_list)
    if not sep_summary.empty:
        sep_csv = summary_dir / "separation_benchmark_summary.csv"
        sep_summary.to_csv(sep_csv, index=False)
        print(f"Wrote separation benchmark summary: {sep_csv}")
        print(sep_summary.to_string(index=False))


if __name__ == "__main__":
    main()
