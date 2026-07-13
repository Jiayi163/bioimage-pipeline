#!/usr/bin/env python3
"""Compare synthetic validation results across GMM rollout stages."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.export import ResultExporter
from bioimage_pipeline.puncta.pipeline import run_puncta_declump
from scripts.evaluate_synthetic_puncta import (
    BASIC_RUNS,
    RunSpec,
    discover_separation_benchmark_cases,
    evaluate_run,
)

STAGE_CONFIGS: dict[str, dict[str, object]] = {
    "baseline": {
        "gmm_multi_start_enabled": False,
        "gmm_use_mixture_acceptance_separation": False,
    },
    "stage1": {
        "gmm_multi_start_enabled": True,
        "gmm_use_mixture_acceptance_separation": False,
    },
    "stage2": {
        "gmm_multi_start_enabled": True,
        "gmm_use_mixture_acceptance_separation": True,
        "gmm_acceptance_min_separation": 1.5,
    },
}


@dataclass
class StageRunMetrics:
    stage: str
    run_name: str
    true_count: int
    predicted_count: int
    accepted_count: int
    selected_component_count: int | None
    winning_init_strategy: str | None
    bic_delta: float | None
    aic_delta: float | None
    fitted_center_distance_px: float | None
    rejection_reason: str | None
    runtime_s: float | None
    pass_validation: bool


def _build_config(stage: str, *, no_selective_routing: bool) -> PunctaDeclumpConfig:
    params = dict(STAGE_CONFIGS[stage])
    params["enable_selective_routing"] = not no_selective_routing
    params["diagnostic_mode"] = "summary"
    params["export_fiji_tiffs"] = False
    params["candidate_detector"] = "python_log"
    return PunctaDeclumpConfig(**params)  # type: ignore[arg-type]


def _run_case(
    data_root: Path,
    spec: RunSpec,
    *,
    stage: str,
    no_selective_routing: bool,
) -> float:
    image = tifffile.imread(data_root / "images" / spec.ground_truth_case / "synthetic_noisy.tif")
    mask = tifffile.imread(data_root / "masks" / spec.ground_truth_case / "synthetic_mask.tif") > 0
    out_dir = data_root / "results" / f"{spec.run_name}_{stage}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _build_config(stage, no_selective_routing=no_selective_routing)
    start = time.perf_counter()
    result = run_puncta_declump(
        image,
        config,
        external_mask=mask,
        output_dir=out_dir,
        stem=spec.measurements_stem,
    )
    exporter = ResultExporter()
    exporter.export_csv(out_dir / f"{spec.measurements_stem}_measurements.csv", result)
    exporter.export_summary_json(out_dir / f"{spec.measurements_stem}_summary.json", result)
    init_diag = result.threshold_metadata.get("gmm_init_diagnostics")
    if init_diag:
        (out_dir / f"{spec.measurements_stem}_gmm_init_diagnostics.json").write_text(
            json.dumps(init_diag, indent=2),
            encoding="utf-8",
        )
    runtime = time.perf_counter() - start
    timing_path = out_dir / f"{spec.measurements_stem}_timing.json"
    timing_path.write_text(
        json.dumps(
            {
                "total_runtime_s": runtime,
                "gaussian_fit_time": result.timing.get("gaussian_fit_time"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return runtime


def _extract_gmm_metrics(csv_path: Path) -> dict[str, object]:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    gmm_rows = df[df["tried_gmm"] == True]  # noqa: E712
    if gmm_rows.empty:
        gmm_rows = df
    first = gmm_rows.iloc[0]
    accepted = df[df["accepted"] == True]  # noqa: E712
    selected_n = None
    if "best_gmm_n_components" in df.columns and pd.notna(first.get("best_gmm_n_components")):
        selected_n = int(first["best_gmm_n_components"])
    elif pd.notna(first.get("n_components_in_model")):
        selected_n = int(first["n_components_in_model"])
    bic_delta = None
    if "gmm_bic_delta_vs_single" in df.columns and pd.notna(first.get("gmm_bic_delta_vs_single")):
        bic_delta = float(first["gmm_bic_delta_vs_single"])
    aic_delta = None
    if "gmm_aic_delta_vs_single" in df.columns and pd.notna(first.get("gmm_aic_delta_vs_single")):
        aic_delta = float(first["gmm_aic_delta_vs_single"])
    rejection = None
    rejected = df[df["accepted"] == False]  # noqa: E712
    if not rejected.empty:
        rejection = str(
            rejected.iloc[0].get("rejection_reason")
            or rejected.iloc[0].get("rejected_component_reason")
            or ""
        ) or None
    distance = first.get("gmm_duplicate_distance_px") if "gmm_duplicate_distance_px" in df.columns else None
    if (distance is None or (isinstance(distance, float) and math.isnan(distance))) and len(accepted) >= 2:
        rows = accepted.sort_values("component_id")
        r0, c0 = float(rows.iloc[0]["final_row"]), float(rows.iloc[0]["final_col"])
        r1, c1 = float(rows.iloc[1]["final_row"]), float(rows.iloc[1]["final_col"])
        distance = math.hypot(c1 - c0, r1 - r0)
    return {
        "accepted_count": int(len(accepted)),
        "selected_component_count": selected_n,
        "winning_init_strategy": None
        if "gmm_winning_init_strategy" not in df.columns or pd.isna(first.get("gmm_winning_init_strategy"))
        else str(first.get("gmm_winning_init_strategy")),
        "bic_delta": bic_delta,
        "aic_delta": aic_delta,
        "fitted_center_distance_px": None
        if distance is None or (isinstance(distance, float) and math.isnan(distance))
        else float(distance),
        "rejection_reason": rejection,
    }


def _summarize_benchmark_metrics(metrics_list: list, stage: str) -> pd.DataFrame:
    pattern = re.compile(rf"^sep_benchmark_sep(\d+)_seed\d+_{re.escape(stage)}$")
    rows: list[dict[str, object]] = []
    for metrics in metrics_list:
        match = pattern.match(metrics.run_name)
        if not match:
            continue
        rows.append(
            {
                "separation_px": int(match.group(1)),
                "run_name": metrics.run_name,
                "predicted_count": metrics.predicted_accepted_count,
                "exact_count_correct": metrics.exact_count_correct,
                "pass_criterion": metrics.pass_criterion,
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return (
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


def _run_separation_benchmark(data_root: Path, stage: str) -> tuple[float, pd.DataFrame]:
    bench_runtime = 0.0
    bench_metrics = []
    for case_name in discover_separation_benchmark_cases(data_root):
        spec = RunSpec(
            run_name=case_name,
            ground_truth_case=case_name,
            measurements_stem=case_name,
            notes="Separation benchmark (forced GMM)",
        )
        bench_runtime += _run_case(data_root, spec, stage=stage, no_selective_routing=True)
        bench_metrics.append(
            evaluate_run(
                run_name=f"{spec.run_name}_{stage}",
                ground_truth_case=spec.ground_truth_case,
                measurements_stem=spec.measurements_stem,
                data_root=data_root,
            )
        )
    summary = _summarize_benchmark_metrics(bench_metrics, stage)
    return bench_runtime, summary


def collect_stage_metrics(data_root: Path, stage: str, *, include_benchmark: bool) -> list[StageRunMetrics]:
    rows: list[StageRunMetrics] = []
    for spec in BASIC_RUNS:
        no_forced = "forced_gmm" in spec.run_name
        runtime = _run_case(data_root, spec, stage=stage, no_selective_routing=no_forced)
        eval_name = f"{spec.run_name}_{stage}"
        metrics = evaluate_run(
            run_name=eval_name,
            ground_truth_case=spec.ground_truth_case,
            measurements_stem=spec.measurements_stem,
            data_root=data_root,
        )
        csv_path = data_root / "results" / eval_name / f"{spec.measurements_stem}_measurements.csv"
        extra = _extract_gmm_metrics(csv_path)
        rows.append(
            StageRunMetrics(
                stage=stage,
                run_name=spec.run_name,
                true_count=metrics.true_spot_count,
                predicted_count=metrics.predicted_accepted_count,
                accepted_count=int(extra.get("accepted_count", metrics.predicted_accepted_count)),
                selected_component_count=extra.get("selected_component_count"),  # type: ignore[arg-type]
                winning_init_strategy=extra.get("winning_init_strategy"),  # type: ignore[arg-type]
                bic_delta=extra.get("bic_delta"),  # type: ignore[arg-type]
                aic_delta=extra.get("aic_delta"),  # type: ignore[arg-type]
                fitted_center_distance_px=extra.get("fitted_center_distance_px"),  # type: ignore[arg-type]
                rejection_reason=extra.get("rejection_reason"),  # type: ignore[arg-type]
                runtime_s=runtime,
                pass_validation=metrics.pass_criterion,
            )
        )

    if include_benchmark:
        bench_runtime, summary = _run_separation_benchmark(data_root, stage)
        if not summary.empty:
            summary_path = data_root / "results" / f"separation_benchmark_{stage}.csv"
            summary.to_csv(summary_path, index=False)
            overall_pass = float(summary["fraction_pass"].mean())
            rows.append(
                StageRunMetrics(
                    stage=stage,
                    run_name="separation_benchmark_overall",
                    true_count=int(summary["runs"].sum()),
                    predicted_count=int(summary["mean_predicted_count"].sum()),
                    accepted_count=int(summary["mean_predicted_count"].sum()),
                    selected_component_count=None,
                    winning_init_strategy=None,
                    bic_delta=None,
                    aic_delta=None,
                    fitted_center_distance_px=None,
                    rejection_reason=f"pass_rate={overall_pass:.1%}",
                    runtime_s=bench_runtime,
                    pass_validation=overall_pass >= 0.5,
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GMM rollout stages on synthetic data.")
    parser.add_argument("--data-root", type=Path, default=Path("synthetic_test_data"))
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["baseline", "stage1", "stage2"],
        choices=list(STAGE_CONFIGS),
    )
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    all_rows: list[StageRunMetrics] = []
    for stage in args.stages:
        print(f"\n=== Running {stage} ===")
        all_rows.extend(
            collect_stage_metrics(data_root, stage, include_benchmark=not args.skip_benchmark)
        )

    out_json = data_root / "results" / "gmm_stage_comparison.json"
    out_csv = data_root / "results" / "gmm_stage_comparison.csv"
    payload = [asdict(row) for row in all_rows]
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(payload).to_csv(out_csv, index=False)
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
