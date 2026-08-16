#!/usr/bin/env python3
"""Run synthetic GMM benchmarks with detailed statistical reporting."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bioimage_pipeline.puncta.benchmark_stats import aggregate_benchmark_group, wilson_score_interval
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.export import ResultExporter
from bioimage_pipeline.puncta.pipeline import run_puncta_declump
from scripts.benchmark_seed_utils import generate_seed_list
from scripts.evaluate_synthetic_puncta import EvaluationMetrics, evaluate_run
from scripts.generate_synthetic_puncta import (
    SEPARATION_BENCHMARK_SIGMA,
    generate_brightness_ratio_benchmark,
    generate_false_split_benchmark,
    generate_separation_benchmark,
    generate_sigma_benchmark,
)


def discover_cases_by_prefix(data_root: Path, prefix: str) -> list[str]:
    images_root = data_root / "images"
    if not images_root.is_dir():
        return []
    return sorted(
        path.name
        for path in images_root.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    )


@dataclass
class BenchmarkRunRecord:
    benchmark: str
    case_name: str
    run_name: str
    metrics: EvaluationMetrics
    runtime_s: float
    search_mode: str
    winning_init_strategy: str | None
    selected_n_components: int | None
    spurious_split_rejected: bool
    fallback_rate: bool
    two_component_selected: bool
    profiling: dict[str, Any]


def _stage2_config(
    *,
    search_mode: str = "full",
    multi_start_enabled: bool = True,
) -> PunctaDeclumpConfig:
    return PunctaDeclumpConfig(
        gmm_multi_start_enabled=multi_start_enabled,
        gmm_use_mixture_acceptance_separation=True,
        gmm_acceptance_min_separation=1.5,
        gmm_multi_start_mode=search_mode,  # type: ignore[arg-type]
        enable_selective_routing=False,
        diagnostic_mode="summary",
        export_fiji_tiffs=False,
        candidate_detector="python_log",
    )


def _ensure_case_exists(
    data_root: Path,
    builder: Callable[..., Any],
    *,
    write: bool,
    **kwargs: Any,
) -> str:
    case = builder(**kwargs)
    noisy = data_root / "images" / case.name / "synthetic_noisy.tif"
    if write or not noisy.is_file():
        write_case_outputs(case, data_root)
    return case.name


def _ensure_result_out_dir(out_dir: Path) -> None:
    """Create result directory before benchmark exports (cache hits may skip pipeline mkdir)."""
    out_dir.mkdir(parents=True, exist_ok=True)


def _select_target_cases(cases: list[str], case: str | None) -> list[str]:
    """Return one case or the full list; exit with guidance when --case is unknown."""
    if case is None:
        return cases
    if case in cases:
        return [case]
    available = ", ".join(sorted(cases))
    raise SystemExit(f"Unknown --case {case!r}. Available cases: {available}")


def _run_pipeline_case(
    data_root: Path,
    case_name: str,
    *,
    run_suffix: str,
    config: PunctaDeclumpConfig,
) -> tuple[Path, float, dict[str, Any]]:
    image = tifffile.imread(data_root / "images" / case_name / "synthetic_noisy.tif")
    mask = tifffile.imread(data_root / "masks" / case_name / "synthetic_mask.tif") > 0
    run_name = f"{case_name}_{run_suffix}"
    out_dir = data_root / "results" / run_name
    _ensure_result_out_dir(out_dir)
    start = time.perf_counter()
    result = run_puncta_declump(
        image,
        config,
        external_mask=mask,
        output_dir=out_dir,
        stem=case_name,
    )
    runtime = time.perf_counter() - start
    _ensure_result_out_dir(out_dir)
    exporter = ResultExporter()
    exporter.export_csv(out_dir / f"{case_name}_measurements.csv", result)
    exporter.export_summary_json(out_dir / f"{case_name}_summary.json", result)
    init_diag = result.threshold_metadata.get("gmm_init_diagnostics")
    if init_diag:
        (out_dir / f"{case_name}_gmm_init_diagnostics.json").write_text(
            json.dumps(init_diag, indent=2, default=str),
            encoding="utf-8",
        )
    meta = {
        "gmm_config": result.threshold_metadata.get("gmm_config", {}),
        "gmm_init_diagnostics": init_diag or [],
        "timing": result.timing,
    }
    return out_dir, runtime, meta


def _extract_run_metadata(
    csv_path: Path,
    summary_meta: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "winning_init_strategy": None,
        "selected_n_components": None,
        "spurious_split_rejected": False,
        "fallback_rate": False,
        "two_component_selected": False,
        "profiling": {},
        "search_mode": summary_meta.get("gmm_config", {}).get("gmm_multi_start_mode", "full"),
    }
    if not csv_path.is_file():
        return out
    df = pd.read_csv(csv_path)
    if df.empty:
        return out
    row = df.iloc[0]
    out["winning_init_strategy"] = (
        None if pd.isna(row.get("gmm_winning_init_strategy")) else str(row.get("gmm_winning_init_strategy"))
    )
    if pd.notna(row.get("best_gmm_n_components")):
        out["selected_n_components"] = int(row["best_gmm_n_components"])
    reason = str(row.get("model_selection_reason") or "")
    out["spurious_split_rejected"] = "spurious_tight_split" in reason
    out["two_component_selected"] = bool(row.get("tried_gmm")) and out["selected_n_components"] == 2
    out["fallback_rate"] = str(row.get("path")) == "fallback"
    init_diag = summary_meta.get("gmm_init_diagnostics") or []
    if init_diag:
        attempts = init_diag[0].get("attempts") or []
        out["profiling"] = {
            "multi_start_attempts": init_diag[0].get("multi_start_attempts"),
            "multi_start_converged": init_diag[0].get("multi_start_converged"),
            "attempts": attempts,
        }
    return out


def run_benchmark_cases(
    *,
    benchmark: str,
    case_names: list[str],
    data_root: Path,
    run_suffix: str,
    config: PunctaDeclumpConfig,
    skip_existing: bool,
) -> list[BenchmarkRunRecord]:
    records: list[BenchmarkRunRecord] = []
    for case_name in case_names:
        run_name = f"{case_name}_{run_suffix}"
        csv_path = data_root / "results" / run_name / f"{case_name}_measurements.csv"
        if skip_existing and csv_path.is_file():
            runtime = 0.0
            meta = {"gmm_init_diagnostics": [], "gmm_config": {}}
            if (data_root / "results" / run_name / f"{case_name}_summary.json").is_file():
                summary = json.loads(
                    (data_root / "results" / run_name / f"{case_name}_summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                meta["gmm_config"] = summary.get("threshold_metadata", {}).get("gmm_config", {})
                meta["gmm_init_diagnostics"] = summary.get("threshold_metadata", {}).get(
                    "gmm_init_diagnostics", []
                )
        else:
            _, runtime, meta = _run_pipeline_case(
                data_root,
                case_name,
                run_suffix=run_suffix,
                config=config,
            )
        extra = _extract_run_metadata(csv_path, meta)
        metrics = evaluate_run(
            run_name=run_name,
            ground_truth_case=case_name,
            measurements_stem=case_name,
            data_root=data_root,
        )
        records.append(
            BenchmarkRunRecord(
                benchmark=benchmark,
                case_name=case_name,
                run_name=run_name,
                metrics=metrics,
                runtime_s=runtime,
                search_mode=str(extra["search_mode"]),
                winning_init_strategy=extra["winning_init_strategy"],
                selected_n_components=extra["selected_n_components"],
                spurious_split_rejected=bool(extra["spurious_split_rejected"]),
                fallback_rate=bool(extra["fallback_rate"]),
                two_component_selected=bool(extra["two_component_selected"]),
                profiling=extra["profiling"],
            )
        )
    return records


def records_to_frame(records: list[BenchmarkRunRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "benchmark": record.benchmark,
                "case_name": record.case_name,
                "run_name": record.run_name,
                "true_spot_count": record.metrics.true_spot_count,
                "predicted_accepted_count": record.metrics.predicted_accepted_count,
                "exact_count_correct": record.metrics.exact_count_correct,
                "pass_criterion": record.metrics.pass_criterion,
                "under_split": record.metrics.under_split,
                "over_split": record.metrics.over_split,
                "mean_localization_error_px": record.metrics.mean_localization_error_px,
                "median_localization_error_px": record.metrics.median_localization_error_px,
                "runtime_s": record.runtime_s,
                "search_mode": record.search_mode,
                "winning_init_strategy": record.winning_init_strategy,
                "selected_n_components": record.selected_n_components,
                "spurious_split_rejected": record.spurious_split_rejected,
                "fallback_rate": record.fallback_rate,
                "two_component_selected": record.two_component_selected,
            }
        )
    return pd.DataFrame(rows)


def summarize_separation_benchmark(frame: pd.DataFrame, *, sigma: float) -> pd.DataFrame:
    pattern = re.compile(r"^sep_benchmark_sep(\d+)_seed(\d+)$")
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        match = pattern.match(str(row["case_name"]))
        if not match:
            continue
        separation = int(match.group(1))
        rows.append({**row.to_dict(), "separation_px": separation, "sigma": sigma})
    if not rows:
        return pd.DataFrame()
    detailed = pd.DataFrame(rows)
    detailed["separation_over_sigma"] = detailed["separation_px"] / sigma
    summary = aggregate_benchmark_group(
        detailed,
        ["separation_px", "sigma"],
    )
    summary["separation_over_sigma"] = summary["separation_px"] / summary["sigma"]
    return summary


def summarize_false_split_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(
        r"^false_split_sig(?P<sig>[\dp]+)_noise(?P<noise>\w+)_amp(?P<amp>\w+)_"
        r"grad(?P<grad>\w+)_ellip(?P<ellip>\w+)_seed(?P<seed>\d+)$"
    )
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        match = pattern.match(str(row["case_name"]))
        if not match:
            continue
        sigma = float(match.group("sig").replace("p", "."))
        enriched = {
            **row.to_dict(),
            "sigma": sigma,
            "noise_level": match.group("noise"),
            "amplitude_level": match.group("amp"),
            "gradient_on": match.group("grad") == "on",
            "ellipticity": match.group("ellip"),
        }
        rows.append(enriched)
    if not rows:
        return pd.DataFrame()
    detailed = pd.DataFrame(rows)
    summary = aggregate_benchmark_group(
        detailed,
        ["sigma", "noise_level", "amplitude_level", "gradient_on", "ellipticity"],
    )
    group_cols = ["sigma", "noise_level", "amplitude_level", "gradient_on", "ellipticity"]
    extra_rates = detailed.groupby(group_cols, dropna=False).agg(
        fallback_rate=("fallback_rate", "mean"),
        two_component_selected_rate=("two_component_selected", "mean"),
        spurious_guard_rate=("spurious_split_rejected", "mean"),
    ).reset_index()
    summary = summary.merge(extra_rates, on=group_cols, how="left")
    summary["exactly_one_rate"] = 1.0 - summary["over_split_rate"]
    summary["false_split_rate"] = summary["over_split_rate"]
    return summary


def summarize_ratio_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(r"^ratio_benchmark_br([\d-]+)_sep(\d+)_seed\d+$")
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        match = pattern.match(str(row["case_name"]))
        if not match:
            continue
        rows.append(
            {
                **row.to_dict(),
                "brightness_ratio": match.group(1).replace("-", ":"),
                "separation_px": int(match.group(2)),
            }
        )
    if not rows:
        return pd.DataFrame()
    detailed = pd.DataFrame(rows)
    return aggregate_benchmark_group(detailed, ["brightness_ratio", "separation_px"])


def summarize_sigma_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(r"^sigma_benchmark_sig([\dp]+)_sep(\d+)_seed\d+$")
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        match = pattern.match(str(row["case_name"]))
        if not match:
            continue
        sigma = float(match.group(1).replace("p", "."))
        separation = int(match.group(2))
        rows.append(
            {
                **row.to_dict(),
                "sigma": sigma,
                "separation_px": separation,
                "separation_over_sigma": separation / sigma,
            }
        )
    if not rows:
        return pd.DataFrame()
    detailed = pd.DataFrame(rows)
    return aggregate_benchmark_group(detailed, ["sigma", "separation_px"])


def summarize_strategy_profiling(records: list[BenchmarkRunRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        attempts = record.profiling.get("attempts") or []
        for attempt in attempts:
            rows.append(
                {
                    "case_name": record.case_name,
                    "strategy": attempt.get("strategy"),
                    "converged": attempt.get("converged"),
                    "merge_collapsed": attempt.get("merge_collapsed"),
                    "selected": attempt.get("selected"),
                    "bic": attempt.get("bic"),
                    "rss": attempt.get("rss"),
                    "optimizer_runtime_s": attempt.get("optimizer_runtime_s"),
                    "n_optimizer_evaluations": attempt.get("n_optimizer_evaluations"),
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["bic_rank"] = frame.groupby("case_name")["bic"].rank(method="min")
    summary = (
        frame.groupby("strategy")
        .agg(
            attempts=("strategy", "count"),
            win_rate=("selected", "mean"),
            convergence_rate=("converged", "mean"),
            collapse_rate=("merge_collapsed", "mean"),
            mean_optimizer_runtime_s=("optimizer_runtime_s", "mean"),
            mean_nfev=("n_optimizer_evaluations", "mean"),
            mean_bic=("bic", "mean"),
            mean_bic_rank=("bic_rank", "mean"),
        )
        .reset_index()
        .sort_values("mean_bic_rank")
    )
    return summary


def verify_seed_completeness(case_names: list[str], expected_seeds: list[int], prefix: str) -> pd.DataFrame:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)_seed(\d+)$")
    rows: list[dict[str, Any]] = []
    grouped: dict[int, set[int]] = {}
    for name in case_names:
        match = pattern.match(name)
        if not match:
            continue
        sep = int(match.group(1))
        seed = int(match.group(2))
        grouped.setdefault(sep, set()).add(seed)
    for sep, found in sorted(grouped.items()):
        missing = sorted(set(expected_seeds) - found)
        rows.append(
            {
                "separation_px": sep,
                "completed_seeds": len(found),
                "expected_seeds": len(expected_seeds),
                "missing_seeds": ",".join(str(seed) for seed in missing) if missing else "",
                "complete": len(missing) == 0,
            }
        )
    return pd.DataFrame(rows)


def build_separation_case_list(separations: list[int], seeds: list[int]) -> list[str]:
    return [
        build_separation_benchmark_case(separation, seed).name
        for separation in separations
        for seed in seeds
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic GMM benchmarks.")
    parser.add_argument("--data-root", type=Path, default=Path("synthetic_test_data"))
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--generate", action="store_true", help="Generate missing benchmark cases.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip pipeline runs when CSV exists.")
    parser.add_argument("--benchmark", choices=[
        "stage2_separation",
        "separation",
        "false_split",
        "brightness_ratio",
        "sigma",
        "compare_search_modes",
        "all",
    ], default="stage2_separation")
    parser.add_argument("--search-mode", choices=["full", "staged_early_stop"], default="full")
    parser.add_argument("--evaluate-only", action="store_true", help="Summarize existing result CSVs without rerunning pipeline.")
    parser.add_argument("--max-cases", type=int, default=0, help="Limit number of cases run/evaluated (0 = all).")
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Run/evaluate one case by exact name (overrides stage2 seed subset).",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    seeds = generate_seed_list(args.num_seeds)
    results_root = data_root / "results" / "benchmark_reports"
    results_root.mkdir(parents=True, exist_ok=True)
    all_records: list[BenchmarkRunRecord] = []

    if args.generate:
        generate_separation_benchmark(data_root, seeds=seeds)
        if args.benchmark in ("false_split", "all"):
            generate_false_split_benchmark(data_root, seeds=seeds[: min(5, len(seeds))])
        if args.benchmark in ("brightness_ratio", "all"):
            generate_brightness_ratio_benchmark(data_root, seeds=seeds)
        if args.benchmark in ("sigma", "all"):
            generate_sigma_benchmark(data_root, seeds=seeds)

    if args.benchmark in ("stage2_separation", "separation", "all"):
        case_names = discover_cases_by_prefix(data_root, "sep_benchmark_sep")
        if args.generate or not case_names:
            generate_separation_benchmark(data_root, seeds=seeds)
            case_names = discover_cases_by_prefix(data_root, "sep_benchmark_sep")
        target_cases = case_names
        if args.benchmark == "stage2_separation":
            target_cases = [
                name
                for name in case_names
                if any(name.endswith(f"_seed{seed}") for seed in generate_seed_list(3))
            ]
        target_cases = _select_target_cases(target_cases, args.case)
        completeness = verify_seed_completeness(
            target_cases,
            generate_seed_list(3) if args.benchmark == "stage2_separation" else seeds,
            prefix="sep_benchmark_sep",
        )
        completeness.to_csv(results_root / "separation_seed_completeness.csv", index=False)
        config = _stage2_config(search_mode="full")
        suffix = "stage2" if args.benchmark == "stage2_separation" else f"sep_{args.search_mode}"
        if args.max_cases > 0:
            target_cases = target_cases[: args.max_cases]
        if args.evaluate_only:
            records = []
            for case_name in target_cases:
                run_name = f"{case_name}_{suffix}"
                csv_path = data_root / "results" / run_name / f"{case_name}_measurements.csv"
                if not csv_path.is_file() and suffix == "stage2":
                    alt = data_root / "results" / f"{case_name}_stage2_full" / f"{case_name}_measurements.csv"
                    if alt.is_file():
                        run_name = f"{case_name}_stage2_full"
                        csv_path = alt
                if not csv_path.is_file():
                    continue
                extra = _extract_run_metadata(csv_path, {"gmm_init_diagnostics": [], "gmm_config": {}})
                metrics = evaluate_run(
                    run_name=run_name,
                    ground_truth_case=case_name,
                    measurements_stem=case_name,
                    data_root=data_root,
                )
                records.append(
                    BenchmarkRunRecord(
                        benchmark="separation",
                        case_name=case_name,
                        run_name=run_name,
                        metrics=metrics,
                        runtime_s=0.0,
                        search_mode=str(extra["search_mode"]),
                        winning_init_strategy=extra["winning_init_strategy"],
                        selected_n_components=extra["selected_n_components"],
                        spurious_split_rejected=bool(extra["spurious_split_rejected"]),
                        fallback_rate=bool(extra["fallback_rate"]),
                        two_component_selected=bool(extra["two_component_selected"]),
                        profiling=extra["profiling"],
                    )
                )
        else:
            records = run_benchmark_cases(
                benchmark="separation",
                case_names=target_cases,
                data_root=data_root,
                run_suffix=suffix,
                config=config,
                skip_existing=args.skip_existing,
            )
        all_records.extend(records)
        frame = records_to_frame(records)
        summary = summarize_separation_benchmark(frame, sigma=SEPARATION_BENCHMARK_SIGMA)
        frame.to_csv(results_root / f"separation_runs_{suffix}.csv", index=False)
        summary.to_csv(results_root / f"separation_summary_{suffix}.csv", index=False)

    if args.benchmark in ("false_split", "all"):
        case_names = discover_cases_by_prefix(data_root, "false_split_")
        if args.generate or len(case_names) < 10:
            generate_false_split_benchmark(data_root, seeds=seeds[: min(5, len(seeds))])
            case_names = discover_cases_by_prefix(data_root, "false_split_")
        records = run_benchmark_cases(
            benchmark="false_split",
            case_names=case_names,
            data_root=data_root,
            run_suffix="stage2_full",
            config=_stage2_config(search_mode="full"),
            skip_existing=args.skip_existing,
        )
        all_records.extend(records)
        frame = records_to_frame(records)
        summary = summarize_false_split_benchmark(frame)
        frame.to_csv(results_root / "false_split_runs.csv", index=False)
        summary.to_csv(results_root / "false_split_summary.csv", index=False)

    if args.benchmark in ("brightness_ratio", "all"):
        case_names = discover_cases_by_prefix(data_root, "ratio_benchmark_")
        if args.generate or not case_names:
            generate_brightness_ratio_benchmark(data_root, seeds=seeds)
            case_names = discover_cases_by_prefix(data_root, "ratio_benchmark_")
        records = run_benchmark_cases(
            benchmark="brightness_ratio",
            case_names=case_names,
            data_root=data_root,
            run_suffix="stage2_full",
            config=_stage2_config(search_mode="full"),
            skip_existing=args.skip_existing,
        )
        all_records.extend(records)
        frame = records_to_frame(records)
        summary = summarize_ratio_benchmark(frame)
        frame.to_csv(results_root / "ratio_runs.csv", index=False)
        summary.to_csv(results_root / "ratio_summary.csv", index=False)

    if args.benchmark in ("sigma", "all"):
        case_names = discover_cases_by_prefix(data_root, "sigma_benchmark_")
        if args.generate or not case_names:
            generate_sigma_benchmark(data_root, seeds=seeds)
            case_names = discover_cases_by_prefix(data_root, "sigma_benchmark_")
        records = run_benchmark_cases(
            benchmark="sigma",
            case_names=case_names,
            data_root=data_root,
            run_suffix="stage2_full",
            config=_stage2_config(search_mode="full"),
            skip_existing=args.skip_existing,
        )
        all_records.extend(records)
        frame = records_to_frame(records)
        summary = summarize_sigma_benchmark(frame)
        frame.to_csv(results_root / "sigma_runs.csv", index=False)
        summary.to_csv(results_root / "sigma_summary.csv", index=False)

    if args.benchmark in ("compare_search_modes", "all"):
        case_names = discover_cases_by_prefix(data_root, "sep_benchmark_sep")
        if not case_names:
            generate_separation_benchmark(data_root, seeds=generate_seed_list(3))
            case_names = discover_cases_by_prefix(data_root, "sep_benchmark_sep")
        subset = [
            name
            for name in case_names
            if any(name.endswith(f"_seed{seed}") for seed in (101, 202, 303))
        ]
        compare_rows: list[dict[str, Any]] = []
        for mode in ("full", "staged_early_stop"):
            records = run_benchmark_cases(
                benchmark=f"separation_{mode}",
                case_names=subset,
                data_root=data_root,
                run_suffix=f"compare_{mode}",
                config=_stage2_config(search_mode=mode),
                skip_existing=args.skip_existing,
            )
            all_records.extend(records)
            frame = records_to_frame(records)
            for case_name in subset:
                full_row = frame[frame["case_name"] == case_name]
                if full_row.empty:
                    continue
                compare_rows.append(
                    {
                        "case_name": case_name,
                        "search_mode": mode,
                        "pass_criterion": bool(full_row.iloc[0]["pass_criterion"]),
                        "predicted_count": int(full_row.iloc[0]["predicted_accepted_count"]),
                        "runtime_s": float(full_row.iloc[0]["runtime_s"]),
                        "winning_init_strategy": full_row.iloc[0]["winning_init_strategy"],
                    }
                )
        compare = pd.DataFrame(compare_rows)
        if not compare.empty:
            pivot = compare.pivot_table(
                index="case_name",
                columns="search_mode",
                values=["pass_criterion", "predicted_count", "runtime_s"],
                aggfunc="first",
            )
            pivot.to_csv(results_root / "search_mode_comparison.csv")

    profiling = summarize_strategy_profiling(all_records)
    if not profiling.empty:
        profiling.to_csv(results_root / "multi_start_strategy_profiling.csv", index=False)

    print(f"Wrote benchmark reports under {results_root}")


if __name__ == "__main__":
    main()
