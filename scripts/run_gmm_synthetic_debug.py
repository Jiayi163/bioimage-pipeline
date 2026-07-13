#!/usr/bin/env python3
"""Run focused GMM diagnostics and ablation on synthetic validation cases.

Usage (from project root):
    python scripts/run_gmm_synthetic_debug.py --basic-cases
    python scripts/run_gmm_synthetic_debug.py --case case3_overlapping_forced_gmm --print-case3-report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bioimage_pipeline.puncta.validation.gmm_probe import (
    GmmDiagnosticProbe,
    write_probe_outputs,
)

# run_name -> ground_truth_case (image/mask folder)
BASIC_DEBUG_CASES: dict[str, str] = {
    "case1_isolated": "case1_isolated",
    "case2_separated": "case2_separated",
    "case3_overlapping_normal": "case3_overlapping",
    "case3_overlapping_forced_gmm": "case3_overlapping",
}


def print_case3_report(report_dict: dict) -> None:
    print("\n=== case3_overlapping_forced_gmm detailed report ===")
    selection = report_dict.get("model_selection", {})
    print(f"Detected peaks: {report_dict.get('n_detected_peaks')}")
    print(f"Peak coords (x,y): {report_dict.get('detected_peak_coords')}")
    print(f"Balanced model attempted n=2: {selection.get('balanced_model_attempted_n2')}")
    print(f"Balanced model attempted n=3: {selection.get('balanced_model_attempted_n3')}")
    print(f"n=3 gate: {selection.get('balanced_model_n3_gate_reason')}")
    print(f"Single BIC/AIC: {selection.get('single_bic')} / {selection.get('single_aic')}")
    print(f"Best mixture n: {selection.get('best_mixture_n_components')}")
    print(f"Best mixture BIC/AIC: {selection.get('best_mixture_bic')} / {selection.get('best_mixture_aic')}")
    print(f"BIC delta (2 vs 1): {selection.get('bic_delta_2_vs_1')}")
    print(f"AIC delta (2 vs 1): {selection.get('aic_delta_2_vs_1')}")
    print(f"Selection reason: {selection.get('selection_reason')}")
    print(f"Exact second-component rejection: {selection.get('exact_second_component_rejection')}")

    two_comp = [
        attempt
        for attempt in report_dict.get("model_attempts", [])
        if attempt.get("n_components") == 2 and attempt.get("converged")
    ]
    if two_comp:
        best = min(two_comp, key=lambda row: row.get("bic") or float("inf"))
        print("\nBest converged 2-component attempt:")
        print(f"  init: {best.get('initialization_method')}")
        print(f"  fitted centers (x,y): {best.get('fitted_centers')}")
        print(f"  amplitudes: {best.get('fitted_amplitudes')}")
        print(f"  sigma_x: {best.get('fitted_sigma_x')}")
        print(f"  sigma_y: {best.get('fitted_sigma_y')}")
        print(f"  pairwise distances: {best.get('pairwise_component_distances_px')}")
        print(f"  post-merge count: {best.get('post_merge_component_count')}")
        print(f"  merge notes: {best.get('merge_notes')}")
        print(f"  component rejections: {best.get('component_rejection_reasons')}")

    print("\nAblation summary:")
    for row in report_dict.get("ablation_results", []):
        print(
            f"  {row.get('mode')}: accepted={row.get('predicted_accepted_count')} "
            f"init={row.get('init_strategy')} filters={row.get('filter_mode')} "
            f"second_rejection={row.get('second_component_rejection')}"
        )

    init_methods = sorted(
        {
            attempt.get("initialization_method")
            for attempt in report_dict.get("model_attempts", [])
            if attempt.get("n_components") == 2
        }
    )
    print(f"\nAll 2-component initialization strategies attempted: {init_methods}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GMM synthetic debug and ablation runner.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("synthetic_test_data"),
    )
    parser.add_argument("--case", action="append", dest="cases", help="Run folder name under results/.")
    parser.add_argument("--basic-cases", action="store_true", help="Debug the four basic validation runs.")
    parser.add_argument(
        "--print-case3-report",
        action="store_true",
        help="Print expanded case3_overlapping_forced_gmm diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    probe = GmmDiagnosticProbe()

    if args.basic_cases:
        selected = list(BASIC_DEBUG_CASES.keys())
    elif args.cases:
        selected = args.cases
    else:
        selected = ["case3_overlapping_forced_gmm"]

    ablation_summaries = []
    for run_name in selected:
        gt_case = BASIC_DEBUG_CASES.get(run_name, run_name)
        report = probe.probe_synthetic_run(
            data_root=data_root,
            run_name=run_name,
            ground_truth_case=gt_case,
        )
        out_dir = data_root / "results" / run_name
        json_path, csv_path = write_probe_outputs(report, out_dir)
        print(f"{run_name}: wrote {json_path.name}, {csv_path.name}")
        print(
            f"  peaks={report.n_detected_peaks} "
            f"n2_attempted={report.model_selection.balanced_model_attempted_n2} "
            f"second_rejection={report.model_selection.exact_second_component_rejection}"
        )
        ablation_summaries.append(
            {
                "run_name": run_name,
                "n_detected_peaks": report.n_detected_peaks,
                "balanced_model_attempted_n2": report.model_selection.balanced_model_attempted_n2,
                "exact_second_component_rejection": report.model_selection.exact_second_component_rejection,
                "ablation_results": [row.__dict__ for row in report.ablation_results],
            }
        )

        if args.print_case3_report and run_name == "case3_overlapping_forced_gmm":
            print_case3_report(report.to_dict())

    ablation_path = data_root / "results" / "gmm_ablation_summary.json"
    ablation_path.write_text(json.dumps(ablation_summaries, indent=2, default=str), encoding="utf-8")
    print(f"Wrote ablation summary: {ablation_path}")


if __name__ == "__main__":
    main()
