#!/usr/bin/env python3
"""Run a ground-truth oracle GMM experiment on one synthetic case.

This uses known synthetic spot centers for initialization only. It does not
modify production pipeline behavior.

Usage (from project root):
    python scripts/run_gmm_oracle_experiment.py --case sep_benchmark_sep3_seed1010
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.validation.gmm_oracle import (
    format_oracle_report,
    run_ground_truth_oracle_experiment,
    write_oracle_report,
)


def _stage2_config() -> PunctaDeclumpConfig:
    return PunctaDeclumpConfig(
        gmm_multi_start_enabled=True,
        gmm_use_mixture_acceptance_separation=True,
        gmm_acceptance_min_separation=1.5,
        gmm_multi_start_mode="full",
        enable_selective_routing=False,
        diagnostic_mode="summary",
        export_fiji_tiffs=False,
        candidate_detector="python_log",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ground-truth oracle GMM diagnostic experiment (synthetic only).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("synthetic_test_data"),
        help="Synthetic dataset root containing images/, masks/, ground_truth/.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="sep_benchmark_sep3_seed1010",
        help="Synthetic case name to probe with ground-truth initialization.",
    )
    parser.add_argument(
        "--object-index",
        type=int,
        default=0,
        help="Mask object index to fit (default: first object).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for oracle report JSON (default: data_root/results/oracle/<case>).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir or (data_root / "results" / "oracle" / args.case)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run_ground_truth_oracle_experiment(
        data_root=data_root,
        case_name=args.case,
        config=_stage2_config(),
        object_index=args.object_index,
    )
    json_path = write_oracle_report(report, output_dir / f"{args.case}_oracle_report.json")
    print(format_oracle_report(report))
    print(f"\nWrote {json_path}")


if __name__ == "__main__":
    main()
