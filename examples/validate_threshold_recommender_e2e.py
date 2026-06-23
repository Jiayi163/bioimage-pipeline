"""Phase 17.6 smoke test: real CellProfiler subset trial + optional confirmed apply.

Generates synthetic spot TIFFs and a minimal ``.cppipe`` when needed, runs the
subset-first threshold recommender against a real CellProfiler install, and
checks that ranking artifacts are produced.

Example:

    python examples/validate_threshold_recommender_e2e.py ^
        --output-dir path\\to\\e2e_output

Use your own assay data instead (recommended for real validation):

    python examples/validate_threshold_recommender_e2e.py ^
        --cppipe path\\to\\assay.cppipe ^
        --input-dir path\\to\\images ^
        --output-dir path\\to\\e2e_output

Synthetic auto-fixtures are useful for smoke testing pipeline generation, but
some CellProfiler installs may still report ``Empty image set list`` until you
validate with your real assay ``.cppipe`` and lab image folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bioimage_pipeline.threshold_recommender import (
    ThresholdRecommenderConfig,
    ThresholdRecommenderTrialResult,
    apply_confirmed_threshold_variant,
    run_threshold_recommender_trial,
)
from bioimage_pipeline.threshold_recommender_e2e import (
    materialize_threshold_recommender_e2e_fixtures,
    validate_threshold_recommender_trial_result,
)
from bioimage_pipeline.threshold_subset import ThresholdSubsetSelection


def _print_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    print("Warnings:")
    for warning in warnings:
        print(f"  - {warning}")
    print()


def _run_trial(args: argparse.Namespace) -> ThresholdRecommenderTrialResult:
    if args.fixtures_dir is not None:
        fixtures_root = args.fixtures_dir.resolve()
    else:
        fixtures_root = args.output_dir.resolve() / "_e2e_fixtures"

    if args.cppipe is None or args.input_dir is None:
        fixtures = materialize_threshold_recommender_e2e_fixtures(
            fixtures_root,
            image_count=args.image_count,
            force=args.regenerate_fixtures,
        )
        cppipe_path = fixtures.cppipe_path
        input_dir = fixtures.input_dir
        print(f"Using generated fixtures in {fixtures.root}")
    else:
        cppipe_path = args.cppipe.resolve()
        input_dir = args.input_dir.resolve()

    config = ThresholdRecommenderConfig(
        imported_cppipe_path=cppipe_path,
        input_dir=input_dir,
        output_dir=args.output_dir.resolve(),
        cellprofiler_executable=args.executable,
        generate_qc=not args.no_qc,
        max_variants=args.max_variants,
        subset_selection=ThresholdSubsetSelection(
            mode="auto",
            sample_count=args.subset_count,
            sample_method="first",
        ),
        fast_optimistic=not args.no_fast_optimistic,
    )

    print("Running threshold recommender subset trial...")
    print(f"  Pipeline: {cppipe_path}")
    print(f"  Input:    {input_dir}")
    print(f"  Output:   {args.output_dir.resolve()}")
    print(f"  Fast optimistic: {config.fast_optimistic}")
    if config.max_variants is not None:
        print(f"  Max variants: {config.max_variants}")

    return run_threshold_recommender_trial(config)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a real CellProfiler threshold recommender E2E smoke test.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Workflow output folder (recommender data under threshold_recommender/).",
    )
    parser.add_argument(
        "--cppipe",
        type=Path,
        help="Imported assay pipeline. Omit to use generated synthetic fixtures.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Input image folder. Omit to use generated synthetic fixtures.",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        help="Where to write synthetic fixtures when --cppipe/--input-dir are omitted.",
    )
    parser.add_argument(
        "--executable",
        default="C:\\Program Files\\CellProfiler\\CellProfiler.exe",
        help="CellProfiler command or full path to CellProfiler.exe.",
    )
    parser.add_argument(
        "--subset-count",
        type=int,
        default=3,
        help="Number of subset images for the trial (default: 3).",
    )
    parser.add_argument(
        "--image-count",
        type=int,
        default=3,
        help="Synthetic fixture image count when generating fixtures (default: 3).",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=3,
        help="Cap variant search breadth during E2E (default: 3).",
    )
    parser.add_argument(
        "--no-fast-optimistic",
        action="store_true",
        help="Disable optimistic-first trial (runs broader variant search).",
    )
    parser.add_argument(
        "--no-qc",
        action="store_true",
        help="Skip QC overlay generation during CellProfiler runs.",
    )
    parser.add_argument(
        "--regenerate-fixtures",
        action="store_true",
        help="Regenerate synthetic TIFFs and pipeline even if they already exist.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="After a successful trial, apply the top-ranked variant to the full input folder.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON summary to stdout.",
    )
    args = parser.parse_args()

    if (args.cppipe is None) != (args.input_dir is None):
        parser.error("Provide both --cppipe and --input-dir, or omit both for synthetic fixtures.")

    try:
        trial_result = _run_trial(args)
    except Exception as exc:
        print(f"Subset trial failed: {exc}", file=sys.stderr)
        return 1

    try:
        warnings = validate_threshold_recommender_trial_result(trial_result)
    except ValueError as exc:
        print(f"Trial validation failed:\n{exc}", file=sys.stderr)
        return 1

    _print_warnings(warnings)

    top = trial_result.ranked_scores[0]
    print("Subset trial passed validation.")
    print(f"  Trial mode: {trial_result.trial_mode}")
    print(f"  Top variant: {top.variant_id} ({top.display_name}) score={top.score:.2f}")
    print(f"  Ranking CSV: {trial_result.ranking_paths['csv']}")
    print(f"  Session: {trial_result.session_path}")

    apply_result = None
    if args.apply:
        from bioimage_pipeline.threshold_recommender import load_recommender_session

        session = load_recommender_session(trial_result.recommender_root)
        config = ThresholdRecommenderConfig(
            imported_cppipe_path=Path(session["imported_cppipe_path"]),
            input_dir=Path(session["subset_manifest"]["source_dir"]),
            output_dir=trial_result.recommender_root.parent,
            cellprofiler_executable=args.executable,
            generate_qc=not args.no_qc,
        )
        source_dir = Path(session["subset_manifest"]["source_dir"])
        image_count = len(list(source_dir.glob("*.tif")))
        print(
            f"Applying {top.variant_id} to full dataset "
            f"({image_count} image(s) in {source_dir})..."
        )
        apply_result = apply_confirmed_threshold_variant(
            config,
            top.variant_id,
            confirmed=True,
        )
        if not apply_result.run_result.success:
            print(
                f"Confirmed apply failed: {apply_result.run_result.error_message}",
                file=sys.stderr,
            )
            return 1
        print(f"Confirmed apply complete: {apply_result.confirmed_run_dir}")

    if args.json:
        payload = {
            "trial_mode": trial_result.trial_mode,
            "top_variant_id": top.variant_id,
            "top_score": top.score,
            "ranking_csv": str(trial_result.ranking_paths["csv"]),
            "session_path": str(trial_result.session_path),
            "warnings": warnings,
            "confirmed_run_dir": (
                str(apply_result.confirmed_run_dir) if apply_result is not None else None
            ),
        }
        print(json.dumps(payload, indent=2))

    print("Threshold recommender E2E validation succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
