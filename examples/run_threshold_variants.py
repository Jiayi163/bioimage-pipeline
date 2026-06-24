"""Run subset-first threshold parameter assistant trials and confirmed full applies.

Phase 17 orchestration CLI:

1. ``trial`` (default) — stage a subset, run candidate variants, screen with heuristics
2. ``apply`` — after user review, run one chosen variant on the full dataset

The imported pipeline file is never modified. Nothing is auto-applied without ``--confirm``.
Heuristic rankings are screening aids only, not ground-truth optimality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bioimage_pipeline.threshold_extraction import (
    load_identify_primary_objects_threshold_profiles,
)
from bioimage_pipeline.threshold_recommender import (
    ThresholdRecommenderConfig,
    apply_confirmed_threshold_variant,
    load_trial_result_from_session,
    run_threshold_recommender_trial,
)
from bioimage_pipeline.threshold_subset import ThresholdSubsetSelection
from bioimage_pipeline.threshold_variant_comparison import (
    ThresholdVariantSizeThresholds,
    threshold_variant_comparison_to_dataframe,
)
from bioimage_pipeline.threshold_variant_scoring import (
    threshold_variant_ranking_to_dataframe,
)
from bioimage_pipeline.threshold_variant_gt_scoring import (
    ground_truth_variant_scores_to_dataframe,
)


def _print_profiles(cppipe_path: Path) -> None:
    profiles = load_identify_primary_objects_threshold_profiles(cppipe_path)
    if not profiles:
        print("No IdentifyPrimaryObjects modules found.")
        return
    print(f"Found {len(profiles)} IdentifyPrimaryObjects module(s):")
    for profile in profiles:
        print(
            f"  - {profile.display_name}: "
            f"strategy={profile.threshold_strategy}, "
            f"method={profile.thresholding_method}, "
            f"correction={profile.threshold_correction_factor}"
        )


def _config_from_args(args: argparse.Namespace) -> ThresholdRecommenderConfig:
    subset_selection = ThresholdSubsetSelection(
        mode="manual" if args.subset_images else "auto",
        sample_count=args.subset_count,
        sample_method=args.subset_method,
        random_seed=args.subset_seed,
    )
    return ThresholdRecommenderConfig(
        imported_cppipe_path=args.cppipe.resolve(),
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        cellprofiler_executable=args.executable,
        generate_qc=not args.no_qc,
        strict=args.strict,
        max_variants=args.max_variants,
        ipo_module_index=args.ipo_module_index,
        ipo_module_num=args.ipo_module_num,
        ipo_object_name=args.ipo_object_name,
        subset_selection=subset_selection,
        manual_subset_image_names=list(args.subset_images or []),
        size_thresholds=ThresholdVariantSizeThresholds(
            tiny_area_px=args.tiny_area_px,
            huge_area_px=args.huge_area_px,
        ),
        full_dataset_trial=args.full_dataset_trial,
        fast_optimistic=not args.no_fast_optimistic,
        force_full_search=args.force_full_search,
        reference_mask_dir=(
            args.reference_mask_dir.resolve()
            if getattr(args, "reference_mask_dir", None) is not None
            else None
        ),
        ground_truth_match_iou_threshold=getattr(
            args, "ground_truth_match_iou_threshold", 0.3
        ),
    )


def _print_trial_summary(trial_result: object, *, json_summary: bool) -> None:
    if json_summary:
        payload = {
            "recommender_root": str(trial_result.recommender_root),
            "subset_dir": str(trial_result.subset_dir),
            "subset_images": trial_result.subset_manifest.image_names,
            "comparison_csv": str(trial_result.comparison_paths["csv"]),
            "ranking_csv": str(trial_result.ranking_paths["csv"]),
            "comparison": threshold_variant_comparison_to_dataframe(
                trial_result.summaries
            ).to_dict(orient="records"),
            "ranking": threshold_variant_ranking_to_dataframe(
                trial_result.ranked_scores
            ).to_dict(orient="records"),
        }
        print(json.dumps(payload, indent=2))
        return

    print(f"Recommender root: {trial_result.recommender_root}")
    print(
        f"Subset: {len(trial_result.subset_manifest.image_names)} image(s) "
        f"in {trial_result.subset_dir}"
    )
    if trial_result.comparison_paths.get("csv"):
        print(f"Comparison CSV: {trial_result.comparison_paths['csv']}")
    if trial_result.ranking_paths.get("csv"):
        print(f"Ranking CSV: {trial_result.ranking_paths['csv']}")
    print()
    print("Comparison:")
    print(
        threshold_variant_comparison_to_dataframe(trial_result.summaries).to_string(
            index=False
        )
    )
    print()
    print("Ranking:")
    if trial_result.ranked_scores:
        ranking_table = threshold_variant_ranking_to_dataframe(trial_result.ranked_scores)
        print(
            ranking_table[
                ["rank", "variant_id", "name", "score", "reason"]
            ].to_string(index=False)
        )
        top = trial_result.ranked_scores[0]
        print()
        print("Top candidate (not auto-applied; review previews and per-image QC first):")
        print(f"  {top.variant_id} — {top.display_name} (score={top.score:.2f})")
        print(f"  Reason: {top.reason}")
        for line in top.explanations[:3]:
            print(f"    - {line}")
        print()
        print(
            "To apply on the full dataset after review:\n"
            f"  python examples/run_threshold_variants.py apply "
            f"--output-dir \"{trial_result.recommender_root.parent}\" "
            f"--input-dir \"{trial_result.subset_manifest.source_dir}\" "
            f"--variant-id {top.variant_id} --confirm"
        )
    else:
        print("  (no ranked variants)")

    if getattr(trial_result, "gt_ranked_scores", None):
        gt_scores = trial_result.gt_ranked_scores
        if gt_scores:
            print()
            print("Ground-truth ranking:")
            print(
                ground_truth_variant_scores_to_dataframe(gt_scores)[
                    ["gt_rank", "variant_id", "name", "gt_score", "gt_label", "mean_f1"]
                ].to_string(index=False)
            )
            top_gt = gt_scores[0]
            print()
            print("Top GT match (review previews before applying):")
            print(
                f"  {top_gt.variant_id} — {top_gt.display_name} "
                f"(gt_score={top_gt.gt_score:.2f}, label={top_gt.gt_label})"
            )
            gt_paths = getattr(trial_result, "ground_truth_comparison_paths", {}) or {}
            if gt_paths.get("ranking_csv"):
                print(f"GT ranking CSV: {gt_paths['ranking_csv']}")


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Root output folder (recommender data under threshold_recommender/).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Full input image folder.",
    )
    parser.add_argument(
        "--executable",
        default="cellprofiler",
        help="CellProfiler command or full path to CellProfiler.exe.",
    )
    parser.add_argument(
        "--tiny-area-px",
        type=float,
        default=2.0,
        help="Area threshold (pixel^2) below which objects count as tiny.",
    )
    parser.add_argument(
        "--huge-area-px",
        type=float,
        default=200.0,
        help="Area threshold (pixel^2) above which objects count as huge.",
    )
    parser.add_argument(
        "--no-qc",
        action="store_true",
        help="Skip QC overlay generation during CellProfiler runs.",
    )


def _add_trial_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cppipe",
        required=True,
        type=Path,
        help="Path to the imported CellProfiler .cppipe pipeline file.",
    )
    parser.add_argument(
        "--ipo-module-index",
        type=int,
        help="IdentifyPrimaryObjects module index when multiple IPO modules exist.",
    )
    parser.add_argument(
        "--ipo-module-num",
        type=int,
        help="IdentifyPrimaryObjects module_num when multiple IPO modules exist.",
    )
    parser.add_argument(
        "--ipo-object-name",
        help="IPO object name when multiple IPO modules exist (e.g. Spots).",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        help="Limit the number of generated/run variants.",
    )
    parser.add_argument(
        "--subset-count",
        type=int,
        default=5,
        help="Number of images for auto-sampled subset trial (default: 5).",
    )
    parser.add_argument(
        "--subset-method",
        choices=("even", "first", "random"),
        default="even",
        help="Auto-sampling method for subset selection (default: even).",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        help="Random seed when --subset-method random.",
    )
    parser.add_argument(
        "--subset-images",
        nargs="+",
        help="Explicit image filenames for a manual subset trial.",
    )
    parser.add_argument(
        "--full-dataset-trial",
        action="store_true",
        help="Debug/power-user: run variant trial on the full input folder.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop the variant batch when one CellProfiler run fails.",
    )
    parser.add_argument(
        "--json-summary",
        action="store_true",
        help="Print comparison and ranking tables as JSON.",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Reload an existing recommender session and print ranking outputs.",
    )
    parser.add_argument(
        "--no-fast-optimistic",
        action="store_true",
        help="Skip the fast optimistic single-candidate trial and run full variant search.",
    )
    parser.add_argument(
        "--force-full-search",
        action="store_true",
        help=(
            "Run the full multi-variant search even when the optimistic candidate "
            "passes QC (optimistic trial still runs for comparison)."
        ),
    )
    parser.add_argument(
        "--reference-mask-dir",
        type=Path,
        help=(
            "Folder with lab-approved reference masks named "
            "<image_stem>_reference_mask.tif for ground-truth scoring."
        ),
    )
    parser.add_argument(
        "--ground-truth-match-iou-threshold",
        type=float,
        default=0.3,
        help="Minimum object IoU to count as a true positive (default: 0.3).",
    )


def _add_apply_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--variant-id",
        required=True,
        help="Variant ID from the trial ranking (e.g. 002_otsu_global).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required flag confirming full-dataset apply.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Subset-first threshold recommender: trial candidate variants on a "
            "small image subset, rank results, then optionally apply one variant "
            "to the full dataset after explicit confirmation."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    trial_parser = subparsers.add_parser(
        "trial",
        help="Run subset trial (default command).",
    )
    _add_shared_arguments(trial_parser)
    _add_trial_arguments(trial_parser)

    apply_parser = subparsers.add_parser(
        "apply",
        help="Apply one confirmed variant to the full dataset.",
    )
    _add_shared_arguments(apply_parser)
    _add_apply_arguments(apply_parser)

    return parser


def _run_trial(args: argparse.Namespace) -> int:
    if args.compare_only:
        trial_result = load_trial_result_from_session(args.output_dir)
        _print_trial_summary(trial_result, json_summary=args.json_summary)
        return 0

    if args.input_dir is None:
        raise SystemExit("--input-dir is required for trial runs.")

    cppipe_path = args.cppipe.resolve()
    print(f"Reading threshold settings from: {cppipe_path}")
    _print_profiles(cppipe_path)

    config = _config_from_args(args)
    print(
        f"Running subset trial on {config.subset_selection.sample_count} image(s) "
        f"({config.subset_selection.sample_method} sampling) unless manual subset specified."
    )
    if config.fast_optimistic:
        print("Fast optimistic mode enabled: trying one Otsu adaptive candidate first.")
    if config.force_full_search:
        print("Force full search enabled: will not accept optimistic candidate.")
    if config.reference_mask_dir is not None:
        print(f"Ground-truth scoring enabled: {config.reference_mask_dir}")
    trial_result = run_threshold_recommender_trial(config)

    failed = sum(1 for result in trial_result.run_results if not result.success)
    successful = len(trial_result.run_results) - failed
    print(f"Variant runs: {successful} succeeded, {failed} failed.")
    if trial_result.trial_mode == "optimistic":
        print("Optimistic candidate passed basic heuristic screening on the subset.")
        if trial_result.optimistic_qc_path is not None:
            print(f"Optimistic QC report: {trial_result.optimistic_qc_path}")
    elif trial_result.fell_back_to_full_search:
        print("Optimistic screening failed; fell back to full multi-variant search.")
        if trial_result.optimistic_qc_path is not None:
            print(f"Optimistic QC report: {trial_result.optimistic_qc_path}")
    elif trial_result.forced_full_search:
        print("Optimistic screening passed but full variant search was forced.")
        if trial_result.optimistic_qc_path is not None:
            print(f"Optimistic QC report: {trial_result.optimistic_qc_path}")

    if trial_result.subset_characterization_paths.get("csv"):
        print(
            "Subset characterization CSV: "
            f"{trial_result.subset_characterization_paths['csv']}"
        )
    if trial_result.per_image_comparison_paths.get("csv"):
        print(
            "Per-image comparison CSV: "
            f"{trial_result.per_image_comparison_paths['csv']}"
        )

    _print_trial_summary(trial_result, json_summary=args.json_summary)
    print("Subset trial complete.")
    return 0 if failed == 0 else 1


def _run_apply(args: argparse.Namespace) -> int:
    if not args.confirm:
        print(
            "Refusing to apply variant to the full dataset without --confirm.",
            file=sys.stderr,
        )
        return 1
    if args.input_dir is None:
        print("--input-dir is required for apply.", file=sys.stderr)
        return 1

    config = ThresholdRecommenderConfig(
        imported_cppipe_path=Path("unused-for-apply"),
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        cellprofiler_executable=args.executable,
        generate_qc=not args.no_qc,
    )
    apply_result = apply_confirmed_threshold_variant(
        config,
        args.variant_id,
        confirmed=True,
    )

    if apply_result.run_result.success:
        print(f"Confirmed full run complete: {apply_result.confirmed_run_dir}")
        print(f"Selection record: {apply_result.selection_path}")
        return 0

    print(
        "Confirmed full run failed: "
        f"{apply_result.run_result.error_message or 'unknown error'}",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] not in {"trial", "apply", "-h", "--help"}:
        trial_argv = ["trial", *argv]
    else:
        trial_argv = argv

    parser = _build_parser()
    args = parser.parse_args(trial_argv)
    command = args.command or "trial"

    try:
        if command == "apply":
            return _run_apply(args)
        return _run_trial(args)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
