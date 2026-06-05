"""Run Phase 10.4 real-data validation on microscopy TIFF images."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bioimage_pipeline.analysis import run_analysis
from bioimage_pipeline.io import read_tiff
from bioimage_pipeline.validation import (
    build_validation_report,
    compare_masks,
    inspect_image,
    write_validation_report,
)


def _collect_tiffs(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.glob("*.tif"))
    paths.extend(sorted(input_dir.glob("*.tiff")))
    return [path for path in paths if path.is_file()]


def _print_properties(image_path: Path) -> None:
    properties = inspect_image(read_tiff(image_path))
    print(f"Image: {image_path.name}")
    print(f"  shape: {properties.shape}")
    print(f"  dtype: {properties.dtype}")
    print(f"  estimated_snr: {properties.estimated_snr}")
    if properties.limitations:
        print("  limitations:")
        for item in properties.limitations:
            print(f"    - {item}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate real microscopy data with Python and optional CellProfiler."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing real TIFF images.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder where analysis and validation reports are written.",
    )
    parser.add_argument(
        "--reference-mask-dir",
        help="Optional folder with reference masks named <image_stem>_reference_mask.tif.",
    )
    parser.add_argument(
        "--engine",
        choices=("python", "cellprofiler", "both"),
        default="python",
        help="Engine(s) to run (default: python).",
    )
    parser.add_argument(
        "--cppipe",
        help="CellProfiler .cppipe file (required when engine includes cellprofiler).",
    )
    parser.add_argument(
        "--executable",
        default="cellprofiler",
        help="CellProfiler command or full path to CellProfiler.exe.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    reference_mask_dir = (
        Path(args.reference_mask_dir) if args.reference_mask_dir else None
    )

    if args.engine in {"cellprofiler", "both"} and not args.cppipe:
        parser.error("--cppipe is required when --engine is cellprofiler or both")

    python_output = output_dir / "python"
    cellprofiler_output = output_dir / "cellprofiler"
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Inspecting input images...")
    for image_path in _collect_tiffs(input_dir):
        _print_properties(image_path)

    python_result = None
    if args.engine in {"python", "both"}:
        print("\nRunning Python engine...")
        python_result = run_analysis(
            input_dir,
            python_output,
            analysis_engine="python",
        )
        print(f"Processed: {len(python_result['processed'])} image(s)")

    cellprofiler_result = None
    if args.engine in {"cellprofiler", "both"}:
        print("\nRunning CellProfiler engine...")
        cellprofiler_result = run_analysis(
            input_dir,
            cellprofiler_output,
            analysis_engine="cellprofiler",
            cppipe_path=args.cppipe,
            cellprofiler_executable=args.executable,
        )
        print(
            "Loaded tables: "
            + ", ".join(sorted(cellprofiler_result["tables"]))
        )

    print("\nBuilding validation reports...")
    for image_path in _collect_tiffs(input_dir):
        stem = image_path.stem
        python_mask_path = python_output / f"{stem}_mask.tif"
        reference_mask_path = None
        if reference_mask_dir is not None:
            candidate = reference_mask_dir / f"{stem}_reference_mask.tif"
            if candidate.exists():
                reference_mask_path = candidate

        cellprofiler_measurements = None
        if cellprofiler_result is not None:
            cellprofiler_measurements = cellprofiler_result["measurements"]

        report = build_validation_report(
            image_path=image_path,
            python_mask_path=python_mask_path if python_mask_path.exists() else None,
            reference_mask_path=reference_mask_path,
            cellprofiler_measurements=cellprofiler_measurements,
        )

        if (
            python_mask_path.exists()
            and reference_mask_path is not None
        ):
            python_mask = read_tiff(python_mask_path) > 0
            reference_mask = read_tiff(reference_mask_path) > 0
            comparison = compare_masks(python_mask, reference_mask)
            print(
                f"{stem}: IoU={comparison.iou:.3f}, "
                f"objects={comparison.object_count_a}/{comparison.object_count_b}"
            )

        report_path = write_validation_report(
            report,
            reports_dir / f"{stem}_validation.json",
        )
        print(f"Wrote report: {report_path}")

    if cellprofiler_result is not None and cellprofiler_result["measurements"] is not None:
        merged_path = output_dir / "cellprofiler_merged_measurements.csv"
        cellprofiler_result["measurements"].to_csv(merged_path, index=False)
        print(f"Saved merged CellProfiler measurements: {merged_path}")

    print("\nReal-data validation complete.")


if __name__ == "__main__":
    main()
