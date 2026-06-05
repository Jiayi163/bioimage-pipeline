"""Run the CellProfiler-to-Fiji workflow (Phase 13)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bioimage_pipeline.analysis import run_cellprofiler_workflow


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CellProfiler .cppipe pipeline and organize results for "
            "Fiji/ImageJ inspection."
        )
    )
    parser.add_argument(
        "--cppipe",
        required=True,
        type=Path,
        help="Path to a CellProfiler .cppipe pipeline file.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Root folder for organized results.",
    )
    parser.add_argument(
        "--executable",
        default="cellprofiler",
        help="CellProfiler command or full path to CellProfiler.exe.",
    )
    parser.add_argument(
        "--no-fiji-export",
        action="store_true",
        help="Skip Fiji-compatible TIFF conversion for mask/label outputs.",
    )
    parser.add_argument(
        "--no-qc",
        action="store_true",
        help="Skip QC overlay generation.",
    )
    parser.add_argument(
        "--json-summary",
        action="store_true",
        help="Print the workflow summary as JSON.",
    )
    args = parser.parse_args()

    print("Running CellProfiler-to-Fiji workflow...")
    result = run_cellprofiler_workflow(
        args.input_dir,
        args.output_dir,
        args.cppipe,
        cellprofiler_executable=args.executable,
        export_fiji_tiffs=not args.no_fiji_export,
        generate_qc=not args.no_qc,
    )

    if args.json_summary:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Results directory: {result.results_dir}")
        print(f"Raw CellProfiler output: {result.raw_output_dir}")
        print(f"Processed images: {len(result.processed_images)}")
        for filename in result.processed_images:
            print(f"  - {filename}")
        print(f"Measurements: {result.measurements_dir}")
        print(f"Loaded tables: {', '.join(sorted(result.tables))}")
        for table_name, summary in sorted(result.table_summary.items()):
            print(
                f"  - {table_name}: {summary['rows']} rows, "
                f"{summary['columns']} columns"
            )
        if result.measurements is not None:
            print(f"Merged measurements: {len(result.measurements)} row(s)")
        print(f"Masks: {result.masks_dir} ({len(result.mask_exports)} file(s))")
        print(f"Labels: {result.labels_dir} ({len(result.label_exports)} file(s))")
        print(f"QC overlays: {result.qc_dir} ({len(result.qc_artifacts)} image(s))")
        print(f"Logs: {result.logs_dir}")

    print("Workflow complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
