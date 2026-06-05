"""Run Phase 10.1 CellProfiler integration validation (real pipeline required)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bioimage_pipeline.cellprofiler_runner import (
    load_cellprofiler_measurements,
    run_cellprofiler_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a real CellProfiler pipeline and verify CSV outputs."
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
        help="Folder containing input TIFF images.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Folder where CellProfiler writes outputs.",
    )
    parser.add_argument(
        "--executable",
        default="cellprofiler",
        help="CellProfiler command or full path to CellProfiler.exe.",
    )
    args = parser.parse_args()

    print("Running CellProfiler pipeline...")
    output_dir = run_cellprofiler_pipeline(
        args.cppipe,
        args.input_dir,
        args.output_dir,
        cellprofiler_executable=args.executable,
    )
    print(f"Output directory: {output_dir}")

    load_result = load_cellprofiler_measurements(output_dir)
    if load_result.warnings:
        print("Import warnings:")
        for warning in load_result.warnings:
            print(f"  - {warning}")
    print(f"Loaded {len(load_result.tables)} CSV file(s):")
    for name, dataframe in load_result.tables.items():
        print(f"  - {name}: {len(dataframe)} rows, {len(dataframe.columns)} columns")

    print("Validation succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
