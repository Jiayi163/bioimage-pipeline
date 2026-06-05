"""Run unified analysis in Python or CellProfiler mode."""

from __future__ import annotations

import argparse
from pathlib import Path

from bioimage_pipeline.analysis import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bioimage analysis with the Python or CellProfiler engine."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing input TIFF images.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder where analysis outputs are written.",
    )
    parser.add_argument(
        "--engine",
        choices=("python", "cellprofiler"),
        default="python",
        help="Analysis engine to use (default: python).",
    )
    parser.add_argument(
        "--cppipe",
        help="Path to a CellProfiler .cppipe file (required for cellprofiler engine).",
    )
    parser.add_argument(
        "--executable",
        default="cellprofiler",
        help="CellProfiler command or full path to CellProfiler.exe.",
    )
    args = parser.parse_args()

    if args.engine == "cellprofiler" and not args.cppipe:
        parser.error("--cppipe is required when --engine is cellprofiler")

    result = run_analysis(
        args.input_dir,
        args.output_dir,
        analysis_engine=args.engine,
        cppipe_path=args.cppipe,
        cellprofiler_executable=args.executable,
    )

    print(f"Engine: {result['analysis_engine']}")
    print(f"Output: {result['output_dir']}")

    if result["analysis_engine"] == "python":
        print(f"Processed: {len(result['processed'])} image(s)")
        if result["failed"]:
            print(f"Failed: {len(result['failed'])} image(s)")
    else:
        print(f"Loaded tables: {', '.join(sorted(result['tables']))}")
        if result["measurements"] is not None:
            print(f"Merged measurements: {len(result['measurements'])} row(s)")


if __name__ == "__main__":
    main()
