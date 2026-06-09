"""Batch Z-max projection for Olympus .oir files.

Matches the Fiji macro ``Stacking+Drectly.ijm``:
recursively find ``.oir`` files, Z-max project, save TIFFs to an output folder.

The Fiji engine writes a manual-run macro that uses Bio-Formats Windowless
Importer, then Z Project Max Intensity, then saves ``.tif`` outputs. Run the
generated macro from the Fiji GUI to avoid command-line Bio-Formats importer
crashes.

Examples
--------
Generate a Fiji GUI macro (default)::

    python examples/run_oir_zmax_batch.py \\
        --input C:/path/to/oir_folder \\
        --output C:/path/to/results

Python fallback (experimental; requires working aicsimageio/bfio Java backend)::

    python examples/run_oir_zmax_batch.py \\
        --input C:/path/to/oir_folder \\
        --output C:/path/to/results \\
        --engine python
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bioimage_pipeline.oir_zmax_batch import run_oir_zmax_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Z-max project Olympus .oir files via Fiji/ImageJ "
            "(Stacking+Drectly.ijm parity)."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Root folder to search recursively for .oir files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Folder where projected TIFF files are written.",
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "fiji", "python"),
        default="fiji",
        help=(
            "Processing engine. Default: fiji (Bio-Formats Windowless Importer). "
            "auto also prefers Fiji and errors if Fiji is missing."
        ),
    )
    parser.add_argument(
        "--fiji",
        default=None,
        help="Accepted for compatibility, but not used for OIR GUI-macro generation.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Accepted for compatibility. OIR import now uses a manual-run Fiji "
            "GUI macro instead of command-line macro execution."
        ),
    )
    return parser


def _print_result_summary(result) -> None:
    print(f"Engine : {result.engine}")
    print(f"Input  : {result.input_dir}")
    print(f"Output : {result.output_dir}")
    if result.manual_macro_path is not None:
        print(f"Macro  : {result.manual_macro_path}")
        print("Run this macro from Fiji GUI: Plugins > Macros > Run...")

    if not result.file_pairs:
        print("No .oir files found.")
        return

    print(f"Files  : {len(result.file_pairs)} .oir file(s)")
    for pair in result.file_pairs:
        print(f"  input : {pair.input_oir}")
        print(f"  output: {pair.output_tif}")
        print(f"  macro : {pair.bioformats_import_command}")

    print(f"Processed {len(result.processed)} file(s).")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    try:
        result = run_oir_zmax_batch(
            input_dir,
            output_dir,
            engine=args.engine,
            fiji_executable=args.fiji,
            fiji_headless=args.headless,
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_result_summary(result)

    if result.failed:
        print(f"Failed {len(result.failed)} file(s):", file=sys.stderr)
        for item in result.failed:
            print(f"  input : {item.get('input_oir', item['file'])}", file=sys.stderr)
            if item.get("output_tif"):
                print(f"  output: {item['output_tif']}", file=sys.stderr)
            if item.get("import_command"):
                print(f"  macro : {item['import_command']}", file=sys.stderr)
            print(f"  error : {item['error']}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
