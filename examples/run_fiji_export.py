"""Run Fiji/ImageJ batch export on a CellProfiler output folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bioimage_pipeline.fiji_runner import run_fiji_batch_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Fiji/ImageJ batch macro to export final mask and label TIFFs "
            "from a CellProfiler output folder."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="CellProfiler raw output folder containing TIFF files.",
    )
    parser.add_argument(
        "--masks-dir",
        required=True,
        type=Path,
        help="Destination folder for final mask TIFFs.",
    )
    parser.add_argument(
        "--labels-dir",
        required=True,
        type=Path,
        help="Destination folder for final label TIFFs.",
    )
    parser.add_argument(
        "--macro",
        type=Path,
        default=None,
        help="Optional Fiji batch macro path. Defaults to examples/fiji_macros/export_folder.ijm.",
    )
    parser.add_argument(
        "--fiji",
        default=None,
        help="Fiji/ImageJ executable path. Defaults to FIJI_EXECUTABLE or auto-detection.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override platform default headless mode.",
    )
    parser.add_argument(
        "--pattern",
        default="*.tif",
        help="TIFF pattern passed to the batch macro.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Optional folder for fiji_stdout.log, fiji_stderr.log, and command logs.",
    )
    parser.add_argument(
        "--json-summary",
        action="store_true",
        help="Print the export result as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        result = run_fiji_batch_export(
            args.input_dir,
            args.masks_dir,
            args.labels_dir,
            macro_path=args.macro,
            fiji_executable=args.fiji,
            headless=args.headless,
            image_pattern=args.pattern,
            log_dir=args.logs_dir,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json_summary:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Fiji command: {' '.join(result.command)}")
        print(f"Return code : {result.returncode}")
        print(f"Masks       : {len(result.mask_exports)} file(s)")
        print(f"Labels      : {len(result.label_exports)} file(s)")
        if result.log_files:
            print(f"Logs        : {next(iter(result.log_files.values())).parent}")

    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
