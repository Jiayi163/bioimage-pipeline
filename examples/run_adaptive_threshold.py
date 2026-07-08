
from __future__ import annotations

import argparse
from pathlib import Path

from bioimage_pipeline.adaptive_import import run_self_adaptive_threshold_on_folder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-adaptive fluorescence nuclei thresholding at import.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing input TIFF images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Staging folder for masks/, labels/, and optional corrected/.",
    )
    parser.add_argument(
        "--pattern",
        default="*.tif",
        help="Glob pattern for input images (default: *.tif).",
    )
    parser.add_argument(
        "--min-object-size",
        type=int,
        default=20,
        help="Minimum object size in pixels after cleanup (default: 20).",
    )
    parser.add_argument(
        "--export-corrected",
        action="store_true",
        help="Also write background-corrected images to staging/corrected/.",
    )
    args = parser.parse_args()

    summary = run_self_adaptive_threshold_on_folder(
        args.input_dir,
        args.output_dir,
        pattern=args.pattern,
        min_object_size=args.min_object_size,
        export_corrected=args.export_corrected,
        logs_dir=args.output_dir.parent / "logs",
    )

    print(f"Processed {len(summary['processed'])} image(s).")
    if summary["failed"]:
        print(f"Failed: {summary['failed']}")
    print(f"Summary: {summary['summary_path']}")
    for filename in summary["processed"]:
        decision = summary["decisions"][filename]
        print(
            f"  {filename}: method={decision['method']} "
            f"confidence={decision['confidence']} "
            f"objects={decision['object_count']}"
        )


if __name__ == "__main__":
    main()
