"""Batch-predict EV probability maps and masks with a trained segmenter."""

from __future__ import annotations

import argparse
from pathlib import Path

from bioimage_pipeline.trainable_segmenter import (
    TrainableSegmenterConfig,
    predict_folder,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run batch trainable-segmentation prediction on projected TIFF images.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing input TIFF images.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to a trained .joblib model bundle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output folder (probability_maps/, masks/, labels/).",
    )
    parser.add_argument("--pattern", default="*.tif", help="Input glob (default: *.tif).")
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=0.5,
        help="EV probability threshold for binary mask (default: 0.5).",
    )
    args = parser.parse_args()

    exports = predict_folder(
        args.input_dir,
        args.model,
        args.output_dir,
        pattern=args.pattern,
        config_override=TrainableSegmenterConfig(
            ev_probability_threshold=args.probability_threshold,
        ),
    )
    print(f"Predicted {len(exports)} image(s) -> {Path(args.output_dir).resolve()}")
    for item in exports:
        print(f"  {item.image_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
