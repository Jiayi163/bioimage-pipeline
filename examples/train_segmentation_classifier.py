"""Train a pixel classifier from labeled image/mask pairs (scikit-image pattern)."""

from __future__ import annotations

import argparse
from pathlib import Path

from bioimage_pipeline.trainable_segmenter import (
    TrainableSegmenterConfig,
    train_segmenter,
)
from bioimage_pipeline.training_data import discover_training_pairs, validate_training_pairs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a scikit-image trainable segmenter (multiscale features + "
            "RandomForest) from labeled TIFF pairs."
        ),
    )
    parser.add_argument(
        "--training-data",
        type=Path,
        required=True,
        help="Folder with images/ and labels/ subdirectories.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        required=True,
        help="Destination path for the saved .joblib model bundle.",
    )
    parser.add_argument("--sigma-min", type=float, default=1.0)
    parser.add_argument("--sigma-max", type=float, default=16.0)
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument(
        "--image-pattern",
        default="*.tif",
        help="Glob for training images (default: *.tif).",
    )
    args = parser.parse_args()

    pairs = discover_training_pairs(args.training_data, image_pattern=args.image_pattern)
    warnings = validate_training_pairs(pairs)
    for message in warnings:
        print(f"warning: {message}")

    config = TrainableSegmenterConfig(
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        image_pattern=args.image_pattern,
    )
    bundle = train_segmenter(args.training_data, args.output_model, config=config)
    print(f"Trained on {bundle.training_pair_count} image(s).")
    print(f"Saved model: {Path(args.output_model).resolve()}")
    print(f"Classes: {bundle.to_metadata()['classes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
