"""Evaluate a trained RF classifier on a held-out split from training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bioimage_pipeline.trainable_rf import batch_predict, load_model
from bioimage_pipeline.training_data import (
    discover_training_pairs,
    load_split_manifest,
    pairs_for_split,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate classifier predictions on a held-out image split.",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    pairs = discover_training_pairs(args.training_data)
    split_manifest = load_split_manifest(args.training_data)
    if not split_manifest:
        raise SystemExit(
            f"Missing split manifest: {args.training_data / 'split_manifest.json'}"
        )
    held_out = pairs_for_split(pairs, split_manifest, args.split)
    input_dir = args.output_dir / "held_out_images"
    input_dir.mkdir(parents=True, exist_ok=True)
    for pair in held_out:
        destination = input_dir / pair.image_path.name
        if not destination.exists():
            destination.write_bytes(pair.image_path.read_bytes())

    exports = batch_predict(
        args.model,
        input_dir,
        args.output_dir,
        threshold=args.threshold,
        negative_control_names={pair.image_path.name for pair in held_out if "noev" in pair.image_path.name.lower()},
    )
    report = {
        "split": args.split,
        "image_count": len(exports),
        "scope": load_model(args.model).training_metadata.get("scope"),
        "exports": [export.image_name for export in exports],
    }
    report_path = args.output_dir / f"classifier_eval_{args.split}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote evaluation report: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
