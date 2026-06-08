"""Fiji-macro-style stack / batch processing from the command line (Phase S.6).

Mirrors the Fiji "Process Folder" / macro workflow:
load input -> run fixed pipeline steps on every frame -> export results.

Examples
--------
Process a folder of TIFFs::

    python examples/run_stack_batch.py \\
        --input path/to/images/ \\
        --output path/to/results/

Process a single multi-page TIFF stack::

    python examples/run_stack_batch.py \\
        --input path/to/stack.tif \\
        --output path/to/results/ \\
        --labeling watershed \\
        --export-processed \\
        --generate-qc

Use a JSON recipe (CLI flags override recipe values)::

    python examples/run_stack_batch.py \\
        --recipe examples/stack_batch_recipe.json \\
        --output path/to/results/

Use a synthetic built-in demo (no real data required)::

    python examples/run_stack_batch.py --demo --output output/demo_test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from bioimage_pipeline.io import save_tiff
from bioimage_pipeline.stack_batch import run_stack_batch_workflow
from bioimage_pipeline.stack_recipe import (
    StackBatchRecipe,
    load_stack_batch_recipe,
    merge_recipe_with_cli,
)


def _make_demo_frame(seed: int, shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 20, size=shape, dtype=np.uint16)
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    for cy, cx, r, intensity in (
        (35 + seed * 5, 35, 14, 220),
        (80, 90 + seed * 4, 11, 200),
    ):
        mask = (rows - cy) ** 2 + (cols - cx) ** 2 <= r**2
        image[mask] = intensity
    return image


def create_demo_stack(output_dir: Path, n_frames: int = 5) -> Path:
    """Write a synthetic multi-frame folder for demo mode."""
    demo_dir = output_dir / "demo_input"
    demo_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        save_tiff(demo_dir / f"frame_{i:02d}.tif", _make_demo_frame(seed=i))
    return demo_dir


def resolve_stack_batch_config(
    *,
    recipe_path: str | Path | None,
    input_path: str | None,
    output_path: str | None,
    blur_sigma: float | None,
    min_object_size: int | None,
    labeling: str | None,
    export_processed: bool,
    generate_qc: bool,
    demo: bool,
) -> tuple[Path, Path, StackBatchRecipe]:
    """Resolve final input/output paths and merged recipe settings."""
    loaded: StackBatchRecipe | None = None
    if recipe_path is not None:
        loaded = load_stack_batch_recipe(recipe_path)

    merged = merge_recipe_with_cli(
        loaded,
        input_path=input_path,
        output_path=output_path,
        blur_sigma=blur_sigma,
        min_object_size=min_object_size,
        labeling=labeling,
        export_processed=export_processed if export_processed else None,
        generate_qc=generate_qc if generate_qc else None,
        demo=demo if demo else None,
    )

    if merged.output is None:
        raise ValueError("--output is required (or set 'output' in the recipe JSON).")

    output_dir = Path(merged.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if merged.demo:
        input_source = create_demo_stack(output_dir)
    elif merged.input:
        input_source = Path(merged.input)
        if not input_source.exists():
            raise FileNotFoundError(
                f"Input path does not exist: {input_source}\n"
                "  Pass a real folder of TIFFs or a .tif/.tiff stack file.\n"
                "  Use --demo to run on a built-in synthetic stack."
            )
    else:
        raise ValueError("Provide --input PATH, set 'input' in the recipe, or use --demo.")

    return input_source, output_dir, merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stack / batch pipeline — load a folder or multi-page TIFF, "
            "run preprocess -> segment -> measure on every frame, export results."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        metavar="PATH",
        help="Folder of TIFF images or path to a multi-page TIFF stack.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Output folder for results (required unless set in --recipe).",
    )
    parser.add_argument(
        "--recipe",
        metavar="PATH",
        help="JSON batch recipe file (CLI flags override recipe values).",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=None,
        metavar="SIGMA",
        help="Gaussian blur sigma (default: 1.0, or recipe value).",
    )
    parser.add_argument(
        "--min-object-size",
        type=int,
        default=None,
        metavar="PIXELS",
        help="Minimum object size in pixels (default: 20, or recipe value).",
    )
    parser.add_argument(
        "--labeling",
        choices=("connected", "watershed"),
        default=None,
        help="Object labeling method (default: connected, or recipe value).",
    )
    parser.add_argument(
        "--export-processed",
        action="store_true",
        help="Also export the blurred/processed intermediate image per frame.",
    )
    parser.add_argument(
        "--generate-qc",
        action="store_true",
        help="Write mask/label QC overlay PNGs per frame into the output folder.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on a built-in synthetic stack (no --input required).",
    )

    args = parser.parse_args(argv)

    try:
        input_source, output_dir, config = resolve_stack_batch_config(
            recipe_path=args.recipe,
            input_path=args.input,
            output_path=args.output,
            blur_sigma=args.blur_sigma,
            min_object_size=args.min_object_size,
            labeling=args.labeling,
            export_processed=args.export_processed,
            generate_qc=args.generate_qc,
            demo=args.demo,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
        return 2

    print(f"Loading stack from: {input_source}")
    print(f"Running pipeline ...")
    result = run_stack_batch_workflow(
        input_source,
        output_dir,
        blur_sigma=config.blur_sigma,
        min_object_size=config.min_object_size,
        labeling_method=config.labeling_method,
        export_processed=config.export_processed,
        generate_qc=config.generate_qc,
    )
    print(f"  {result.stack.frame_count} frame(s), shape per frame: {result.stack.shape}")

    print("\nDone.")
    print(f"  Processed : {len(result.processed)} frame(s)")
    if result.failed:
        print(f"  Failed    : {len(result.failed)} frame(s)")
        for failure in result.failed:
            print(
                f"    frame {failure['frame_index']} ({failure['filename']}): "
                f"{failure['error']}"
            )
    if result.measurements is not None:
        print(f"  Objects   : {len(result.measurements)} total across all frames")
    if result.qc_artifacts:
        overlay_count = sum(len(v) for v in result.qc_artifacts.values())
        print(f"  QC PNGs   : {overlay_count} overlay(s)")
    print(f"  Output    : {result.output_dir}")
    print()
    print("Per-frame files:  {stem}_f{index:03d}_mask.tif / _labels.tif / _measurements.csv")
    print("Combined table :  all_measurements.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
