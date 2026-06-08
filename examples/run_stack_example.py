"""End-to-end stack/batch example — synthetic Z-stack (Phase S.7).

Demonstrates the complete Fiji-style stack workflow:

1. Generate a 5-frame synthetic Z-stack (folder of TIFFs).
2. Load as an ImageStack (auto-detected from folder).
3. Run the default preprocess -> segment -> measure pipeline on every frame.
4. Export per-frame masks, labels, and measurements.
5. Generate QC overlay PNGs for each frame.
6. Print a summary table.

No real data required.  Run from the project root::

    python examples/run_stack_example.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bioimage_pipeline.io import save_tiff
from bioimage_pipeline.stack_batch import run_stack_batch_workflow


def _make_frame(seed: int, shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Two bright circular nuclei on a dark background, slightly shifted per frame."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 20, size=shape, dtype=np.uint16)
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    offset = seed * 4
    for cy, cx, radius, intensity in (
        (40 + offset, 40, 13, 225),
        (85, 88 + offset, 11, 205),
    ):
        disk = (rows - cy) ** 2 + (cols - cx) ** 2 <= radius**2
        image[disk] = intensity
    return image


def build_synthetic_stack(output_dir: Path, n_frames: int = 5) -> Path:
    """Create a folder of synthetic TIFF frames and return its path."""
    src = output_dir / "input_stack"
    src.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        save_tiff(src / f"slice_{i:02d}.tif", _make_frame(seed=i))
    return src


def main() -> None:
    output_dir = Path("output") / "stack_example"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Stack / Batch Example ===")
    print()

    src = build_synthetic_stack(output_dir, n_frames=5)
    print(f"[1] Synthetic stack written to: {src}")

    results_dir = output_dir / "results"
    result = run_stack_batch_workflow(
        src,
        results_dir,
        blur_sigma=1.0,
        min_object_size=15,
        export_processed=True,
        generate_qc=True,
    )
    print(f"[2] Loaded and processed: {result.stack.frame_count} frames, shape {result.stack.shape}")
    print(f"[3] Processed: {len(result.processed)} frames, failed: {len(result.failed)}")

    overlay_count = sum(len(v) for v in result.qc_artifacts.values())
    print(f"[4] QC overlays: {overlay_count} PNG(s) across {len(result.qc_artifacts)} frame(s)")

    if result.measurements is not None:
        m = result.measurements
        print()
        print("Objects per frame:")
        for frame_idx, group in m.groupby("frame_index"):
            print(
                f"  Frame {frame_idx}: {len(group)} object(s), "
                f"mean area = {group['area'].mean():.1f} px"
            )
        print()
        print(f"Total objects across all frames: {len(m)}")

    print()
    print(f"Results: {results_dir.resolve()}")
    print("Per-frame files: {stem}_f{idx:03d}_mask.tif / _labels.tif / "
          "_processed.tif / _measurements.csv")
    print("Combined CSV   : all_measurements.csv")


if __name__ == "__main__":
    main()
