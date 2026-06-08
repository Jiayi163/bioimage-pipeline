"""Stack/batch workflow orchestration (Phases S.3-S.7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bioimage_pipeline.analysis import build_default_pipeline
from bioimage_pipeline.batch import run_pipeline_on_stack
from bioimage_pipeline.qc import generate_qc_for_stack
from bioimage_pipeline.stack import ImageStack, load_stack
from bioimage_pipeline.stack_recipe import StackBatchRecipe


@dataclass
class StackBatchResult:
    """Summary of a completed stack/batch workflow run."""

    stack: ImageStack
    output_dir: Path
    processed: list[str]
    failed: list[dict[str, Any]]
    measurements: Any = None
    qc_artifacts: dict[int, dict[str, Path]] = field(default_factory=dict)


def run_stack_batch_workflow(
    input_source: str | Path,
    output_dir: str | Path,
    *,
    blur_sigma: float = 1.0,
    min_object_size: int = 20,
    labeling_method: str = "connected",
    export_processed: bool = False,
    generate_qc: bool = False,
) -> StackBatchResult:
    """Load a stack, run the default pipeline on every frame, optionally generate QC.

    Args:
        input_source: Folder of TIFFs or path to a multi-page TIFF stack.
        output_dir: Folder where per-frame outputs and ``all_measurements.csv`` are written.
        blur_sigma: Gaussian blur sigma for preprocessing.
        min_object_size: Minimum object size in pixels after thresholding.
        labeling_method: ``"connected"`` or ``"watershed"``.
        export_processed: Write blurred/processed intensity TIFF per frame.
        generate_qc: Write mask/label overlay PNGs per frame into ``output_dir``.

    Returns:
        :class:`StackBatchResult` with paths, measurements, and optional QC artifacts.
    """
    stack = load_stack(input_source)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    pipeline = build_default_pipeline(
        blur_sigma=blur_sigma,
        min_object_size=min_object_size,
        labeling_method=labeling_method,  # type: ignore[arg-type]
    )

    batch_result = run_pipeline_on_stack(
        pipeline,
        stack,
        out_path,
        export_processed=export_processed,
    )

    qc_artifacts: dict[int, dict[str, Path]] = {}
    if generate_qc:
        qc_artifacts = generate_qc_for_stack(stack, out_path, image_format="png")

    return StackBatchResult(
        stack=stack,
        output_dir=out_path.resolve(),
        processed=batch_result["processed"],
        failed=batch_result["failed"],
        measurements=batch_result.get("measurements"),
        qc_artifacts=qc_artifacts,
    )


def run_stack_batch_from_recipe(recipe: StackBatchRecipe) -> StackBatchResult:
    """Run :func:`run_stack_batch_workflow` using a :class:`StackBatchRecipe`."""
    if recipe.input is None:
        raise ValueError("Recipe must include an 'input' path (or use demo mode).")
    if recipe.output is None:
        raise ValueError("Recipe must include an 'output' path.")

    return run_stack_batch_workflow(
        recipe.input,
        recipe.output,
        blur_sigma=recipe.blur_sigma,
        min_object_size=recipe.min_object_size,
        labeling_method=recipe.labeling_method,
        export_processed=recipe.export_processed,
        generate_qc=recipe.generate_qc,
    )
