"""Batch processing helpers."""

from pathlib import Path

from bioimage_pipeline.pipeline import ImagePipeline


def process_folder(input_dir: str | Path, output_dir: str | Path, pipeline: ImagePipeline) -> None:
    """Process a folder of images with the provided pipeline."""
    raise NotImplementedError("Batch folder processing is not implemented yet.")
