"""Placeholder example for constructing a bioimage pipeline."""

from bioimage_pipeline.pipeline import ImagePipeline


def main() -> None:
    """Create an empty pipeline placeholder."""
    pipeline = ImagePipeline()
    print(f"Created pipeline with {len(pipeline.steps)} steps.")


if __name__ == "__main__":
    main()
