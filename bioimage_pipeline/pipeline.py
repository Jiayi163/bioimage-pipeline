"""Pipeline orchestration primitives."""

from collections.abc import Callable, Iterable
from typing import Any


class ImagePipeline:
    """A sequence of image processing steps."""

    def __init__(self, steps: Iterable[Callable[[Any], Any]] | None = None) -> None:
        self.steps = list(steps or [])

    def add_step(self, step: Callable[[Any], Any]) -> None:
        """Add an image processing step to the pipeline."""
        self.steps.append(step)

    def run(self, image: Any) -> Any:
        """Run all configured steps on a single image."""
        raise NotImplementedError("Pipeline execution is not implemented yet.")
