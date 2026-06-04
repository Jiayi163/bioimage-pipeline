"""Pipeline orchestration primitives."""

from collections.abc import Callable, Iterable
from typing import Any


class Pipeline:
    """CellProfiler-style pipeline: ordered steps sharing a data dictionary."""

    def __init__(
        self,
        steps: Iterable[Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
    ) -> None:
        self.steps = list(steps or [])

    def add_step(self, step: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        """Add a processing step to the pipeline."""
        self.steps.append(step)

    def run(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run all steps in order, passing the shared data dictionary."""
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")

        current = dict(data)
        for index, step in enumerate(self.steps):
            try:
                result = step(current)
            except Exception as exc:
                raise RuntimeError(f"Pipeline step {index} failed") from exc

            if not isinstance(result, dict):
                raise TypeError(f"Pipeline step {index} must return a dictionary")

            current = result

        return current
