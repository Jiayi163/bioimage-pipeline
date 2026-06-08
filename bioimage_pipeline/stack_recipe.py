"""JSON batch recipe load/save for stack processing (Phase S.6)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

LabelingMethod = Literal["connected", "watershed"]
_SUPPORTED_LABELING = frozenset({"connected", "watershed"})


@dataclass
class StackBatchRecipe:
    """Serializable configuration for a stack/batch pipeline run."""

    blur_sigma: float = 1.0
    min_object_size: int = 20
    labeling_method: LabelingMethod = "connected"
    export_processed: bool = False
    generate_qc: bool = False
    input: str | None = None
    output: str | None = None
    demo: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StackBatchRecipe:
        """Build a recipe from a parsed JSON object."""
        labeling = data.get("labeling_method", data.get("labeling", "connected"))
        if labeling not in _SUPPORTED_LABELING:
            supported = ", ".join(sorted(_SUPPORTED_LABELING))
            raise ValueError(
                f"Unsupported labeling_method: {labeling!r}. "
                f"Choose one of: {supported}."
            )

        blur_sigma = float(data.get("blur_sigma", 1.0))
        if blur_sigma < 0:
            raise ValueError("blur_sigma must be >= 0")

        min_object_size = int(data.get("min_object_size", 20))
        if min_object_size < 1:
            raise ValueError("min_object_size must be >= 1")

        return cls(
            blur_sigma=blur_sigma,
            min_object_size=min_object_size,
            labeling_method=labeling,  # type: ignore[arg-type]
            export_processed=bool(data.get("export_processed", False)),
            generate_qc=bool(data.get("generate_qc", False)),
            input=data.get("input"),
            output=data.get("output"),
            demo=bool(data.get("demo", False)),
        )


def load_stack_batch_recipe(path: str | Path) -> StackBatchRecipe:
    """Load a batch recipe from a JSON file."""
    recipe_path = Path(path)
    if not recipe_path.is_file():
        raise FileNotFoundError(f"Recipe file not found: {recipe_path}")

    try:
        data = json.loads(recipe_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in recipe file: {recipe_path}") from exc

    if not isinstance(data, dict):
        raise ValueError("Recipe JSON must be an object at the top level.")

    return StackBatchRecipe.from_dict(data)


def save_stack_batch_recipe(recipe: StackBatchRecipe, path: str | Path) -> Path:
    """Write a batch recipe to a JSON file."""
    recipe_path = Path(path)
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(
        json.dumps(recipe.to_dict(), indent=2),
        encoding="utf-8",
    )
    return recipe_path.resolve()


def merge_recipe_with_cli(
    recipe: StackBatchRecipe | None,
    *,
    input_path: str | None = None,
    output_path: str | None = None,
    blur_sigma: float | None = None,
    min_object_size: int | None = None,
    labeling: str | None = None,
    export_processed: bool | None = None,
    generate_qc: bool | None = None,
    demo: bool | None = None,
) -> StackBatchRecipe:
    """Merge a loaded recipe with explicit CLI overrides (CLI wins)."""
    base = recipe or StackBatchRecipe()

    resolved_labeling = labeling if labeling is not None else base.labeling_method
    if resolved_labeling not in _SUPPORTED_LABELING:
        supported = ", ".join(sorted(_SUPPORTED_LABELING))
        raise ValueError(
            f"Unsupported labeling_method: {resolved_labeling!r}. "
            f"Choose one of: {supported}."
        )

    return StackBatchRecipe(
        blur_sigma=blur_sigma if blur_sigma is not None else base.blur_sigma,
        min_object_size=(
            min_object_size if min_object_size is not None else base.min_object_size
        ),
        labeling_method=resolved_labeling,  # type: ignore[arg-type]
        export_processed=(
            export_processed if export_processed is not None else base.export_processed
        ),
        generate_qc=generate_qc if generate_qc is not None else base.generate_qc,
        input=input_path if input_path is not None else base.input,
        output=output_path if output_path is not None else base.output,
        demo=demo if demo is not None else base.demo,
    )
