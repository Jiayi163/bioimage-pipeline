"""Tests for stack batch JSON recipes (Phase S.6)."""

import json
from pathlib import Path

import pytest

from bioimage_pipeline.stack_recipe import (
    StackBatchRecipe,
    load_stack_batch_recipe,
    merge_recipe_with_cli,
    save_stack_batch_recipe,
)


def test_stack_batch_recipe_round_trip(tmp_path: Path) -> None:
    recipe = StackBatchRecipe(
        blur_sigma=1.5,
        min_object_size=30,
        labeling_method="watershed",
        export_processed=True,
        generate_qc=True,
        input="input_dir",
        output="output_dir",
    )
    path = save_stack_batch_recipe(recipe, tmp_path / "recipe.json")
    loaded = load_stack_batch_recipe(path)

    assert loaded.blur_sigma == 1.5
    assert loaded.min_object_size == 30
    assert loaded.labeling_method == "watershed"
    assert loaded.export_processed is True
    assert loaded.generate_qc is True
    assert loaded.input == "input_dir"
    assert loaded.output == "output_dir"


def test_load_recipe_accepts_labeling_alias(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    path.write_text(
        json.dumps({"labeling": "watershed", "output": "out"}),
        encoding="utf-8",
    )
    recipe = load_stack_batch_recipe(path)
    assert recipe.labeling_method == "watershed"


def test_load_recipe_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_stack_batch_recipe(path)


def test_load_recipe_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_stack_batch_recipe(tmp_path / "missing.json")


def test_merge_recipe_cli_overrides_recipe_values() -> None:
    base = StackBatchRecipe(blur_sigma=1.0, labeling_method="connected")
    merged = merge_recipe_with_cli(
        base,
        blur_sigma=2.5,
        labeling="watershed",
        export_processed=True,
    )
    assert merged.blur_sigma == 2.5
    assert merged.labeling_method == "watershed"
    assert merged.export_processed is True


def test_merge_recipe_keeps_recipe_when_cli_not_set() -> None:
    base = StackBatchRecipe(export_processed=True, generate_qc=True)
    merged = merge_recipe_with_cli(base)
    assert merged.export_processed is True
    assert merged.generate_qc is True


def test_merge_recipe_invalid_labeling_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported labeling_method"):
        merge_recipe_with_cli(None, labeling="invalid")
