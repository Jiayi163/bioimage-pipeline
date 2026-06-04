"""Tests for pipeline orchestration."""

import numpy as np
import pandas as pd
import pytest

from bioimage_pipeline.pipeline import Pipeline


def test_pipeline_runs_steps_in_order() -> None:
    def add_processed(data: dict) -> dict:
        data["processed"] = data["image"] + 1
        return data

    def add_mask(data: dict) -> dict:
        data["mask"] = data["processed"] > 0
        return data

    pipeline = Pipeline([add_processed, add_mask])
    result = pipeline.run({"image": np.array([0, 1], dtype=np.uint8)})

    assert "processed" in result
    assert "mask" in result
    assert result["processed"][1] == 2


def test_pipeline_step_must_return_dict() -> None:
    def bad_step(data: dict) -> dict:
        return "not a dict"  # type: ignore[return-value]

    pipeline = Pipeline([bad_step])

    with pytest.raises(TypeError, match="dictionary"):
        pipeline.run({"image": np.zeros((2, 2))})


def test_pipeline_wraps_step_errors() -> None:
    def failing_step(data: dict) -> dict:
        raise ValueError("boom")

    pipeline = Pipeline([failing_step])

    with pytest.raises(RuntimeError, match="step 0"):
        pipeline.run({"image": np.zeros((2, 2))})


def test_pipeline_data_must_be_dict() -> None:
    pipeline = Pipeline([])

    with pytest.raises(TypeError, match="dictionary"):
        pipeline.run([])  # type: ignore[arg-type]
