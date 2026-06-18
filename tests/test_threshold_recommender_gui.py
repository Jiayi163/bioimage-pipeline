"""Tests for threshold recommender GUI helpers."""

from __future__ import annotations

from bioimage_pipeline.gui.threshold_recommender_helpers import (
    build_recommender_config,
    selected_manual_subset_names,
)


def test_selected_manual_subset_names_returns_empty_for_auto_mode() -> None:
    assert selected_manual_subset_names(["a.tif", "b.tif"], []) == []


def test_selected_manual_subset_names_returns_selected_basenames() -> None:
    names = ["a.tif", "b.tif", "c.tif"]
    assert selected_manual_subset_names(names, [0, 2]) == ["a.tif", "c.tif"]


def test_build_recommender_config_uses_manual_mode_when_names_provided() -> None:
    config = build_recommender_config(
        imported_cppipe_path="pipeline.cppipe",
        input_dir="input",
        output_dir="output",
        cellprofiler_executable="cellprofiler",
        subset_count=5,
        subset_method="even",
        manual_subset_image_names=["a.tif"],
    )
    assert config.subset_selection.mode == "manual"
    assert config.manual_subset_image_names == ["a.tif"]


def test_build_recommender_config_defaults_to_auto_sampling() -> None:
    config = build_recommender_config(
        imported_cppipe_path="pipeline.cppipe",
        input_dir="input",
        output_dir="output",
        cellprofiler_executable="cellprofiler",
        subset_count=3,
        subset_method="first",
    )
    assert config.subset_selection.mode == "auto"
    assert config.subset_selection.sample_count == 3


def test_build_recommender_config_defaults_to_fast_optimistic() -> None:
    config = build_recommender_config(
        imported_cppipe_path="pipeline.cppipe",
        input_dir="input",
        output_dir="output",
        cellprofiler_executable="cellprofiler",
        subset_count=3,
        subset_method="first",
    )

    assert config.fast_optimistic is True


def test_build_recommender_config_can_disable_fast_optimistic() -> None:
    config = build_recommender_config(
        imported_cppipe_path="pipeline.cppipe",
        input_dir="input",
        output_dir="output",
        cellprofiler_executable="cellprofiler",
        subset_count=3,
        subset_method="first",
        fast_optimistic=False,
    )

    assert config.fast_optimistic is False
