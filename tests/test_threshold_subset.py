"""Tests for threshold subset selection and staging."""

from __future__ import annotations

from pathlib import Path

import pytest

from bioimage_pipeline.threshold_subset import (
    ThresholdSubsetSelection,
    list_candidate_input_images,
    load_subset_manifest,
    materialize_input_subset,
    sample_image_paths,
    select_input_subset,
)


def _write_images(folder: Path, count: int) -> list[Path]:
    paths: list[Path] = []
    for index in range(count):
        path = folder / f"img_{index:03d}.tif"
        path.write_bytes(b"fake")
        paths.append(path)
    return paths


def test_sample_image_paths_even_spreads_across_sorted_list() -> None:
    images = [Path(f"img_{index:03d}.tif") for index in range(10)]
    sampled = sample_image_paths(images, count=5, method="even")
    assert [path.name for path in sampled] == [
        "img_000.tif",
        "img_002.tif",
        "img_004.tif",
        "img_007.tif",
        "img_009.tif",
    ]


def test_select_input_subset_manual_names(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_images(input_dir, 4)

    selected = select_input_subset(
        input_dir,
        ThresholdSubsetSelection(mode="manual", selected_paths=("img_001.tif", "img_003.tif")),
    )
    assert [path.name for path in selected] == ["img_001.tif", "img_003.tif"]


def test_materialize_input_subset_writes_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    staged_dir = tmp_path / "subset"
    input_dir.mkdir()
    paths = _write_images(input_dir, 3)

    manifest = materialize_input_subset(
        input_dir,
        staged_dir,
        paths[:2],
        mode="manual",
        sample_method="even",
    )

    assert manifest.image_names == ["img_000.tif", "img_001.tif"]
    assert (staged_dir / "img_000.tif").is_file()
    assert (staged_dir / "img_001.tif").is_file()
    loaded = load_subset_manifest(staged_dir / "subset_manifest.json")
    assert loaded.image_names == manifest.image_names


def test_select_input_subset_auto_uses_sample_count(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_images(input_dir, 8)

    selected = select_input_subset(
        input_dir,
        ThresholdSubsetSelection(sample_count=3, sample_method="first"),
    )
    assert len(selected) == 3
    assert selected[0].name == "img_000.tif"


def test_select_input_subset_raises_for_missing_manual_image(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_images(input_dir, 2)

    with pytest.raises(ValueError, match="not found"):
        select_input_subset(
            input_dir,
            manual_image_names=["missing.tif"],
        )


def test_list_candidate_input_images_finds_tiffs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_images(input_dir, 2)
    assert len(list_candidate_input_images(input_dir)) == 2
