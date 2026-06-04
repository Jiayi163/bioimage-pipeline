"""Tests for batch processing."""

import numpy as np

from bioimage_pipeline.batch import run_pipeline_on_folder
from bioimage_pipeline.io import save_tiff
from bioimage_pipeline.pipeline import Pipeline
from bioimage_pipeline.preprocess import gaussian_blur
from bioimage_pipeline.segment import label_objects, remove_small_objects_from_mask
from bioimage_pipeline.threshold import otsu_threshold


def _build_basic_pipeline() -> Pipeline:
    def blur_step(data: dict) -> dict:
        data["processed"] = gaussian_blur(data["image"], sigma=1)
        return data

    def threshold_step(data: dict) -> dict:
        data["mask"] = otsu_threshold(data["processed"])
        return data

    def clean_step(data: dict) -> dict:
        data["mask"] = remove_small_objects_from_mask(data["mask"], min_size=5)
        return data

    def label_step(data: dict) -> dict:
        data["labels"] = label_objects(data["mask"])
        return data

    return Pipeline([blur_step, threshold_step, clean_step, label_step])


def test_run_pipeline_on_folder_processes_multiple_tiffs(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    image_one = np.zeros((30, 40), dtype=np.uint8)
    image_one[:, 20:] = 200
    image_two = np.zeros((30, 40), dtype=np.uint8)
    image_two[5:10, 5:10] = 180

    save_tiff(input_dir / "a.tif", image_one)
    save_tiff(input_dir / "b.tif", image_two)

    result = run_pipeline_on_folder(_build_basic_pipeline(), input_dir, output_dir)

    assert len(result["processed"]) == 2
    assert result["failed"] == []
    assert (output_dir / "a_mask.tif").exists()
    assert (output_dir / "b_labels.tif").exists()


def test_run_pipeline_on_folder_reports_failed_images(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    image = np.zeros((20, 40), dtype=np.uint8)
    image[:, 20:] = 200
    save_tiff(input_dir / "good.tif", image)

    broken_path = input_dir / "bad.tif"
    broken_path.write_text("not a tiff", encoding="utf-8")

    result = run_pipeline_on_folder(_build_basic_pipeline(), input_dir, output_dir)

    assert "good.tif" in result["processed"]
    assert len(result["failed"]) == 1
    assert result["failed"][0]["filename"] == "bad.tif"
