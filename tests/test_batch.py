"""Tests for batch processing."""

import numpy as np
import tifffile

from bioimage_pipeline.batch import run_pipeline_on_folder, run_pipeline_on_stack
from bioimage_pipeline.io import save_tiff
from bioimage_pipeline.pipeline import Pipeline
from bioimage_pipeline.preprocess import gaussian_blur
from bioimage_pipeline.segment import label_objects, remove_small_objects_from_mask
from bioimage_pipeline.stack import load_stack_from_folder, load_stack_from_tiff
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


# ---------------------------------------------------------------------------
# run_pipeline_on_stack (S.3 + S.4 + S.5)
# ---------------------------------------------------------------------------


def _make_image(bright: bool = True) -> np.ndarray:
    img = np.zeros((30, 40), dtype=np.uint8)
    if bright:
        img[:, 20:] = 200
    return img


def test_run_pipeline_on_stack_from_folder_processes_all_frames(tmp_path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        save_tiff(src / f"img_{i}.tif", _make_image())

    stack = load_stack_from_folder(src)
    result = run_pipeline_on_stack(_build_basic_pipeline(), stack, tmp_path / "out")

    assert len(result["processed"]) == 3
    assert result["failed"] == []


def test_run_pipeline_on_stack_per_frame_files_created(tmp_path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    save_tiff(src / "img_00.tif", _make_image())
    save_tiff(src / "img_01.tif", _make_image())

    out = tmp_path / "out"
    stack = load_stack_from_folder(src)
    run_pipeline_on_stack(_build_basic_pipeline(), stack, out)

    assert (out / "img_00_f000_mask.tif").exists()
    assert (out / "img_00_f000_labels.tif").exists()
    assert (out / "img_01_f001_mask.tif").exists()


def test_run_pipeline_on_stack_combined_csv_has_frame_index_column(tmp_path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(2):
        save_tiff(src / f"img_{i}.tif", _make_image())

    stack = load_stack_from_folder(src)

    def measure_step(data):
        import pandas as pd
        data["measurements"] = pd.DataFrame({"area": [10]})
        return data

    pipeline = Pipeline([measure_step])
    result = run_pipeline_on_stack(pipeline, stack, tmp_path / "out")

    assert result["measurements"] is not None
    assert "stack_id" in result["measurements"].columns
    assert "frame_index" in result["measurements"].columns
    assert (tmp_path / "out" / "all_measurements.csv").exists()


def test_run_pipeline_on_stack_from_multipage_tiff(tmp_path) -> None:
    planes = [_make_image() for _ in range(4)]
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, np.stack(planes), photometric="minisblack")

    stack = load_stack_from_tiff(path)
    result = run_pipeline_on_stack(_build_basic_pipeline(), stack, tmp_path / "out")

    assert len(result["processed"]) == 4


def test_run_pipeline_on_stack_isolates_frame_failures(tmp_path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    save_tiff(src / "good.tif", _make_image())
    save_tiff(src / "good2.tif", _make_image())

    stack = load_stack_from_folder(src)

    call_count = {"n": 0}

    def failing_step(data):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated failure")
        data["mask"] = data["image"] > 100
        return data

    pipeline = Pipeline([failing_step])
    result = run_pipeline_on_stack(pipeline, stack, tmp_path / "out")

    assert len(result["failed"]) == 1
    assert len(result["processed"]) == 1


def test_run_pipeline_on_stack_export_processed_flag(tmp_path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    save_tiff(src / "img_00.tif", _make_image())

    stack = load_stack_from_folder(src)
    out = tmp_path / "out"
    run_pipeline_on_stack(
        _build_basic_pipeline(), stack, out, export_processed=True
    )
    assert (out / "img_00_f000_processed.tif").exists()


def test_run_pipeline_on_stack_no_processed_tiff_by_default(tmp_path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    save_tiff(src / "img_00.tif", _make_image())

    stack = load_stack_from_folder(src)
    out = tmp_path / "out"
    run_pipeline_on_stack(_build_basic_pipeline(), stack, out)
    assert not (out / "img_00_f000_processed.tif").exists()
