"""Tests for unified analysis mode."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bioimage_pipeline.cellprofiler_runner import CellProfilerMeasurementsResult
from bioimage_pipeline.analysis import (
    AnalysisConfig,
    build_default_pipeline,
    run_analysis,
    run_analysis_from_config,
)
from bioimage_pipeline.io import save_tiff
from bioimage_pipeline.pipeline import Pipeline


def _touching_circles_image(shape: tuple[int, int] = (80, 80)) -> np.ndarray:
    image = np.full(shape, 40, dtype=np.uint16)
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    for center_y, center_x, radius, intensity in (
        (40, 28, 12, 500),
        (40, 52, 12, 500),
    ):
        circle = (rows - center_y) ** 2 + (cols - center_x) ** 2 <= radius**2
        image[circle] = intensity
    return image


def test_build_default_pipeline_runs_end_to_end() -> None:
    image = np.zeros((40, 40), dtype=np.uint8)
    image[:, 20:] = 200

    result = build_default_pipeline().run({"image": image, "filename": "test.tif"})

    assert "mask" in result
    assert "labels" in result
    assert "measurements" in result
    assert not result["measurements"].empty


def test_build_default_pipeline_watershed_splits_touching_objects() -> None:
    image = _touching_circles_image()
    data = {"image": image, "filename": "touching.tif"}

    connected = build_default_pipeline(labeling_method="connected").run(data)
    watershed = build_default_pipeline(labeling_method="watershed").run(
        dict(data)
    )

    assert connected["labels"].max() == 1
    assert watershed["labels"].max() == 2
    assert len(connected["measurements"]) == 1
    assert len(watershed["measurements"]) == 2


def test_run_analysis_accepts_watershed_labeling_method(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    save_tiff(input_dir / "touching.tif", _touching_circles_image())

    result = run_analysis(
        input_dir,
        output_dir,
        analysis_engine="python",
        labeling_method="watershed",
    )

    assert result["processed"] == ["touching.tif"]
    measurements = pd.read_csv(output_dir / "touching_measurements.csv")
    assert len(measurements) == 2


def test_run_analysis_invalid_labeling_method_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported labeling_method"):
        build_default_pipeline(labeling_method="dbscan")  # type: ignore[arg-type]


def test_run_analysis_python_mode_processes_images(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    image = np.zeros((30, 40), dtype=np.uint8)
    image[:, 20:] = 200
    save_tiff(input_dir / "sample.tif", image)

    result = run_analysis(input_dir, output_dir, analysis_engine="python")

    assert result["analysis_engine"] == "python"
    assert result["output_dir"] == output_dir.resolve()
    assert result["processed"] == ["sample.tif"]
    assert result["failed"] == []
    assert result["tables"] is None
    assert (output_dir / "sample_mask.tif").exists()
    assert (output_dir / "all_measurements.csv").exists()


def test_run_analysis_python_mode_accepts_custom_pipeline(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    image = np.zeros((20, 20), dtype=np.uint8)
    image[5:15, 5:15] = 220
    save_tiff(input_dir / "custom.tif", image)

    def only_mask_step(data: dict) -> dict:
        data["mask"] = data["image"] > 0
        return data

    custom_pipeline = Pipeline([only_mask_step])
    result = run_analysis(
        input_dir,
        output_dir,
        analysis_engine="python",
        pipeline=custom_pipeline,
    )

    assert result["processed"] == ["custom.tif"]
    assert (output_dir / "custom_mask.tif").exists()
    assert not (output_dir / "custom_labels.tif").exists()


def test_run_analysis_from_config_python_mode(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    image = np.zeros((20, 30), dtype=np.uint8)
    image[:, 15:] = 180
    save_tiff(input_dir / "cfg.tif", image)

    config = AnalysisConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        analysis_engine="python",
    )
    result = run_analysis_from_config(config)

    assert result["analysis_engine"] == "python"
    assert result["processed"] == ["cfg.tif"]


def test_run_analysis_invalid_engine_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported analysis_engine"):
        run_analysis("input", "output", analysis_engine="java")  # type: ignore[arg-type]


def test_run_analysis_cellprofiler_requires_cppipe(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    with pytest.raises(ValueError, match="cppipe_path is required"):
        run_analysis(input_dir, output_dir, analysis_engine="cellprofiler")


@patch("bioimage_pipeline.analysis.run_cellprofiler_pipeline")
@patch("bioimage_pipeline.analysis.load_cellprofiler_measurements")
@patch("bioimage_pipeline.analysis.merge_cellprofiler_tables")
def test_run_analysis_cellprofiler_mode(
    mock_merge: MagicMock,
    mock_load: MagicMock,
    mock_run: MagicMock,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    cppipe = tmp_path / "pipeline.cppipe"
    input_dir.mkdir()
    cppipe.write_text("pipeline", encoding="utf-8")

    image_tables = {
        "MyExpt_Image": pd.DataFrame({"Image_Number": [1], "FileName": ["a.tif"]}),
    }
    object_tables = {
        "MyExpt_Objects": pd.DataFrame(
            {"Image_Number": [1], "ObjectNumber": [1], "AreaShape_Area": [100]}
        ),
    }
    mock_run.return_value = output_dir.resolve()
    mock_load.return_value = CellProfilerMeasurementsResult(
        tables={**image_tables, **object_tables},
        metadata={},
        warnings=[],
    )
    mock_merge.return_value = (
        pd.DataFrame(
            {
                "Image_Number": [1],
                "ObjectNumber": [1],
                "AreaShape_Area": [100],
                "FileName": ["a.tif"],
            }
        ),
        [],
    )

    result = run_analysis(
        input_dir,
        output_dir,
        analysis_engine="cellprofiler",
        cppipe_path=cppipe,
        cellprofiler_executable="cellprofiler",
    )

    mock_run.assert_called_once_with(
        cppipe_path=cppipe,
        input_dir=input_dir,
        output_dir=output_dir,
        extra_args=None,
        cellprofiler_executable="cellprofiler",
    )
    mock_load.assert_called_once_with(output_dir.resolve())
    mock_merge.assert_called_once()

    assert result["analysis_engine"] == "cellprofiler"
    assert result["output_dir"] == output_dir.resolve()
    assert result["processed"] is None
    assert result["failed"] is None
    assert set(result["tables"].keys()) == {"MyExpt_Image", "MyExpt_Objects"}
    assert result["measurements"] is not None


@patch("bioimage_pipeline.analysis.run_cellprofiler_pipeline")
@patch("bioimage_pipeline.analysis.load_cellprofiler_measurements")
@patch("bioimage_pipeline.analysis.merge_cellprofiler_tables")
def test_run_analysis_cellprofiler_skips_merge_when_disabled(
    mock_merge: MagicMock,
    mock_load: MagicMock,
    mock_run: MagicMock,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    cppipe = tmp_path / "pipeline.cppipe"
    input_dir.mkdir()
    cppipe.write_text("pipeline", encoding="utf-8")

    tables = {"MyExpt_Image": pd.DataFrame({"Image_Number": [1]})}
    mock_run.return_value = output_dir.resolve()
    mock_load.return_value = CellProfilerMeasurementsResult(
        tables=tables,
        metadata={},
        warnings=[],
    )

    result = run_analysis(
        input_dir,
        output_dir,
        analysis_engine="cellprofiler",
        cppipe_path=cppipe,
        merge_measurements=False,
    )

    mock_merge.assert_not_called()
    assert result["measurements"] is None
    assert result["tables"] == tables
