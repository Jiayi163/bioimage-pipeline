"""Tests for threshold variant measurement comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bioimage_pipeline.threshold_variant_comparison import (
    ThresholdVariantMeasurementSummary,
    ThresholdVariantPerImageSummary,
    ThresholdVariantSizeThresholds,
    compare_threshold_variant_per_image,
    compare_threshold_variant_run_results,
    load_threshold_variant_per_image_comparison,
    save_threshold_variant_comparison,
    save_threshold_variant_per_image_comparison,
    summarize_threshold_variant_measurements,
    threshold_variant_comparison_to_dataframe,
)
from bioimage_pipeline.threshold_variant_runner import ThresholdVariantRunResult
from bioimage_pipeline.threshold_variants import ThresholdVariantSpec

SPOTS_OBJECTS_CSV = """Image_Number,ObjectNumber,AreaShape_Area,Location_Center_X,Location_Center_Y,Intensity_MeanIntensity_Green
1,1,1.0,10.0,10.0,50.0
1,2,4.0,20.0,20.0,100.0
1,3,9.0,30.0,30.0,150.0
1,4,250.0,40.0,40.0,300.0
"""

IMAGE_CSV = """Image_Number,FileName,Count_Spots
1,sample.tif,4
"""


def _write_measurements(
    measurements_dir: Path,
    *,
    object_csv_name: str = "MyExpt_Spots.csv",
) -> None:
    measurements_dir.mkdir(parents=True, exist_ok=True)
    (measurements_dir / object_csv_name).write_text(SPOTS_OBJECTS_CSV, encoding="utf-8")
    (measurements_dir / "MyExpt_Image.csv").write_text(IMAGE_CSV, encoding="utf-8")


def _run_result(
    measurements_dir: Path,
    *,
    variant_id: str,
    display_name: str,
    success: bool = True,
    error_message: str | None = None,
) -> ThresholdVariantRunResult:
    spec = ThresholdVariantSpec(
        variant_id=variant_id,
        display_name=display_name,
        target_module_index=1,
    )
    return ThresholdVariantRunResult(
        spec=spec,
        variant_dir=measurements_dir.parent,
        pipeline_path=measurements_dir.parent / "pipeline.cppipe",
        raw_output_dir=measurements_dir.parent / "cellprofiler_raw",
        measurements_dir=measurements_dir,
        qc_dir=measurements_dir.parent / "qc",
        logs_dir=measurements_dir.parent / "logs",
        success=success,
        error_message=error_message,
    )


def test_summarize_detects_object_table_by_columns_not_filename(
    tmp_path: Path,
) -> None:
    measurements_dir = tmp_path / "measurements"
    _write_measurements(measurements_dir, object_csv_name="CustomObjectExport.csv")

    summary = summarize_threshold_variant_measurements(
        measurements_dir,
        variant_id="001_baseline",
        display_name="Baseline (original)",
        success=True,
    )

    assert summary.object_count == 4
    assert summary.object_table_names == ["CustomObjectExport"]
    assert summary.median_area == pytest.approx(6.5)
    assert summary.median_intensity == pytest.approx(125.0)


def test_summarize_computes_tiny_and_huge_fractions_with_configurable_thresholds(
    tmp_path: Path,
) -> None:
    measurements_dir = tmp_path / "measurements"
    _write_measurements(measurements_dir)

    summary = summarize_threshold_variant_measurements(
        measurements_dir,
        variant_id="003_otsu_adaptive_cf_0_9",
        display_name="Otsu Adaptive (CF 0.9)",
        success=True,
        size_thresholds=ThresholdVariantSizeThresholds(
            tiny_area_px=2.0,
            huge_area_px=200.0,
        ),
    )

    assert summary.tiny_frac == pytest.approx(0.25)
    assert summary.huge_frac == pytest.approx(0.25)
    assert summary.normal_frac == pytest.approx(0.5)


def test_summarize_respects_custom_size_thresholds(tmp_path: Path) -> None:
    measurements_dir = tmp_path / "measurements"
    _write_measurements(measurements_dir)

    summary = summarize_threshold_variant_measurements(
        measurements_dir,
        variant_id="002_otsu_global",
        display_name="Otsu Global",
        success=True,
        size_thresholds=ThresholdVariantSizeThresholds(
            tiny_area_px=10.0,
            huge_area_px=100.0,
        ),
    )

    assert summary.tiny_frac == pytest.approx(0.75)
    assert summary.huge_frac == pytest.approx(0.25)
    assert summary.normal_frac == pytest.approx(0.0)


def test_summarize_failed_variant_returns_empty_metrics(tmp_path: Path) -> None:
    measurements_dir = tmp_path / "measurements"
    _write_measurements(measurements_dir)

    summary = summarize_threshold_variant_measurements(
        measurements_dir,
        variant_id="002_otsu_global",
        display_name="Otsu Global",
        success=False,
        error_message="CellProfiler failed",
    )

    assert summary.success is False
    assert summary.object_count is None
    assert summary.median_area is None
    assert summary.error_message == "CellProfiler failed"


def test_compare_threshold_variant_run_results_builds_summary_table(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "variant_001_baseline" / "measurements"
    global_dir = tmp_path / "variant_002_otsu_global" / "measurements"
    _write_measurements(baseline_dir)
    _write_measurements(global_dir)

    results = [
        _run_result(baseline_dir, variant_id="001_baseline", display_name="Baseline"),
        _run_result(global_dir, variant_id="002_otsu_global", display_name="Otsu Global"),
        _run_result(
            tmp_path / "variant_003_failed" / "measurements",
            variant_id="003_failed",
            display_name="Failed",
            success=False,
            error_message="boom",
        ),
    ]
    (tmp_path / "variant_003_failed" / "measurements").mkdir(parents=True)

    summaries = compare_threshold_variant_run_results(results)
    dataframe = threshold_variant_comparison_to_dataframe(summaries)

    assert len(summaries) == 3
    assert list(dataframe["variant_id"]) == [
        "001_baseline",
        "002_otsu_global",
        "003_failed",
    ]
    assert dataframe.loc[0, "object_count"] == 4
    assert pd.isna(dataframe.loc[2, "object_count"])


def test_save_threshold_variant_comparison_writes_csv_and_json(
    tmp_path: Path,
) -> None:
    summary = ThresholdVariantMeasurementSummary(
        variant_id="001_baseline",
        display_name="Baseline (original)",
        success=True,
        object_count=842,
        median_area=6.2,
        tiny_frac=0.05,
        huge_frac=0.01,
        median_intensity=183.4,
        object_table_names=["MyExpt_Spots"],
    )

    paths = save_threshold_variant_comparison([summary], tmp_path / "comparison")

    assert paths["csv"].exists()
    assert paths["json"].exists()

    saved_csv = pd.read_csv(paths["csv"])
    assert saved_csv.loc[0, "variant_id"] == "001_baseline"
    assert saved_csv.loc[0, "object_count"] == 842

    saved_json = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert saved_json[0]["display_name"] == "Baseline (original)"
    assert saved_json[0]["median_intensity"] == pytest.approx(183.4)


def test_summarize_warns_when_no_object_tables_found(tmp_path: Path) -> None:
    measurements_dir = tmp_path / "measurements"
    measurements_dir.mkdir(parents=True)
    (measurements_dir / "MyExpt_Image.csv").write_text(IMAGE_CSV, encoding="utf-8")

    summary = summarize_threshold_variant_measurements(
        measurements_dir,
        variant_id="001_baseline",
        display_name="Baseline",
        success=True,
    )

    assert summary.object_count is None
    assert any("No object measurement tables" in warning for warning in summary.warnings)


def test_compare_threshold_variant_per_image_builds_rows(tmp_path: Path) -> None:
    measurements_dir = tmp_path / "variant_001_baseline" / "measurements"
    _write_measurements(measurements_dir)
    run_result = _run_result(
        measurements_dir,
        variant_id="001_baseline",
        display_name="Baseline (original)",
    )

    rows = compare_threshold_variant_per_image(
        [run_result],
        image_names=["sample.tif"],
    )

    assert len(rows) == 1
    assert rows[0].object_count == 4
    assert rows[0].image_name == "sample.tif"


def test_load_threshold_variant_per_image_comparison_round_trip(tmp_path: Path) -> None:
    rows = [
        ThresholdVariantPerImageSummary(
            variant_id="001_baseline",
            display_name="Baseline",
            image_number=1,
            image_name="sample.tif",
            success=True,
            object_count=10,
            foreground_coverage=0.02,
            warnings=["test warning"],
        )
    ]
    paths = save_threshold_variant_per_image_comparison(rows, tmp_path)
    loaded = load_threshold_variant_per_image_comparison(paths["json"])

    assert len(loaded) == 1
    assert loaded[0].variant_id == "001_baseline"
    assert loaded[0].object_count == 10
    assert loaded[0].warnings == ["test warning"]
