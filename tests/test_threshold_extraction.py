"""Tests for read-only IdentifyPrimaryObjects threshold setting extraction."""

from __future__ import annotations

from pathlib import Path

from bioimage_pipeline.cppipe_io import (
    append_module,
    create_pipeline_from_catalog,
    parse_cppipe_text,
    update_module_setting,
)
from bioimage_pipeline.threshold_extraction import (
    IDENTIFY_PRIMARY_OBJECTS_THRESHOLD_SETTING_KEYS,
    extract_identify_primary_objects_threshold_profiles,
    load_identify_primary_objects_threshold_profiles,
)

SAMPLE_CPPIPE = """CellProfiler Pipeline: http://www.cellprofiler.org
Version:5

Images:[module_num:1|svn_version:'Unknown'|variable_revision_number:1|show_window:False|notes:[]]
Filter images?:No

IdentifyPrimaryObjects:[module_num:2|svn_version:'Unknown'|variable_revision_number:1|show_window:False|notes:[]]
Select the input image:Green
Name the primary objects to be identified:Spots
Threshold strategy:Adaptive
Thresholding method:Otsu
Threshold smoothing scale:1.2
Threshold correction factor:0.95
Lower and upper bounds on threshold:0.05,0.9
Typical diameter of objects, in pixel units (Min,Max):3,12
Method to distinguish clumped objects:None

ExportToSpreadsheet:[module_num:3|svn_version:'Unknown'|variable_revision_number:1|show_window:False|notes:[]]
Select the column delimiter:Comma
"""


def test_extract_returns_empty_when_no_ipo_modules() -> None:
    pipeline = parse_cppipe_text(
        """CellProfiler Pipeline: http://www.cellprofiler.org
Version:5

Images:[module_num:1|svn_version:'Unknown'|variable_revision_number:1|show_window:False|notes:[]]
Filter images?:No
"""
    )
    assert extract_identify_primary_objects_threshold_profiles(pipeline) == []


def test_extract_reads_threshold_settings_from_imported_pipeline() -> None:
    pipeline = parse_cppipe_text(SAMPLE_CPPIPE)
    profiles = extract_identify_primary_objects_threshold_profiles(pipeline)

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.module_index == 1
    assert profile.module_num == 2
    assert profile.input_image == "Green"
    assert profile.object_name == "Spots"
    assert profile.threshold_strategy == "Adaptive"
    assert profile.thresholding_method == "Otsu"
    assert profile.threshold_smoothing_scale == "1.2"
    assert profile.threshold_correction_factor == "0.95"
    assert profile.threshold_bounds == "0.05,0.9"
    assert profile.typical_diameter == "3,12"
    assert profile.declumping_method == "None"
    assert profile.threshold_settings["Threshold strategy"] == "Adaptive"
    assert profile.threshold_settings["Threshold correction factor"] == "0.95"
    assert "Select the input image" not in profile.threshold_settings
    assert profile.all_module_settings["Select the input image"] == "Green"


def test_extract_supports_multiple_ipo_modules() -> None:
    pipeline = parse_cppipe_text(SAMPLE_CPPIPE)
    pipeline = append_module(pipeline, "IdentifyPrimaryObjects")
    second_ipo_index = len(pipeline.modules) - 1
    pipeline = update_module_setting(
        pipeline,
        second_ipo_index,
        "Name the primary objects to be identified",
        "RedSpots",
    )
    pipeline = update_module_setting(
        pipeline,
        second_ipo_index,
        "Threshold strategy",
        "Global",
    )

    profiles = extract_identify_primary_objects_threshold_profiles(pipeline)
    assert len(profiles) == 2
    assert profiles[0].object_name == "Spots"
    assert profiles[0].threshold_strategy == "Adaptive"
    assert profiles[1].object_name == "RedSpots"
    assert profiles[1].threshold_strategy == "Global"


def test_extract_from_catalog_ipo_module_includes_defaults() -> None:
    pipeline = append_module(create_pipeline_from_catalog(), "IdentifyPrimaryObjects")
    profiles = extract_identify_primary_objects_threshold_profiles(pipeline)

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.threshold_strategy == "Global"
    assert profile.thresholding_method == "Otsu"
    assert profile.threshold_correction_factor == "1"
    assert profile.typical_diameter == "10,40"
    assert profile.threshold_settings.keys() <= IDENTIFY_PRIMARY_OBJECTS_THRESHOLD_SETTING_KEYS


def test_load_from_cppipe_path(tmp_path: Path) -> None:
    cppipe_path = tmp_path / "pipeline.cppipe"
    original_text = SAMPLE_CPPIPE
    cppipe_path.write_text(original_text, encoding="utf-8")

    profiles = load_identify_primary_objects_threshold_profiles(cppipe_path)
    assert len(profiles) == 1
    assert profiles[0].object_name == "Spots"
    assert cppipe_path.read_text(encoding="utf-8") == original_text
