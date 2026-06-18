"""Tests for IdentifyPrimaryObjects threshold pipeline variant generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from bioimage_pipeline.cppipe_io import (
    append_module,
    load_cppipe,
    parse_cppipe_text,
    update_module_setting,
)
from bioimage_pipeline.threshold_extraction import (
    IDENTIFY_PRIMARY_OBJECTS_THRESHOLD_SETTING_KEYS,
    extract_identify_primary_objects_threshold_profiles,
)
from bioimage_pipeline.threshold_variants import (
    AmbiguousIdentifyPrimaryObjectsError,
    OPTIMISTIC_CORRECTION_FACTOR,
    apply_threshold_variant_spec,
    estimate_adaptive_window_size_from_diameter,
    estimate_threshold_smoothing_scale_from_diameter,
    generate_basic_threshold_variant_specs,
    generate_optimistic_threshold_variant_spec,
    profile_supports_robust_background,
    select_ipo_threshold_profile,
    write_threshold_pipeline_variants,
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

MODERN_IPO_CPPIPE = """CellProfiler Pipeline: http://www.cellprofiler.org
Version:5

Images:[module_num:1|svn_version:'Unknown'|variable_revision_number:1|show_window:False|notes:[]]
Filter images?:No

IdentifyPrimaryObjects:[module_num:2|svn_version:'Unknown'|variable_revision_number:14|show_window:False|notes:[]]
Select the input image:Green
Name the primary objects to be identified:Spots
Use advanced settings?:Yes
Threshold setting version:11
Threshold strategy:Adaptive
Thresholding method:Otsu
Threshold smoothing scale:1.2
Threshold correction factor:0.95
Lower and upper bounds on threshold:0.05,0.9
Size of adaptive window:10
Typical diameter of objects, in pixel units (Min,Max):3,12
Method to distinguish clumped objects:None

ExportToSpreadsheet:[module_num:3|svn_version:'Unknown'|variable_revision_number:1|show_window:False|notes:[]]
Select the column delimiter:Comma
"""


def _profile_from_sample() -> tuple[object, object]:
    pipeline = parse_cppipe_text(SAMPLE_CPPIPE)
    profile = extract_identify_primary_objects_threshold_profiles(pipeline)[0]
    return pipeline, profile


def test_generate_basic_variant_specs_from_sample_profile() -> None:
    _, profile = _profile_from_sample()
    specs = generate_basic_threshold_variant_specs(profile)

    assert specs[0].is_baseline is True
    assert specs[0].variant_id == "001_baseline"
    assert specs[1].display_name == "Otsu Global"
    assert specs[1].thresholding_method == "Otsu"
    assert specs[1].threshold_strategy == "Global"
    assert specs[2].threshold_strategy == "Adaptive"
    assert specs[2].threshold_correction_factor == "0.9"
    assert specs[5].display_name == "Minimum Cross-Entropy Global"
    assert all(
        spec.target_module_index == profile.module_index for spec in specs
    )
    assert len(specs) == 9
    assert not profile_supports_robust_background(profile)


def test_generate_basic_variant_specs_includes_robust_background_when_supported() -> None:
    pipeline = parse_cppipe_text(MODERN_IPO_CPPIPE)
    profile = extract_identify_primary_objects_threshold_profiles(pipeline)[0]

    assert profile_supports_robust_background(profile)
    specs = generate_basic_threshold_variant_specs(profile)
    robust_specs = [
        spec for spec in specs if spec.thresholding_method == "Robust Background"
    ]
    assert len(robust_specs) == 1
    assert robust_specs[0].threshold_strategy == "Global"


def test_baseline_variant_preserves_original_pipeline_text(tmp_path: Path) -> None:
    cppipe_path = tmp_path / "imported.cppipe"
    original_text = SAMPLE_CPPIPE
    cppipe_path.write_text(original_text, encoding="utf-8")

    _, profile = _profile_from_sample()
    specs = generate_basic_threshold_variant_specs(profile)
    baseline_spec = specs[0]

    artifacts = write_threshold_pipeline_variants(
        cppipe_path,
        tmp_path / "threshold_variants",
        [baseline_spec],
    )

    assert len(artifacts) == 1
    assert artifacts[0].pipeline_path.name == "pipeline.cppipe"
    assert artifacts[0].variant_dir.name == "variant_001_baseline"
    assert artifacts[0].pipeline_path.read_text(encoding="utf-8") == original_text
    assert cppipe_path.read_text(encoding="utf-8") == original_text


def test_modified_variants_change_only_threshold_settings(tmp_path: Path) -> None:
    cppipe_path = tmp_path / "imported.cppipe"
    cppipe_path.write_text(SAMPLE_CPPIPE, encoding="utf-8")

    pipeline = load_cppipe(cppipe_path)
    _, profile = _profile_from_sample()
    specs = generate_basic_threshold_variant_specs(profile)
    modified_specs = [spec for spec in specs if not spec.is_baseline]

    artifacts = write_threshold_pipeline_variants(
        cppipe_path,
        tmp_path / "threshold_variants",
        modified_specs,
    )
    assert len(artifacts) == len(modified_specs)

    original_settings = {
        setting.key: setting.value
        for setting in pipeline.modules[profile.module_index].settings
    }

    for artifact in artifacts:
        variant_pipeline = load_cppipe(artifact.pipeline_path)
        variant_settings = {
            setting.key: setting.value
            for setting in variant_pipeline.modules[profile.module_index].settings
        }

        spec_threshold_values = {
            "Threshold strategy": artifact.spec.threshold_strategy,
            "Thresholding method": artifact.spec.thresholding_method,
            "Threshold correction factor": artifact.spec.threshold_correction_factor,
            "Threshold smoothing scale": artifact.spec.threshold_smoothing_scale,
            "Lower and upper bounds on threshold": artifact.spec.threshold_bounds,
            "Size of adaptive window": artifact.spec.adaptive_window_size,
        }
        applied_threshold_keys = {
            key for key, value in spec_threshold_values.items() if value is not None
        }

        for key in IDENTIFY_PRIMARY_OBJECTS_THRESHOLD_SETTING_KEYS:
            if key in applied_threshold_keys:
                assert variant_settings.get(key) == spec_threshold_values[key]
            else:
                assert variant_settings.get(key) == original_settings.get(key)

        for key, value in original_settings.items():
            if key in IDENTIFY_PRIMARY_OBJECTS_THRESHOLD_SETTING_KEYS:
                continue
            assert variant_settings.get(key) == value


def test_original_cppipe_is_unchanged_after_writing_variants(tmp_path: Path) -> None:
    cppipe_path = tmp_path / "imported.cppipe"
    original_text = SAMPLE_CPPIPE
    cppipe_path.write_text(original_text, encoding="utf-8")

    _, profile = _profile_from_sample()
    specs = generate_basic_threshold_variant_specs(profile)

    write_threshold_pipeline_variants(
        cppipe_path,
        tmp_path / "threshold_variants",
        specs,
    )

    assert cppipe_path.read_text(encoding="utf-8") == original_text


def test_select_ipo_profile_requires_explicit_target_for_multiple_modules() -> None:
    pipeline = parse_cppipe_text(SAMPLE_CPPIPE)
    pipeline = append_module(pipeline, "IdentifyPrimaryObjects")
    second_ipo_index = len(pipeline.modules) - 1
    pipeline = update_module_setting(
        pipeline,
        second_ipo_index,
        "Name the primary objects to be identified",
        "RedSpots",
    )
    profiles = extract_identify_primary_objects_threshold_profiles(pipeline)

    with pytest.raises(AmbiguousIdentifyPrimaryObjectsError):
        select_ipo_threshold_profile(profiles)

    selected = select_ipo_threshold_profile(
        profiles,
        object_name="RedSpots",
    )
    assert selected.object_name == "RedSpots"

    specs = generate_basic_threshold_variant_specs(selected)
    assert all(spec.target_module_index == second_ipo_index for spec in specs)


def test_write_variants_targets_only_selected_ipo_module(tmp_path: Path) -> None:
    pipeline = parse_cppipe_text(SAMPLE_CPPIPE)
    pipeline = append_module(pipeline, "IdentifyPrimaryObjects")
    second_ipo_index = len(pipeline.modules) - 1
    pipeline = update_module_setting(
        pipeline,
        second_ipo_index,
        "Name the primary objects to be identified",
        "RedSpots",
    )

    cppipe_path = tmp_path / "imported.cppipe"
    cppipe_path.write_text(pipeline.to_text(), encoding="utf-8")

    profiles = extract_identify_primary_objects_threshold_profiles(pipeline)
    selected = select_ipo_threshold_profile(profiles, object_name="Spots")
    spec = generate_basic_threshold_variant_specs(selected)[1]

    artifacts = write_threshold_pipeline_variants(
        cppipe_path,
        tmp_path / "threshold_variants",
        [spec],
    )
    variant_pipeline = load_cppipe(artifacts[0].pipeline_path)

    first_settings = {
        setting.key: setting.value
        for setting in variant_pipeline.modules[profiles[0].module_index].settings
    }
    second_settings = {
        setting.key: setting.value
        for setting in variant_pipeline.modules[profiles[1].module_index].settings
    }

    assert first_settings["Threshold strategy"] == "Global"
    assert second_settings["Threshold strategy"] == profiles[1].threshold_strategy


def test_generate_optimistic_variant_spec_uses_otsu_adaptive_and_correction_factor() -> None:
    _, profile = _profile_from_sample()
    spec = generate_optimistic_threshold_variant_spec(profile)

    assert spec.variant_id == "001_optimistic_otsu_adaptive"
    assert spec.thresholding_method == "Otsu"
    assert spec.threshold_strategy == "Adaptive"
    assert spec.threshold_correction_factor == OPTIMISTIC_CORRECTION_FACTOR
    assert spec.threshold_smoothing_scale == "1.2"
    assert spec.adaptive_window_size is None


def test_estimate_smoothing_and_adaptive_window_from_typical_diameter() -> None:
    assert estimate_threshold_smoothing_scale_from_diameter("3,12") == "1.2"
    assert estimate_threshold_smoothing_scale_from_diameter(None) is None
    assert estimate_adaptive_window_size_from_diameter("3,12") == "37"
    assert estimate_adaptive_window_size_from_diameter("bad") is None


def test_optimistic_variant_changes_only_threshold_settings(tmp_path: Path) -> None:
    cppipe_path = tmp_path / "imported.cppipe"
    cppipe_path.write_text(MODERN_IPO_CPPIPE, encoding="utf-8")

    pipeline = load_cppipe(cppipe_path)
    profile = extract_identify_primary_objects_threshold_profiles(pipeline)[0]
    spec = generate_optimistic_threshold_variant_spec(profile)

    assert spec.adaptive_window_size == "37"

    artifacts = write_threshold_pipeline_variants(
        cppipe_path,
        tmp_path / "optimistic",
        [spec],
    )
    variant_pipeline = load_cppipe(artifacts[0].pipeline_path)
    variant_settings = {
        setting.key: setting.value
        for setting in variant_pipeline.modules[profile.module_index].settings
    }
    original_settings = {
        setting.key: setting.value
        for setting in pipeline.modules[profile.module_index].settings
    }

    assert variant_settings["Threshold strategy"] == "Adaptive"
    assert variant_settings["Thresholding method"] == "Otsu"
    assert variant_settings["Threshold correction factor"] == OPTIMISTIC_CORRECTION_FACTOR
    assert variant_settings["Threshold smoothing scale"] == "1.2"
    assert variant_settings["Size of adaptive window"] == "37"
    assert variant_settings["Typical diameter of objects, in pixel units (Min,Max)"] == (
        original_settings["Typical diameter of objects, in pixel units (Min,Max)"]
    )
    assert variant_settings["Method to distinguish clumped objects"] == (
        original_settings["Method to distinguish clumped objects"]
    )


def test_apply_threshold_variant_spec_is_read_only_on_baseline() -> None:
    pipeline, profile = _profile_from_sample()
    baseline_spec = generate_basic_threshold_variant_specs(profile)[0]
    unchanged = apply_threshold_variant_spec(pipeline, baseline_spec)
    assert unchanged is pipeline
