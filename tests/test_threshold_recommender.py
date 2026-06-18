"""Tests for subset-first threshold recommender orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bioimage_pipeline.threshold_recommender import (
    ThresholdRecommenderConfig,
    apply_confirmed_threshold_variant,
    load_recommender_session,
    run_threshold_recommender_trial,
)
from bioimage_pipeline.threshold_subset import ThresholdSubsetSelection
from bioimage_pipeline.threshold_variant_comparison import (
    ThresholdVariantMeasurementSummary,
)
from bioimage_pipeline.threshold_variant_runner import ThresholdVariantRunResult
from bioimage_pipeline.threshold_variants import ThresholdVariantArtifact, ThresholdVariantSpec
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


def _setup_input_and_cppipe(tmp_path: Path) -> tuple[Path, Path]:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for index in range(6):
        (input_dir / f"img_{index:03d}.tif").write_bytes(b"fake")
    cppipe_path = tmp_path / "pipeline.cppipe"
    cppipe_path.write_text(SAMPLE_CPPIPE, encoding="utf-8")
    return input_dir, cppipe_path


def _artifact(
    variant_dir: Path,
    variant_id: str,
    *,
    display_name: str | None = None,
) -> ThresholdVariantArtifact:
    return ThresholdVariantArtifact(
        spec=ThresholdVariantSpec(
            variant_id=variant_id,
            display_name=display_name or variant_id,
            target_module_index=1,
        ),
        variant_dir=variant_dir,
        pipeline_path=variant_dir / "pipeline.cppipe",
    )


def _run_result(artifact: ThresholdVariantArtifact) -> ThresholdVariantRunResult:
    return ThresholdVariantRunResult(
        spec=artifact.spec,
        variant_dir=artifact.variant_dir,
        pipeline_path=artifact.pipeline_path,
        raw_output_dir=artifact.variant_dir / "cellprofiler_raw",
        measurements_dir=artifact.variant_dir / "measurements",
        qc_dir=artifact.variant_dir / "qc",
        logs_dir=artifact.variant_dir / "logs",
        success=True,
    )


def _good_optimistic_summary(artifact: ThresholdVariantArtifact) -> ThresholdVariantMeasurementSummary:
    return ThresholdVariantMeasurementSummary(
        variant_id=artifact.spec.variant_id,
        display_name=artifact.spec.display_name,
        success=True,
        object_count=120,
        tiny_frac=0.05,
        huge_frac=0.02,
        normal_frac=0.93,
        median_intensity=180.0,
    )


@patch("bioimage_pipeline.threshold_recommender.run_threshold_variant_artifacts")
@patch("bioimage_pipeline.threshold_recommender.write_threshold_pipeline_variants")
def test_run_threshold_recommender_trial_stages_subset_and_saves_session(
    mock_write: MagicMock,
    mock_run: MagicMock,
    tmp_path: Path,
) -> None:
    input_dir, cppipe_path = _setup_input_and_cppipe(tmp_path)
    variant_root = tmp_path / "out" / "threshold_recommender" / "threshold_variants"
    artifact = _artifact(variant_root / "variant_001_baseline", "001_baseline")
    artifact.variant_dir.mkdir(parents=True)
    artifact.pipeline_path.write_text(SAMPLE_CPPIPE, encoding="utf-8")
    mock_write.return_value = [artifact]
    mock_run.return_value = [_run_result(artifact)]

    config = ThresholdRecommenderConfig(
        imported_cppipe_path=cppipe_path,
        input_dir=input_dir,
        output_dir=tmp_path / "out",
        subset_selection=ThresholdSubsetSelection(sample_count=3, sample_method="even"),
        max_variants=1,
        fast_optimistic=False,
    )
    trial_result = run_threshold_recommender_trial(config)

    assert len(trial_result.subset_manifest.image_names) == 3
    assert (trial_result.subset_dir / trial_result.subset_manifest.image_names[0]).is_file()
    assert trial_result.session_path.is_file()
    mock_run.assert_called_once()
    run_input = mock_run.call_args.args[1]
    assert Path(run_input).resolve() == trial_result.subset_dir.resolve()
    assert cppipe_path.read_text(encoding="utf-8") == SAMPLE_CPPIPE


@patch("bioimage_pipeline.threshold_recommender.compare_threshold_variant_run_results")
@patch("bioimage_pipeline.threshold_recommender.run_threshold_variant_artifacts")
@patch("bioimage_pipeline.threshold_recommender.write_threshold_pipeline_variants")
def test_run_threshold_recommender_trial_optimistic_pass_skips_full_search(
    mock_write: MagicMock,
    mock_run: MagicMock,
    mock_compare: MagicMock,
    tmp_path: Path,
) -> None:
    input_dir, cppipe_path = _setup_input_and_cppipe(tmp_path)
    optimistic_root = tmp_path / "out" / "threshold_recommender" / "optimistic"
    artifact = _artifact(
        optimistic_root / "variant_001_optimistic_otsu_adaptive",
        "001_optimistic_otsu_adaptive",
        display_name="Optimistic Otsu Adaptive",
    )
    artifact.variant_dir.mkdir(parents=True)
    artifact.pipeline_path.write_text(SAMPLE_CPPIPE, encoding="utf-8")
    mock_write.return_value = [artifact]
    mock_run.return_value = [_run_result(artifact)]
    mock_compare.return_value = [_good_optimistic_summary(artifact)]

    config = ThresholdRecommenderConfig(
        imported_cppipe_path=cppipe_path,
        input_dir=input_dir,
        output_dir=tmp_path / "out",
        subset_selection=ThresholdSubsetSelection(sample_count=3, sample_method="even"),
        fast_optimistic=True,
    )
    trial_result = run_threshold_recommender_trial(config)

    assert trial_result.trial_mode == "optimistic"
    assert trial_result.fell_back_to_full_search is False
    assert trial_result.optimistic_qc is not None
    assert trial_result.optimistic_qc.passed is True
    assert trial_result.optimistic_qc_path is not None
    assert trial_result.optimistic_qc_path.is_file()
    assert mock_run.call_count == 1
    assert mock_write.call_count == 1


@patch("bioimage_pipeline.threshold_recommender.compare_threshold_variant_run_results")
@patch("bioimage_pipeline.threshold_recommender.run_threshold_variant_artifacts")
@patch("bioimage_pipeline.threshold_recommender.write_threshold_pipeline_variants")
def test_run_threshold_recommender_trial_optimistic_fail_falls_back_to_full_search(
    mock_write: MagicMock,
    mock_run: MagicMock,
    mock_compare: MagicMock,
    tmp_path: Path,
) -> None:
    input_dir, cppipe_path = _setup_input_and_cppipe(tmp_path)
    optimistic_root = tmp_path / "out" / "threshold_recommender" / "optimistic"
    variant_root = tmp_path / "out" / "threshold_recommender" / "threshold_variants"
    optimistic_artifact = _artifact(
        optimistic_root / "variant_001_optimistic_otsu_adaptive",
        "001_optimistic_otsu_adaptive",
        display_name="Optimistic Otsu Adaptive",
    )
    full_artifact = _artifact(variant_root / "variant_001_baseline", "001_baseline")
    optimistic_artifact.variant_dir.mkdir(parents=True)
    optimistic_artifact.pipeline_path.write_text(SAMPLE_CPPIPE, encoding="utf-8")
    full_artifact.variant_dir.mkdir(parents=True)
    full_artifact.pipeline_path.write_text(SAMPLE_CPPIPE, encoding="utf-8")

    mock_write.side_effect = [[optimistic_artifact], [full_artifact]]
    mock_run.side_effect = [
        [_run_result(optimistic_artifact)],
        [_run_result(full_artifact)],
    ]
    mock_compare.side_effect = [
        [
            ThresholdVariantMeasurementSummary(
                variant_id=optimistic_artifact.spec.variant_id,
                display_name=optimistic_artifact.spec.display_name,
                success=True,
                object_count=0,
            )
        ],
        [_good_optimistic_summary(full_artifact)],
    ]

    config = ThresholdRecommenderConfig(
        imported_cppipe_path=cppipe_path,
        input_dir=input_dir,
        output_dir=tmp_path / "out",
        subset_selection=ThresholdSubsetSelection(sample_count=3, sample_method="even"),
        fast_optimistic=True,
        max_variants=1,
    )
    trial_result = run_threshold_recommender_trial(config)

    assert trial_result.trial_mode == "full_search"
    assert trial_result.fell_back_to_full_search is True
    assert trial_result.optimistic_qc is not None
    assert trial_result.optimistic_qc.passed is False
    assert mock_run.call_count == 2
    assert mock_write.call_count == 2
    session = load_recommender_session(tmp_path / "out" / "threshold_recommender")
    assert session["fell_back_to_full_search"] is True


def test_apply_confirmed_threshold_variant_requires_confirmation(tmp_path: Path) -> None:
    config = ThresholdRecommenderConfig(
        imported_cppipe_path=tmp_path / "pipeline.cppipe",
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "out",
    )
    with pytest.raises(ValueError, match="explicit confirmation"):
        apply_confirmed_threshold_variant(config, "001_baseline", confirmed=False)


@patch("bioimage_pipeline.threshold_recommender.run_threshold_variant_artifact")
def test_apply_confirmed_threshold_variant_copies_pipeline_into_confirmed_run(
    mock_run: MagicMock,
    tmp_path: Path,
) -> None:
    input_dir, cppipe_path = _setup_input_and_cppipe(tmp_path)
    variant_root = tmp_path / "out" / "threshold_recommender" / "threshold_variants"
    artifact = _artifact(variant_root / "variant_001_baseline", "001_baseline")
    artifact.variant_dir.mkdir(parents=True)
    artifact.pipeline_path.write_text(SAMPLE_CPPIPE, encoding="utf-8")

    config = ThresholdRecommenderConfig(
        imported_cppipe_path=cppipe_path,
        input_dir=input_dir,
        output_dir=tmp_path / "out",
        max_variants=1,
        fast_optimistic=False,
    )
    with patch(
        "bioimage_pipeline.threshold_recommender.write_threshold_pipeline_variants",
        return_value=[artifact],
    ), patch(
        "bioimage_pipeline.threshold_recommender.run_threshold_variant_artifacts",
        return_value=[_run_result(artifact)],
    ):
        run_threshold_recommender_trial(config)

    mock_run.return_value = _run_result(artifact)
    apply_config = ThresholdRecommenderConfig(
        imported_cppipe_path=cppipe_path,
        input_dir=input_dir,
        output_dir=tmp_path / "out",
    )
    apply_result = apply_confirmed_threshold_variant(
        apply_config,
        "001_baseline",
        confirmed=True,
    )

    confirmed_pipeline = apply_result.confirmed_run_dir / "pipeline.cppipe"
    assert confirmed_pipeline.is_file()
    used_artifact = mock_run.call_args.args[0]
    assert used_artifact.pipeline_path.resolve() == confirmed_pipeline.resolve()
    assert used_artifact.variant_dir.resolve() == apply_result.confirmed_run_dir.resolve()
    assert cppipe_path.read_text(encoding="utf-8") == SAMPLE_CPPIPE
    session = load_recommender_session(tmp_path / "out" / "threshold_recommender")
    assert session["imported_cppipe_path"] == str(cppipe_path.resolve())
