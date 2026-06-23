"""Unit tests for threshold recommender E2E helpers (no real CellProfiler)."""

from __future__ import annotations

from pathlib import Path

import pytest

from bioimage_pipeline.threshold_recommender_e2e import (
    build_spot_detection_cppipe,
    materialize_threshold_recommender_e2e_fixtures,
    validate_threshold_recommender_trial_result,
)
from bioimage_pipeline.threshold_recommender import ThresholdRecommenderTrialResult
from bioimage_pipeline.threshold_subset import SUBSET_MANIFEST_FILENAME, ThresholdSubsetManifest
from bioimage_pipeline.threshold_variant_runner import ThresholdVariantRunResult
from bioimage_pipeline.threshold_variants import ThresholdVariantArtifact, ThresholdVariantSpec
from bioimage_pipeline.threshold_variant_scoring import ThresholdVariantScore


def test_build_spot_detection_cppipe_includes_ipo_and_exports() -> None:
    pipeline = build_spot_detection_cppipe()
    names = [module.name for module in pipeline.modules]
    assert "IdentifyPrimaryObjects" in names
    assert "ExportToSpreadsheet" in names
    assert "SaveImages" in names


def test_materialize_threshold_recommender_e2e_fixtures_writes_files(tmp_path: Path) -> None:
    fixtures = materialize_threshold_recommender_e2e_fixtures(tmp_path, image_count=2)
    assert fixtures.cppipe_path.is_file()
    assert len(fixtures.image_names) == 2
    for name in fixtures.image_names:
        assert (fixtures.input_dir / name).is_file()


def test_validate_threshold_recommender_trial_result_raises_on_empty_ranking(
    tmp_path: Path,
) -> None:
    root = tmp_path / "threshold_recommender"
    subset_dir = root / "subset"
    subset_dir.mkdir(parents=True)
    manifest = ThresholdSubsetManifest(
        source_dir=tmp_path / "input",
        staged_dir=subset_dir,
        mode="auto",
        sample_count=1,
        sample_method="first",
        image_names=["a.tif"],
    )
    trial_result = ThresholdRecommenderTrialResult(
        recommender_root=root,
        subset_dir=subset_dir,
        subset_manifest=manifest,
        variants_dir=root / "optimistic",
        artifacts=[],
        run_results=[],
        summaries=[],
        ranked_scores=[],
        comparison_paths={"csv": tmp_path / "missing_comparison.csv"},
        ranking_paths={"csv": tmp_path / "missing_ranking.csv"},
        session_path=root / "recommender_session.json",
        preview_index={},
        trial_mode="optimistic",
    )

    with pytest.raises(ValueError, match="ranked variant"):
        validate_threshold_recommender_trial_result(trial_result)


def test_validate_threshold_recommender_trial_result_accepts_successful_trial(
    tmp_path: Path,
) -> None:
    root = tmp_path / "threshold_recommender"
    subset_dir = root / "subset"
    subset_dir.mkdir(parents=True)
    subset_manifest_path = subset_dir / SUBSET_MANIFEST_FILENAME
    subset_manifest_path.write_text(
        '{"source_dir":"input","staged_dir":"subset","mode":"auto",'
        '"sample_count":1,"sample_method":"first","image_names":["a.tif"]}',
        encoding="utf-8",
    )
    comparison_csv = root / "comparison.csv"
    ranking_csv = root / "ranking.csv"
    session_path = root / "recommender_session.json"
    session_path.write_text("{}", encoding="utf-8")
    comparison_csv.write_text("variant_id\n001\n", encoding="utf-8")
    ranking_csv.write_text("rank,variant_id\n1,001\n", encoding="utf-8")

    spec = ThresholdVariantSpec(
        variant_id="001_baseline",
        display_name="Baseline",
        target_module_index=4,
    )
    artifact = ThresholdVariantArtifact(
        spec=spec,
        variant_dir=root / "variant_001_baseline",
        pipeline_path=root / "variant_001_baseline" / "pipeline.cppipe",
    )
    run_result = ThresholdVariantRunResult(
        spec=spec,
        variant_dir=artifact.variant_dir,
        pipeline_path=artifact.pipeline_path,
        raw_output_dir=artifact.variant_dir / "cellprofiler_raw",
        measurements_dir=artifact.variant_dir / "measurements",
        qc_dir=artifact.variant_dir / "qc",
        logs_dir=artifact.variant_dir / "logs",
        success=True,
    )
    score = ThresholdVariantScore(
        rank=1,
        variant_id="001_baseline",
        display_name="Baseline",
        score=0.8,
        reason="ok",
    )
    trial_result = ThresholdRecommenderTrialResult(
        recommender_root=root,
        subset_dir=subset_dir,
        subset_manifest=ThresholdSubsetManifest(
            source_dir=tmp_path / "input",
            staged_dir=subset_dir,
            mode="auto",
            sample_count=1,
            sample_method="first",
            image_names=["a.tif"],
        ),
        variants_dir=root / "optimistic",
        artifacts=[artifact],
        run_results=[run_result],
        summaries=[],
        ranked_scores=[score],
        comparison_paths={"csv": comparison_csv},
        ranking_paths={"csv": ranking_csv},
        session_path=session_path,
        preview_index={},
        trial_mode="full_search",
    )

    warnings = validate_threshold_recommender_trial_result(trial_result)
    assert warnings == []
