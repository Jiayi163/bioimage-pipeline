"""Tests for threshold variant CellProfiler batch runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bioimage_pipeline.cellprofiler_runner import (
    RESULTS_LOGS_DIR,
    RESULTS_MEASUREMENTS_DIR,
    RESULTS_QC_DIR,
    RESULTS_RAW_DIR,
    CellProfilerRunResult,
)
from bioimage_pipeline.threshold_variant_runner import (
    run_threshold_variant_artifact,
    run_threshold_variant_artifacts,
)
from bioimage_pipeline.threshold_variants import (
    ThresholdVariantArtifact,
    ThresholdVariantSpec,
    write_threshold_pipeline_variants,
)
from bioimage_pipeline.threshold_extraction import (
    extract_identify_primary_objects_threshold_profiles,
)
from bioimage_pipeline.cppipe_io import parse_cppipe_text
from bioimage_pipeline.threshold_variants import generate_basic_threshold_variant_specs

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


def _successful_run_result(output_dir: Path, log_dir: Path) -> CellProfilerRunResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_files = {
        "stdout": log_dir / "cellprofiler_stdout.log",
        "stderr": log_dir / "cellprofiler_stderr.log",
    }
    for path in log_files.values():
        path.write_text("ok", encoding="utf-8")
    (output_dir / "MyExpt_Image.csv").write_text(
        "Image_Number,FileName\n1,sample.tif\n",
        encoding="utf-8",
    )
    return CellProfilerRunResult(
        output_dir=output_dir.resolve(),
        command=["cellprofiler", "-c", "-r"],
        returncode=0,
        stdout="done",
        stderr="",
        log_files=log_files,
    )


def _failed_run_result(output_dir: Path, log_dir: Path) -> CellProfilerRunResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_files = {
        "stdout": log_dir / "cellprofiler_stdout.log",
        "stderr": log_dir / "cellprofiler_stderr.log",
    }
    for path in log_files.values():
        path.write_text("error", encoding="utf-8")
    return CellProfilerRunResult(
        output_dir=output_dir.resolve(),
        command=["cellprofiler", "-c", "-r"],
        returncode=1,
        stdout="",
        stderr="CellProfiler failed",
        log_files=log_files,
    )


def _make_artifacts(tmp_path: Path) -> tuple[Path, Path, list[ThresholdVariantArtifact]]:
    imported_path = tmp_path / "imported.cppipe"
    imported_path.write_text(SAMPLE_CPPIPE, encoding="utf-8")
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    pipeline = parse_cppipe_text(SAMPLE_CPPIPE)
    profile = extract_identify_primary_objects_threshold_profiles(pipeline)[0]
    specs = generate_basic_threshold_variant_specs(profile)[:3]
    variants_root = tmp_path / "threshold_variants"
    artifacts = write_threshold_pipeline_variants(
        imported_path,
        variants_root,
        specs,
    )
    return imported_path, input_dir, artifacts


def _artifact_spec(
    variant_dir: Path,
    *,
    variant_id: str = "001_baseline",
    pipeline_path: Path | None = None,
) -> ThresholdVariantArtifact:
    pipeline = pipeline_path or (variant_dir / "pipeline.cppipe")
    return ThresholdVariantArtifact(
        spec=ThresholdVariantSpec(
            variant_id=variant_id,
            display_name="Baseline",
            target_module_index=1,
            is_baseline=True,
        ),
        variant_dir=variant_dir,
        pipeline_path=pipeline,
    )


@patch("bioimage_pipeline.threshold_variant_runner.copy_cellprofiler_measurements")
@patch("bioimage_pipeline.threshold_variant_runner.run_cellprofiler_pipeline_logged")
def test_run_threshold_variant_artifacts_creates_isolated_output_dirs(
    mock_run_logged: MagicMock,
    mock_copy: MagicMock,
    tmp_path: Path,
) -> None:
    imported_path, input_dir, artifacts = _make_artifacts(tmp_path)

    def run_side_effect(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        log_dir = Path(kwargs["log_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        return _successful_run_result(output_dir, log_dir)

    mock_run_logged.side_effect = run_side_effect
    mock_copy.side_effect = lambda source, destination: [
        Path(destination) / "MyExpt_Image.csv"
    ]

    results = run_threshold_variant_artifacts(
        artifacts,
        input_dir,
        generate_qc=False,
    )

    assert len(results) == 3
    assert mock_run_logged.call_count == 3
    assert imported_path.read_text(encoding="utf-8") == SAMPLE_CPPIPE

    for artifact, result in zip(artifacts, results):
        assert result.success is True
        assert result.variant_dir == artifact.variant_dir.resolve()
        assert result.raw_output_dir == artifact.variant_dir / RESULTS_RAW_DIR
        assert result.measurements_dir == artifact.variant_dir / RESULTS_MEASUREMENTS_DIR
        assert result.qc_dir == artifact.variant_dir / RESULTS_QC_DIR
        assert result.logs_dir == artifact.variant_dir / RESULTS_LOGS_DIR
        assert result.raw_output_dir.exists()
        assert result.measurements_dir.exists()
        assert result.qc_dir.exists()
        assert result.logs_dir.exists()


@patch("bioimage_pipeline.threshold_variant_runner.run_cellprofiler_pipeline_logged")
def test_run_threshold_variant_artifacts_uses_variant_pipeline_paths(
    mock_run_logged: MagicMock,
    tmp_path: Path,
) -> None:
    _, input_dir, artifacts = _make_artifacts(tmp_path)
    mock_run_logged.side_effect = lambda **kwargs: _successful_run_result(
        Path(kwargs["output_dir"]),
        Path(kwargs["log_dir"]),
    )

    run_threshold_variant_artifacts(artifacts, input_dir, generate_qc=False)

    used_pipelines = [
        Path(call.kwargs["cppipe_path"]).resolve()
        for call in mock_run_logged.call_args_list
    ]
    expected_pipelines = [artifact.pipeline_path.resolve() for artifact in artifacts]
    assert used_pipelines == expected_pipelines
    for pipeline_path in used_pipelines:
        assert pipeline_path.name == "pipeline.cppipe"
        assert pipeline_path.parent.name.startswith("variant_")


@patch("bioimage_pipeline.threshold_variant_runner.copy_cellprofiler_measurements")
@patch("bioimage_pipeline.threshold_variant_runner.run_cellprofiler_pipeline_logged")
def test_run_threshold_variant_artifacts_continues_after_single_failure(
    mock_run_logged: MagicMock,
    mock_copy: MagicMock,
    tmp_path: Path,
) -> None:
    _, input_dir, artifacts = _make_artifacts(tmp_path)

    def run_side_effect(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        log_dir = Path(kwargs["log_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        if "variant_002" in str(output_dir):
            return _failed_run_result(output_dir, log_dir)
        return _successful_run_result(output_dir, log_dir)

    mock_run_logged.side_effect = run_side_effect
    mock_copy.return_value = []

    results = run_threshold_variant_artifacts(
        artifacts,
        input_dir,
        generate_qc=False,
    )

    assert len(results) == 3
    assert results[0].success is True
    assert results[1].success is False
    assert results[2].success is True
    assert results[1].error_message is not None
    assert results[1].logs_dir.exists()
    assert (results[1].logs_dir / "cellprofiler_stderr.log").exists()


@patch("bioimage_pipeline.threshold_variant_runner.run_cellprofiler_pipeline_logged")
def test_run_threshold_variant_artifacts_strict_mode_raises(
    mock_run_logged: MagicMock,
    tmp_path: Path,
) -> None:
    _, input_dir, artifacts = _make_artifacts(tmp_path)
    mock_run_logged.return_value = _failed_run_result(
        tmp_path / "raw",
        tmp_path / "logs",
    )

    with pytest.raises(RuntimeError, match="threshold variant CellProfiler runs failed"):
        run_threshold_variant_artifacts(
            artifacts[:1],
            input_dir,
            generate_qc=False,
            strict=True,
        )


def test_run_threshold_variant_artifact_rejects_pipeline_outside_variant_dir(
    tmp_path: Path,
) -> None:
    imported_path = tmp_path / "imported.cppipe"
    imported_path.write_text(SAMPLE_CPPIPE, encoding="utf-8")
    variant_dir = tmp_path / "threshold_variants" / "variant_001_baseline"
    variant_dir.mkdir(parents=True)
    (variant_dir / "pipeline.cppipe").write_text(SAMPLE_CPPIPE, encoding="utf-8")

    artifact = _artifact_spec(variant_dir, pipeline_path=imported_path)
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(ValueError, match="outside variant directory"):
        run_threshold_variant_artifact(artifact, input_dir, generate_qc=False)

    assert imported_path.read_text(encoding="utf-8") == SAMPLE_CPPIPE


@patch("bioimage_pipeline.threshold_variant_runner.run_cellprofiler_pipeline_logged")
def test_run_threshold_variant_artifact_does_not_modify_imported_cppipe(
    mock_run_logged: MagicMock,
    tmp_path: Path,
) -> None:
    imported_path, input_dir, artifacts = _make_artifacts(tmp_path)
    original_text = imported_path.read_text(encoding="utf-8")

    mock_run_logged.side_effect = lambda **kwargs: _successful_run_result(
        Path(kwargs["output_dir"]),
        Path(kwargs["log_dir"]),
    )

    run_threshold_variant_artifacts(artifacts, input_dir, generate_qc=False)

    assert imported_path.read_text(encoding="utf-8") == original_text
