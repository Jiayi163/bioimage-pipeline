"""Run CellProfiler headlessly for generated threshold pipeline variants."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from bioimage_pipeline.cellprofiler_runner import (
    RESULTS_LOGS_DIR,
    RESULTS_MEASUREMENTS_DIR,
    RESULTS_QC_DIR,
    RESULTS_RAW_DIR,
    CellProfilerRunResult,
    copy_cellprofiler_measurements,
    extract_processed_image_names,
    format_cellprofiler_failure,
    load_cellprofiler_measurements_lenient,
    run_cellprofiler_pipeline_logged,
)
from bioimage_pipeline.export import organize_cellprofiler_tiffs_for_fiji
from bioimage_pipeline.qc import generate_qc_for_cellprofiler_results
from bioimage_pipeline.threshold_variants import (
    ThresholdVariantArtifact,
    ThresholdVariantSpec,
)


@dataclass
class ThresholdVariantRunResult:
    """Outcome of running one threshold variant through CellProfiler."""

    spec: ThresholdVariantSpec
    variant_dir: Path
    pipeline_path: Path
    raw_output_dir: Path
    measurements_dir: Path
    qc_dir: Path
    logs_dir: Path
    success: bool
    error_message: str | None = None
    runtime_seconds: float = 0.0
    cellprofiler_run: CellProfilerRunResult | None = None
    measurement_files: list[Path] = field(default_factory=list)
    qc_artifacts: dict[str, dict[str, Path]] = field(default_factory=dict)
    log_files: dict[str, Path] = field(default_factory=dict)


def _ensure_under_directory(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to use path outside variant directory: {resolved}"
        ) from exc
    return resolved


def _variant_run_directories(variant_dir: Path) -> dict[str, Path]:
    root = variant_dir.resolve()
    directories = {
        "raw": root / RESULTS_RAW_DIR,
        "measurements": root / RESULTS_MEASUREMENTS_DIR,
        "qc": root / RESULTS_QC_DIR,
        "logs": root / RESULTS_LOGS_DIR,
    }
    for path in directories.values():
        _ensure_under_directory(path, root)
        path.mkdir(parents=True, exist_ok=True)
    return directories


def _validate_variant_artifact(artifact: ThresholdVariantArtifact) -> None:
    variant_dir = artifact.variant_dir.resolve()
    pipeline_path = _ensure_under_directory(artifact.pipeline_path, variant_dir)
    if not pipeline_path.is_file():
        raise FileNotFoundError(
            f"Variant pipeline file not found: {pipeline_path}"
        )


def _generate_variant_qc(
    input_dir: Path,
    raw_output_dir: Path,
    measurements_dir: Path,
    qc_dir: Path,
) -> dict[str, dict[str, Path]]:
    load_result = load_cellprofiler_measurements_lenient(measurements_dir)
    processed_images = extract_processed_image_names(load_result.tables)
    if not processed_images:
        return {}

    with tempfile.TemporaryDirectory(prefix="threshold_variant_qc_") as staging:
        staging_path = Path(staging)
        masks_dir = staging_path / "masks"
        labels_dir = staging_path / "labels"
        organized = organize_cellprofiler_tiffs_for_fiji(
            raw_output_dir,
            masks_dir,
            labels_dir,
        )
        if not organized.masks and not organized.labels:
            return {}

        return generate_qc_for_cellprofiler_results(
            input_dir,
            masks_dir,
            labels_dir,
            qc_dir,
            processed_images,
        )


def run_threshold_variant_artifact(
    artifact: ThresholdVariantArtifact,
    input_dir: str | Path,
    *,
    cellprofiler_executable: str | None = None,
    generate_qc: bool = True,
    cellprofiler_extra_args: Sequence[str] | None = None,
) -> ThresholdVariantRunResult:
    """Run CellProfiler for one threshold variant artifact."""
    started = time.perf_counter()
    input_path = Path(input_dir).resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    _validate_variant_artifact(artifact)
    directories = _variant_run_directories(artifact.variant_dir)
    executable = cellprofiler_executable or "cellprofiler"

    try:
        run_result = run_cellprofiler_pipeline_logged(
            cppipe_path=artifact.pipeline_path,
            input_dir=input_path,
            output_dir=directories["raw"],
            extra_args=cellprofiler_extra_args,
            cellprofiler_executable=executable,
            log_dir=directories["logs"],
        )
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        return ThresholdVariantRunResult(
            spec=artifact.spec,
            variant_dir=artifact.variant_dir.resolve(),
            pipeline_path=artifact.pipeline_path.resolve(),
            raw_output_dir=directories["raw"].resolve(),
            measurements_dir=directories["measurements"].resolve(),
            qc_dir=directories["qc"].resolve(),
            logs_dir=directories["logs"].resolve(),
            success=False,
            error_message=str(exc),
            runtime_seconds=time.perf_counter() - started,
        )

    if not run_result.succeeded:
        error_message = format_cellprofiler_failure(
            returncode=run_result.returncode,
            stdout=run_result.stdout,
            stderr=run_result.stderr,
            log_files=run_result.log_files,
        )
        return ThresholdVariantRunResult(
            spec=artifact.spec,
            variant_dir=artifact.variant_dir.resolve(),
            pipeline_path=artifact.pipeline_path.resolve(),
            raw_output_dir=directories["raw"].resolve(),
            measurements_dir=directories["measurements"].resolve(),
            qc_dir=directories["qc"].resolve(),
            logs_dir=directories["logs"].resolve(),
            success=False,
            error_message=error_message,
            runtime_seconds=time.perf_counter() - started,
            cellprofiler_run=run_result,
            log_files=dict(run_result.log_files),
        )

    measurement_files = copy_cellprofiler_measurements(
        directories["raw"],
        directories["measurements"],
    )

    qc_artifacts: dict[str, dict[str, Path]] = {}
    if generate_qc:
        try:
            qc_artifacts = _generate_variant_qc(
                input_path,
                directories["raw"],
                directories["measurements"],
                directories["qc"],
            )
        except (FileNotFoundError, OSError, ValueError):
            qc_artifacts = {}

    return ThresholdVariantRunResult(
        spec=artifact.spec,
        variant_dir=artifact.variant_dir.resolve(),
        pipeline_path=artifact.pipeline_path.resolve(),
        raw_output_dir=directories["raw"].resolve(),
        measurements_dir=directories["measurements"].resolve(),
        qc_dir=directories["qc"].resolve(),
        logs_dir=directories["logs"].resolve(),
        success=True,
        runtime_seconds=time.perf_counter() - started,
        cellprofiler_run=run_result,
        measurement_files=measurement_files,
        qc_artifacts=qc_artifacts,
        log_files=dict(run_result.log_files),
    )


def run_threshold_variant_artifacts(
    artifacts: Iterable[ThresholdVariantArtifact],
    input_dir: str | Path,
    *,
    cellprofiler_executable: str | None = None,
    generate_qc: bool = True,
    cellprofiler_extra_args: Sequence[str] | None = None,
    strict: bool = False,
) -> list[ThresholdVariantRunResult]:
    """Run CellProfiler for each threshold variant artifact in isolation."""
    results: list[ThresholdVariantRunResult] = []
    failures: list[ThresholdVariantRunResult] = []

    for artifact in artifacts:
        result = run_threshold_variant_artifact(
            artifact,
            input_dir,
            cellprofiler_executable=cellprofiler_executable,
            generate_qc=generate_qc,
            cellprofiler_extra_args=cellprofiler_extra_args,
        )
        results.append(result)
        if not result.success:
            failures.append(result)
            if strict:
                break

    if strict and failures:
        summary = "; ".join(
            f"{result.spec.variant_id}: {result.error_message or 'failed'}"
            for result in failures
        )
        raise RuntimeError(
            "One or more threshold variant CellProfiler runs failed: " + summary
        )

    return results
