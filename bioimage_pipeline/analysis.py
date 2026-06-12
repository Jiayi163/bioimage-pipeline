"""Unified analysis entry point for Python and CellProfiler engines."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd

from bioimage_pipeline.batch import run_pipeline_on_folder
from bioimage_pipeline.cellprofiler_runner import (
    RESULTS_LABELS_DIR,
    RESULTS_LOGS_DIR,
    RESULTS_MASKS_DIR,
    RESULTS_MEASUREMENTS_DIR,
    RESULTS_QC_DIR,
    RESULTS_RAW_DIR,
    CellProfilerRunResult,
    format_cellprofiler_failure,
    copy_cellprofiler_measurements,
    extract_processed_image_names,
    inspect_cellprofiler_logs,
    load_cellprofiler_measurements,
    load_cellprofiler_measurements_lenient,
    merge_cellprofiler_tables,
    run_cellprofiler_pipeline,
    run_cellprofiler_pipeline_logged,
    summarize_cellprofiler_tables,
)
from bioimage_pipeline.export import (
    export_measurements_csv,
    organize_cellprofiler_tiffs_for_fiji,
)
from bioimage_pipeline.fiji_runner import (
    FijiExportResult,
    format_fiji_error_summary,
    run_fiji_batch_export,
)
from bioimage_pipeline.workflow_timing import (
    elapsed_since,
    finalize_workflow_timing,
    init_workflow_timing,
    log_timing_breakdown,
)
from bioimage_pipeline.oir_projection_lifecycle import (
    AUDIT_LOG,
    OirProjectionLifecycleRecorder,
)

AnalysisEngine = Literal["python", "cellprofiler"]
LabelingMethod = Literal["connected", "watershed"]
_SUPPORTED_ENGINES = frozenset({"python", "cellprofiler"})
_SUPPORTED_LABELING = frozenset({"connected", "watershed"})

RESULTS_STAGING_DIR = "staging"
RESULTS_OIR_PROJECTION_DIR = "oir_projection"


def resolve_workflow_output_dir(output_dir: str | Path) -> Path:
    """Return a stable absolute results directory for workflow outputs."""
    path = Path(output_dir).expanduser()
    if not str(path).strip():
        raise ValueError("Output directory is required.")
    resolved = path.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def generate_qc_for_cellprofiler_results(*args: Any, **kwargs: Any) -> Any:
    """Lazy proxy for QC generation to keep CellProfiler/Fiji imports lightweight."""
    from bioimage_pipeline.qc import generate_qc_for_cellprofiler_results as _generate

    return _generate(*args, **kwargs)


def _validate_labeling_method(labeling_method: str) -> LabelingMethod:
    if labeling_method not in _SUPPORTED_LABELING:
        supported = ", ".join(sorted(_SUPPORTED_LABELING))
        raise ValueError(
            f"Unsupported labeling_method: {labeling_method!r}. "
            f"Choose one of: {supported}."
        )
    return labeling_method  # type: ignore[return-value]


def build_default_pipeline(
    *,
    blur_sigma: float = 1.0,
    min_object_size: int = 20,
    labeling_method: LabelingMethod = "connected",
    clean_mask_before_labeling: bool = False,
    watershed_min_distance: int = 8,
    watershed_min_peak_ratio: float = 0.5,
) -> Pipeline:
    """Build the standard lightweight Python analysis pipeline.

    Steps: Gaussian blur → Otsu threshold → small-object removal →
    optional morphological cleanup → labeling → region measurement.

    Use ``labeling_method="watershed"`` to split touching objects (Phase 11).

    For the experimental self-adaptive import pipeline (Phase 17, deferred),
    call :func:`bioimage_pipeline.adaptive_import.run_self_adaptive_threshold`
    directly or use ``run_cellprofiler_workflow(..., adaptive_threshold=True)``.
    """
    from bioimage_pipeline.measure import measure_objects
    from bioimage_pipeline.pipeline import Pipeline
    from bioimage_pipeline.preprocess import gaussian_blur
    from bioimage_pipeline.segment import (
        clean_mask,
        label_objects,
        remove_small_objects_from_mask,
        split_touching_objects,
    )
    from bioimage_pipeline.threshold import otsu_threshold

    validated_labeling = _validate_labeling_method(labeling_method)

    def blur_step(data: dict[str, Any]) -> dict[str, Any]:
        data["processed"] = gaussian_blur(data["image"], sigma=blur_sigma)
        return data

    def threshold_step(data: dict[str, Any]) -> dict[str, Any]:
        data["mask"] = otsu_threshold(data["processed"])
        return data

    def clean_step(data: dict[str, Any]) -> dict[str, Any]:
        data["mask"] = remove_small_objects_from_mask(
            data["mask"],
            min_size=min_object_size,
        )
        if clean_mask_before_labeling:
            data["mask"] = clean_mask(data["mask"], min_size=min_object_size)
        return data

    def label_step(data: dict[str, Any]) -> dict[str, Any]:
        if validated_labeling == "watershed":
            data["labels"] = split_touching_objects(
                data["mask"],
                min_distance=watershed_min_distance,
                min_peak_ratio=watershed_min_peak_ratio,
            )
        else:
            data["labels"] = label_objects(data["mask"])
        return data

    def measure_step(data: dict[str, Any]) -> dict[str, Any]:
        data["measurements"] = measure_objects(
            data["labels"],
            intensity_image=data["image"],
        )
        return data

    return Pipeline(
        [blur_step, threshold_step, clean_step, label_step, measure_step]
    )


@dataclass
class CellProfilerWorkflowConfig:
    """Configuration for :func:`run_cellprofiler_workflow`."""

    input_dir: str | Path
    output_dir: str | Path
    cppipe_path: str | Path
    cellprofiler_executable: str = "cellprofiler"
    cellprofiler_extra_args: Sequence[str] | None = None
    merge_measurements: bool = True
    export_fiji_tiffs: bool = True
    generate_qc: bool = True
    fiji_image_pattern: str = "*.tif"
    fiji_executable: str | Path | None = None
    fiji_macro_path: str | Path | None = None
    fiji_headless: bool | None = None
    fiji_timeout: float | None = None
    fiji_fallback_to_python: bool = True
    oir_projection_engine: str | None = None
    force_oir_reproject: bool = False
    adaptive_threshold: bool = False
    adaptive_min_object_size: int = 20
    adaptive_image_pattern: str = "*.tif"


@dataclass
class CellProfilerWorkflowResult:
    """Summary of a completed CellProfiler-to-Fiji workflow run."""

    results_dir: Path
    raw_output_dir: Path
    measurements_dir: Path
    masks_dir: Path
    labels_dir: Path
    qc_dir: Path
    logs_dir: Path
    processed_images: list[str]
    tables: dict[str, pd.DataFrame]
    table_summary: dict[str, dict[str, int]]
    measurements: pd.DataFrame | None
    mask_exports: list[Path]
    label_exports: list[Path]
    qc_artifacts: dict[str, dict[str, Path]]
    log_files: dict[str, Path]
    cellprofiler_run: CellProfilerRunResult
    timing: dict[str, float] | None = None
    export_engine: str | None = None
    export_mode: str | None = None
    fiji_export_result: FijiExportResult | None = None
    export_warnings: list[str] | None = None
    adaptive_threshold_summary: dict[str, Any] | None = None
    import_warnings: list[str] | None = None

    @property
    def output_dir(self) -> Path:
        """Backward-compatible alias for :attr:`results_dir`."""
        return self.results_dir

    @property
    def fiji_exports(self) -> list[Path]:
        """Backward-compatible list of organized Fiji-compatible TIFF exports."""
        return [*self.mask_exports, *self.label_exports]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the workflow result for logging or JSON output."""
        return {
            "results_dir": str(self.results_dir),
            "raw_output_dir": str(self.raw_output_dir),
            "measurements_dir": str(self.measurements_dir),
            "masks_dir": str(self.masks_dir),
            "labels_dir": str(self.labels_dir),
            "qc_dir": str(self.qc_dir),
            "logs_dir": str(self.logs_dir),
            "processed_images": list(self.processed_images),
            "tables": sorted(self.tables),
            "table_summary": self.table_summary,
            "measurements_rows": (
                len(self.measurements) if self.measurements is not None else 0
            ),
            "mask_exports": [str(path) for path in self.mask_exports],
            "label_exports": [str(path) for path in self.label_exports],
            "qc_artifacts": {
                filename: {key: str(path) for key, path in artifact_map.items()}
                for filename, artifact_map in self.qc_artifacts.items()
            },
            "log_files": {key: str(path) for key, path in self.log_files.items()},
            "cellprofiler_returncode": self.cellprofiler_run.returncode,
            "timing": dict(self.timing or {}),
            "export_engine": self.export_engine,
            "export_mode": self.export_mode,
            "fiji_export": (
                self.fiji_export_result.to_dict()
                if self.fiji_export_result is not None
                else None
            ),
            "export_warnings": list(self.export_warnings or []),
            "adaptive_threshold_summary": self.adaptive_threshold_summary,
            "import_warnings": list(self.import_warnings or []),
        }


@dataclass
class AnalysisConfig:
    """Configuration for :func:`run_analysis`."""

    input_dir: str | Path
    output_dir: str | Path
    analysis_engine: AnalysisEngine = "python"
    pattern: str = "*.tif"
    pipeline: Pipeline | None = None
    blur_sigma: float = 1.0
    min_object_size: int = 20
    labeling_method: LabelingMethod = "connected"
    clean_mask_before_labeling: bool = False
    watershed_min_distance: int = 8
    watershed_min_peak_ratio: float = 0.5
    cppipe_path: str | Path | None = None
    cellprofiler_executable: str = "cellprofiler"
    cellprofiler_extra_args: Sequence[str] | None = None
    merge_measurements: bool = True


def _validate_engine(analysis_engine: str) -> AnalysisEngine:
    if analysis_engine not in _SUPPORTED_ENGINES:
        supported = ", ".join(sorted(_SUPPORTED_ENGINES))
        raise ValueError(
            f"Unsupported analysis_engine: {analysis_engine!r}. "
            f"Choose one of: {supported}."
        )
    return analysis_engine  # type: ignore[return-value]


def _run_python_analysis(config: AnalysisConfig) -> dict[str, Any]:
    pipeline = config.pipeline or build_default_pipeline(
        blur_sigma=config.blur_sigma,
        min_object_size=config.min_object_size,
        labeling_method=config.labeling_method,
        clean_mask_before_labeling=config.clean_mask_before_labeling,
        watershed_min_distance=config.watershed_min_distance,
        watershed_min_peak_ratio=config.watershed_min_peak_ratio,
    )
    batch_result = run_pipeline_on_folder(
        pipeline,
        config.input_dir,
        config.output_dir,
        pattern=config.pattern,
    )
    return {
        "analysis_engine": "python",
        "output_dir": Path(config.output_dir).resolve(),
        "processed": batch_result["processed"],
        "failed": batch_result["failed"],
        "tables": None,
        "measurements": None,
    }


def _run_cellprofiler_analysis(config: AnalysisConfig) -> dict[str, Any]:
    if config.cppipe_path is None:
        raise ValueError(
            "cppipe_path is required when analysis_engine is 'cellprofiler'."
        )

    output_dir = run_cellprofiler_pipeline(
        cppipe_path=config.cppipe_path,
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        extra_args=config.cellprofiler_extra_args,
        cellprofiler_executable=config.cellprofiler_executable,
    )
    load_result = load_cellprofiler_measurements(output_dir)
    measurements = None
    import_warnings = list(load_result.warnings)
    if config.merge_measurements:
        measurements, merge_warnings = merge_cellprofiler_tables(
            load_result.tables,
            metadata=load_result.metadata,
        )
        import_warnings.extend(merge_warnings)

    return {
        "analysis_engine": "cellprofiler",
        "output_dir": output_dir,
        "processed": None,
        "failed": None,
        "tables": load_result.tables,
        "measurements": measurements,
        "import_warnings": import_warnings,
    }


def run_analysis(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    analysis_engine: AnalysisEngine = "python",
    pattern: str = "*.tif",
    pipeline: Pipeline | None = None,
    blur_sigma: float = 1.0,
    min_object_size: int = 20,
    labeling_method: LabelingMethod = "connected",
    clean_mask_before_labeling: bool = False,
    watershed_min_distance: int = 8,
    watershed_min_peak_ratio: float = 0.5,
    cppipe_path: str | Path | None = None,
    cellprofiler_executable: str = "cellprofiler",
    cellprofiler_extra_args: Sequence[str] | None = None,
    merge_measurements: bool = True,
) -> dict[str, Any]:
    """Run image analysis with the Python or CellProfiler engine.

    Args:
        input_dir: Folder containing input images.
        output_dir: Folder where outputs are written.
        analysis_engine: ``"python"`` for the built-in pipeline, or
            ``"cellprofiler"`` to run a ``.cppipe`` file headlessly.
        pattern: Glob pattern for input TIFFs (Python mode only).
        pipeline: Optional custom :class:`~bioimage_pipeline.pipeline.Pipeline`.
            When omitted in Python mode, :func:`build_default_pipeline` is used.
        blur_sigma: Gaussian blur sigma for the default Python pipeline.
        min_object_size: Minimum object size for the default Python pipeline.
        labeling_method: ``"connected"`` or ``"watershed"`` for the default
            Python pipeline label step.
        clean_mask_before_labeling: Apply morphological mask cleanup before
            labeling in the default Python pipeline.
        watershed_min_distance: Seed spacing when ``labeling_method`` is
            ``"watershed"``.
        watershed_min_peak_ratio: Seed threshold ratio when ``labeling_method``
            is ``"watershed"``.
        cppipe_path: CellProfiler pipeline file (required for CellProfiler mode).
        cellprofiler_executable: CellProfiler command or executable path.
        cellprofiler_extra_args: Extra CLI flags forwarded to CellProfiler.
        merge_measurements: When ``True``, merge CellProfiler CSV tables into
            one DataFrame in the result ``measurements`` field.

    Returns:
        Result dictionary. Common keys: ``analysis_engine``, ``output_dir``.
        Python mode also includes ``processed`` and ``failed``.
        CellProfiler mode also includes ``tables`` and optionally
        ``measurements``.

    Raises:
        ValueError: If ``analysis_engine`` is unsupported or required options
            are missing for the selected engine.
    """
    config = AnalysisConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        analysis_engine=_validate_engine(analysis_engine),
        pattern=pattern,
        pipeline=pipeline,
        blur_sigma=blur_sigma,
        min_object_size=min_object_size,
        labeling_method=_validate_labeling_method(labeling_method),
        clean_mask_before_labeling=clean_mask_before_labeling,
        watershed_min_distance=watershed_min_distance,
        watershed_min_peak_ratio=watershed_min_peak_ratio,
        cppipe_path=cppipe_path,
        cellprofiler_executable=cellprofiler_executable,
        cellprofiler_extra_args=cellprofiler_extra_args,
        merge_measurements=merge_measurements,
    )

    if config.analysis_engine == "python":
        return _run_python_analysis(config)
    return _run_cellprofiler_analysis(config)


def _format_oir_projection_failure(failed: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {item.get('file', 'unknown')}: {item.get('error', 'unknown error')}"
        for item in failed
    )


def _attach_oir_projection_lifecycle_logs(
    log_files: dict[str, Path],
    lifecycle: OirProjectionLifecycleRecorder | None,
) -> None:
    if lifecycle is None:
        return
    lifecycle.record_stage("workflow_end")
    log_files["oir_projection_lifecycle"] = lifecycle.lifecycle_path
    log_files["oir_projection_audit"] = lifecycle.logs_dir / AUDIT_LOG


def _prepare_cellprofiler_input_dir(
    input_dir: Path,
    *,
    results_dir: Path,
    logs_dir: Path,
    oir_projection_engine: str | None = None,
    fiji_executable: str | Path | None = None,
    fiji_headless: bool | None = None,
    fiji_timeout: float | None = None,
    force_oir_reproject: bool = False,
    lifecycle: OirProjectionLifecycleRecorder | None = None,
) -> tuple[Path, Path | None]:
    """Project ``.oir`` stacks to TIFF when needed and return the CP input folder."""
    import time

    from bioimage_pipeline.oir_zmax_batch import (
        OirZmaxEngine,
        run_oir_zmax_batch,
    )
    from bioimage_pipeline.prepare_input_profile import (
        PrepareInputFileRecord,
        PrepareInputProfile,
        build_investigation_notes,
        detect_file_type,
        scan_prepare_input_folder,
        write_prepare_input_profile,
    )

    source_dir = input_dir.resolve()
    scan = scan_prepare_input_folder(source_dir)

    if not scan.oir_files:
        profile = PrepareInputProfile(
            input_dir=str(source_dir),
            action="passthrough_tiff",
            scan=scan,
            projection_output_dir=None,
            file_records=[
                PrepareInputFileRecord(
                    input_path=str(path),
                    detected_type=detect_file_type(path),
                    output_path=str(path),
                    input_bytes=path.stat().st_size if path.is_file() else 0,
                    notes=["Top-level TIFF passed through without prepare_input conversion."],
                )
                for path in scan.tiff_files
            ],
        )
        profile.investigation_notes = build_investigation_notes(profile)
        profile_path, _ = write_prepare_input_profile(logs_dir, profile)
        return source_dir, profile_path

    projection_dir = results_dir / RESULTS_OIR_PROJECTION_DIR
    if lifecycle is not None:
        lifecycle.record_stage("prepare_input_entry")
    engine: OirZmaxEngine = (
        oir_projection_engine  # type: ignore[assignment]
        if oir_projection_engine in {"python", "fiji", "auto"}
        else "auto"
    )
    projection_started = time.perf_counter()
    result = run_oir_zmax_batch(
        source_dir,
        projection_dir,
        engine=engine,
        logs_dir=logs_dir,
        fiji_executable=fiji_executable,
        fiji_headless=fiji_headless,
        fiji_timeout=fiji_timeout,
        force_oir_reproject=force_oir_reproject,
        lifecycle=lifecycle,
    )
    projection_seconds = time.perf_counter() - projection_started

    if result.cache_hits and not result.reprojected:
        action = "oir_projection_cache_hit"
    else:
        action = f"oir_projection_{result.engine}"

    profile = PrepareInputProfile(
        input_dir=str(source_dir),
        action=action,
        scan=scan,
        engine=result.engine,
        projection_output_dir=str(projection_dir.resolve()),
        projection_seconds=projection_seconds,
        file_records=list(result.file_profiles),
    )
    profile.investigation_notes = build_investigation_notes(profile)
    profile_path, _ = write_prepare_input_profile(logs_dir, profile)

    summary_path = logs_dir / "oir_projection_summary.json"
    summary_payload: dict[str, Any] = {
        "input_dir": str(result.input_dir),
        "output_dir": str(result.output_dir),
        "engine": result.engine,
        "input_oir_files": [str(pair.input_oir) for pair in result.file_pairs],
        "output_tif_paths": [str(pair.output_tif) for pair in result.file_pairs],
        "processed": list(result.processed),
        "failed": list(result.failed),
        "files_created_in_output_dir": list(result.files_created),
        "remapped_outputs": list(result.remapped_outputs),
        "prepare_input_profile": str(profile_path),
        "projection_seconds": projection_seconds,
        "scan_seconds": scan.scan_seconds,
        "directories_scanned": scan.directories_scanned,
        "tiff_files_found": scan.tiff_count,
        "cache_hits": list(result.cache_hits),
        "reprojected": list(result.reprojected),
        "force_oir_reproject": result.force_oir_reproject,
        "oir_projection_cache_debug": str(
            logs_dir / "oir_projection_cache_debug.txt"
        ),
        "file_pairs": [
            {
                "input_oir": str(pair.input_oir),
                "output_tif": str(pair.output_tif),
            }
            for pair in result.file_pairs
        ],
        "file_profiles": [
            {
                "input_path": record.input_path,
                "detected_type": record.detected_type,
                "output_path": record.output_path,
                "input_bytes": record.input_bytes,
                "output_bytes": record.output_bytes,
                "read_seconds": record.read_seconds,
                "conversion_seconds": record.conversion_seconds,
                "write_seconds": record.write_seconds,
                "total_seconds": record.total_seconds,
                "output_existed_before_run": record.output_existed_before_run,
                "notes": list(record.notes),
            }
            for record in profile.file_records
        ],
    }
    if result.fiji_executable is not None:
        summary_payload["fiji_executable"] = str(result.fiji_executable)
    if result.fiji_headless is not None:
        summary_payload["fiji_headless"] = result.fiji_headless
    if result.fiji_returncode is not None:
        summary_payload["fiji_returncode"] = result.fiji_returncode
    if result.generated_macro_path is not None:
        summary_payload["generated_macro"] = str(result.generated_macro_path)
    if result.fiji_log_files:
        summary_payload["fiji_log_files"] = {
            key: str(path) for key, path in result.fiji_log_files.items()
        }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    if result.failed:
        raise RuntimeError(
            "OIR Z-max projection failed for one or more files:\n"
            + _format_oir_projection_failure(result.failed)
        )
    if not result.processed:
        raise RuntimeError(
            "Input folder contains .oir files but OIR Z-max produced no projected "
            "TIFF files. Install aicsimageio/bfio or switch OIR projection engine "
            "to Fiji."
        )
    return projection_dir.resolve(), summary_path


def _prepare_workflow_directories(results_dir: Path) -> dict[str, Path]:
    directories = {
        "results": results_dir,
        "raw": results_dir / RESULTS_RAW_DIR,
        "measurements": results_dir / RESULTS_MEASUREMENTS_DIR,
        "masks": results_dir / RESULTS_MASKS_DIR,
        "labels": results_dir / RESULTS_LABELS_DIR,
        "qc": results_dir / RESULTS_QC_DIR,
        "logs": results_dir / RESULTS_LOGS_DIR,
        "oir_projection": results_dir / RESULTS_OIR_PROJECTION_DIR,
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def _write_workflow_summary(
    logs_dir: Path,
    result: CellProfilerWorkflowResult,
) -> Path:
    summary_path = logs_dir / "workflow_summary.json"
    summary_path.write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )
    return summary_path


def _validate_cellprofiler_workflow_config(config: CellProfilerWorkflowConfig) -> None:
    """Validate workflow paths and external executables before running."""
    from bioimage_pipeline.cellprofiler_runner import (
        cellprofiler_not_found_message,
        find_cellprofiler_executable,
    )
    from bioimage_pipeline.fiji_runner import find_fiji_executable

    input_path = Path(config.input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    cppipe_path = Path(config.cppipe_path)
    if not cppipe_path.is_file():
        raise FileNotFoundError(f"CellProfiler pipeline file not found: {cppipe_path}")

    if find_cellprofiler_executable(config.cellprofiler_executable) is None:
        raise RuntimeError(
            f"{cellprofiler_not_found_message()} "
            f"Tried: {config.cellprofiler_executable!r}."
        )

    if config.fiji_executable and config.export_fiji_tiffs:
        if find_fiji_executable(config.fiji_executable) is None:
            raise FileNotFoundError(
                f"Fiji executable not found: {config.fiji_executable}"
            )

    if config.fiji_macro_path and config.export_fiji_tiffs:
        macro_path = Path(config.fiji_macro_path)
        if not macro_path.is_file():
            raise FileNotFoundError(f"Fiji macro not found: {macro_path}")


def _materialize_pipeline_for_run(cppipe_path: str | Path) -> Path:
    """Load and validate the CellProfiler pipeline before headless execution."""
    path = Path(cppipe_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CellProfiler pipeline file not found: {path}")

    try:
        from bioimage_pipeline.cppipe_io import (
            load_and_validate_imported_pipeline,
            prepare_pipeline_for_cellprofiler,
        )

        pipeline = load_and_validate_imported_pipeline(path)
        prepare_pipeline_for_cellprofiler(pipeline)
    except ValueError:
        # Headless runs may use pipelines that were authored outside the GUI
        # builder; CellProfiler validates module content at execution time.
        pass
    return path


def _finalize_and_log_timing(
    timing: dict[str, float],
    total_started: float,
    logs_dir: Path | None = None,
) -> dict[str, float]:
    """Finalize timing aggregates and print the breakdown."""
    finalize_workflow_timing(timing, total_started)
    log_timing_breakdown(timing, logs_dir=logs_dir)
    return timing


def run_cellprofiler_workflow(
    input_dir: str | Path,
    output_dir: str | Path,
    cppipe_path: str | Path,
    *,
    cellprofiler_executable: str = "cellprofiler",
    cellprofiler_extra_args: Sequence[str] | None = None,
    merge_measurements: bool = True,
    export_fiji_tiffs: bool = True,
    generate_qc: bool = True,
    fiji_image_pattern: str = "*.tif",
    fiji_executable: str | Path | None = None,
    fiji_macro_path: str | Path | None = None,
    fiji_headless: bool | None = None,
    fiji_timeout: float | None = None,
    fiji_fallback_to_python: bool = True,
    oir_projection_engine: str | None = None,
    force_oir_reproject: bool = False,
    adaptive_threshold: bool = False,
    adaptive_min_object_size: int = 20,
    adaptive_image_pattern: str = "*.tif",
) -> CellProfilerWorkflowResult:
    """Run the CellProfiler-to-Fiji workflow and organize all outputs.

    Steps: optional self-adaptive Python threshold staging → headless
    CellProfiler run → capture logs → collect CSV/TIFF outputs → write
    Fiji-compatible masks and labels → generate QC overlays.

    Args:
        input_dir: Folder containing input images for CellProfiler.
        output_dir: Root folder for organized results (``measurements/``,
            ``masks/``, ``labels/``, ``qc/``, ``logs/``, ``cellprofiler_raw/``).
        cppipe_path: CellProfiler pipeline file.
        cellprofiler_executable: CellProfiler command or executable path.
        cellprofiler_extra_args: Extra CLI flags forwarded to CellProfiler.
        merge_measurements: Merge CSV tables into one DataFrame.
        export_fiji_tiffs: Convert discovered TIFF outputs into Fiji-friendly
            files under ``masks/`` and ``labels/``.
        generate_qc: Create mask/label overlay figures under ``qc/``.
        fiji_image_pattern: Glob pattern for TIFF discovery in raw CP output.
        fiji_executable: Optional Fiji/ImageJ executable path. When omitted,
            common paths and ``FIJI_EXECUTABLE`` are checked.
        fiji_macro_path: Optional batch folder macro. Defaults to the bundled
            ``examples/fiji_macros/export_folder.ijm``.
        fiji_headless: Override platform default headless mode.
        fiji_timeout: Optional Fiji subprocess timeout in seconds.
        fiji_fallback_to_python: Fall back to in-process Python TIFF export when
            Fiji is unavailable or the batch macro fails.
        adaptive_threshold: When ``True``, run experimental Phase 17 self-adaptive
            Python thresholding into ``staging/`` before CellProfiler (opt-in;
            prototype — see DEVELOPMENT_PLAN.md Phase 17).
        adaptive_min_object_size: Minimum object size for adaptive threshold cleanup.
        adaptive_image_pattern: Glob pattern for adaptive threshold input TIFFs.

    Returns:
        :class:`CellProfilerWorkflowResult` with organized paths and summaries.
    """
    config = CellProfilerWorkflowConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        cppipe_path=cppipe_path,
        cellprofiler_executable=cellprofiler_executable,
        cellprofiler_extra_args=cellprofiler_extra_args,
        merge_measurements=merge_measurements,
        export_fiji_tiffs=export_fiji_tiffs,
        generate_qc=generate_qc,
        fiji_image_pattern=fiji_image_pattern,
        fiji_executable=fiji_executable,
        fiji_macro_path=fiji_macro_path,
        fiji_headless=fiji_headless,
        fiji_timeout=fiji_timeout,
        fiji_fallback_to_python=fiji_fallback_to_python,
        oir_projection_engine=oir_projection_engine,
        force_oir_reproject=force_oir_reproject,
        adaptive_threshold=adaptive_threshold,
        adaptive_min_object_size=adaptive_min_object_size,
        adaptive_image_pattern=adaptive_image_pattern,
    )
    return run_cellprofiler_workflow_from_config(config)


def run_cellprofiler_workflow_from_config(
    config: CellProfilerWorkflowConfig,
) -> CellProfilerWorkflowResult:
    """Run :func:`run_cellprofiler_workflow` from a config object."""
    total_started = time.perf_counter()
    timing = init_workflow_timing()
    directories: dict[str, Path] | None = None
    oir_projection_log: Path | None = None
    materialized_cppipe: Path | None = None

    checkpoint = time.perf_counter()
    _validate_cellprofiler_workflow_config(config)
    timing["config_validation_seconds"] = elapsed_since(checkpoint)

    checkpoint = time.perf_counter()
    materialized_cppipe = _materialize_pipeline_for_run(config.cppipe_path)
    timing["pipeline_materialization_seconds"] = elapsed_since(checkpoint)

    checkpoint = time.perf_counter()
    results_dir = resolve_workflow_output_dir(config.output_dir)
    lifecycle = OirProjectionLifecycleRecorder(
        logs_dir=results_dir / RESULTS_LOGS_DIR,
        results_dir=results_dir,
        projection_dir=results_dir / RESULTS_OIR_PROJECTION_DIR,
    )
    lifecycle.record_workflow_start()
    directories = _prepare_workflow_directories(results_dir)
    lifecycle.record_stage("setup_directories")
    timing["setup_directories_seconds"] = elapsed_since(checkpoint)

    checkpoint = time.perf_counter()
    cellprofiler_input_dir, oir_projection_log = _prepare_cellprofiler_input_dir(
        Path(config.input_dir),
        results_dir=directories["results"],
        logs_dir=directories["logs"],
        oir_projection_engine=config.oir_projection_engine,
        fiji_executable=config.fiji_executable,
        fiji_headless=config.fiji_headless,
        fiji_timeout=config.fiji_timeout,
        force_oir_reproject=config.force_oir_reproject,
        lifecycle=lifecycle,
    )
    timing["prepare_input_seconds"] = elapsed_since(checkpoint)

    adaptive_summary: dict[str, Any] | None = None
    if config.adaptive_threshold:
        from bioimage_pipeline.adaptive_import import (
            run_self_adaptive_threshold_on_folder,
        )

        checkpoint = time.perf_counter()
        staging_dir = directories["results"] / RESULTS_STAGING_DIR
        adaptive_summary = run_self_adaptive_threshold_on_folder(
            config.input_dir,
            staging_dir,
            pattern=config.adaptive_image_pattern,
            min_object_size=config.adaptive_min_object_size,
            logs_dir=directories["logs"],
        )
        for mask_path in (staging_dir / "masks").glob("*.tif"):
            shutil.copy2(mask_path, directories["masks"] / mask_path.name)
        for label_path in (staging_dir / "labels").glob("*.tif"):
            shutil.copy2(label_path, directories["labels"] / label_path.name)
        timing["adaptive_threshold_seconds"] = elapsed_since(checkpoint)

    run_result = run_cellprofiler_pipeline_logged(
        cppipe_path=materialized_cppipe,
        input_dir=cellprofiler_input_dir,
        output_dir=directories["raw"],
        extra_args=config.cellprofiler_extra_args,
        cellprofiler_executable=config.cellprofiler_executable,
        log_dir=directories["logs"],
    )
    timing["cellprofiler_startup_seconds"] = run_result.startup_seconds
    timing["cellprofiler_subprocess_seconds"] = run_result.subprocess_seconds
    if not run_result.succeeded:
        _finalize_and_log_timing(timing, total_started, directories["logs"])
        failure_log_files = dict(run_result.log_files)
        if oir_projection_log is not None:
            failure_log_files["oir_projection"] = oir_projection_log
        _attach_oir_projection_lifecycle_logs(failure_log_files, lifecycle)
        _write_workflow_summary(
            directories["logs"],
            CellProfilerWorkflowResult(
                results_dir=directories["results"].resolve(),
                raw_output_dir=directories["raw"].resolve(),
                measurements_dir=directories["measurements"].resolve(),
                masks_dir=directories["masks"].resolve(),
                labels_dir=directories["labels"].resolve(),
                qc_dir=directories["qc"].resolve(),
                logs_dir=directories["logs"].resolve(),
                processed_images=[],
                tables={},
                table_summary={},
                measurements=None,
                mask_exports=[],
                label_exports=[],
                qc_artifacts={},
                log_files=failure_log_files,
                cellprofiler_run=run_result,
                timing=timing,
                export_engine=None,
                export_mode=None,
            ),
        )
        raise RuntimeError(
            "CellProfiler command failed: "
            + format_cellprofiler_failure(
                returncode=run_result.returncode,
                stdout=run_result.stdout,
                stderr=run_result.stderr,
                log_files=run_result.log_files,
            )
        )

    checkpoint = time.perf_counter()
    copy_cellprofiler_measurements(
        directories["raw"],
        directories["measurements"],
    )
    timing["copy_measurements_seconds"] = elapsed_since(checkpoint)

    checkpoint = time.perf_counter()
    log_errors = inspect_cellprofiler_logs(
        log_dir=directories["logs"],
        log_files=run_result.log_files,
        stdout=run_result.stdout,
        stderr=run_result.stderr,
    )
    timing["inspect_cp_logs_seconds"] = elapsed_since(checkpoint)
    if log_errors:
        _finalize_and_log_timing(timing, total_started, directories["logs"])
        failure_log_files = dict(run_result.log_files)
        if oir_projection_log is not None:
            failure_log_files["oir_projection"] = oir_projection_log
        _attach_oir_projection_lifecycle_logs(failure_log_files, lifecycle)
        _write_workflow_summary(
            directories["logs"],
            CellProfilerWorkflowResult(
                results_dir=directories["results"].resolve(),
                raw_output_dir=directories["raw"].resolve(),
                measurements_dir=directories["measurements"].resolve(),
                masks_dir=directories["masks"].resolve(),
                labels_dir=directories["labels"].resolve(),
                qc_dir=directories["qc"].resolve(),
                logs_dir=directories["logs"].resolve(),
                processed_images=[],
                tables={},
                table_summary={},
                measurements=None,
                mask_exports=[],
                label_exports=[],
                qc_artifacts={},
                log_files=failure_log_files,
                cellprofiler_run=run_result,
                timing=timing,
                export_engine=None,
                export_mode=None,
            ),
        )
        raise RuntimeError(
            "CellProfiler pipeline failed: "
            + format_cellprofiler_failure(
                returncode=run_result.returncode,
                stdout=run_result.stdout,
                stderr="\n".join(log_errors),
                log_files=run_result.log_files,
            )
        )

    checkpoint = time.perf_counter()
    load_result = load_cellprofiler_measurements_lenient(
        directories["measurements"],
    )
    tables = load_result.tables
    import_warnings = list(load_result.warnings)
    timing["load_measurements_seconds"] = elapsed_since(checkpoint)

    measurements = None
    checkpoint = time.perf_counter()
    if config.merge_measurements:
        measurements, merge_warnings = merge_cellprofiler_tables(
            tables,
            metadata=load_result.metadata,
        )
        import_warnings.extend(merge_warnings)
    if measurements is not None:
        export_measurements_csv(
            directories["measurements"] / "merged_measurements.csv",
            measurements,
        )
    timing["csv_merge_export_seconds"] = elapsed_since(checkpoint)

    processed_images = extract_processed_image_names(tables)

    mask_exports: list[Path] = []
    label_exports: list[Path] = []
    fiji_export_result: FijiExportResult | None = None
    export_engine: str | None = None
    export_mode: str | None = None
    export_warnings: list[str] = []
    export_warning_log: Path | None = None
    if config.export_fiji_tiffs:
        if config.adaptive_threshold:
            mask_exports = sorted(directories["masks"].glob("*.tif"))
            label_exports = sorted(directories["labels"].glob("*.tif"))
            export_engine = "python_adaptive_staging"
            export_mode = "staged"
        else:
            try:
                fiji_export_result = run_fiji_batch_export(
                    directories["raw"],
                    directories["masks"],
                    directories["labels"],
                    macro_path=config.fiji_macro_path,
                    fiji_executable=config.fiji_executable,
                    headless=config.fiji_headless,
                    timeout=config.fiji_timeout,
                    image_pattern=config.fiji_image_pattern,
                    log_dir=directories["logs"],
                )
            except (FileNotFoundError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
                export_warnings.append(
                    f"Fiji batch export unavailable; using Python TIFF fallback. {exc}"
                )
            else:
                timing["fiji_startup_seconds"] = fiji_export_result.startup_seconds
                timing["fiji_subprocess_seconds"] = fiji_export_result.subprocess_seconds
                timing["fiji_postprocess_seconds"] = fiji_export_result.postprocess_seconds

            if fiji_export_result is not None and fiji_export_result.succeeded:
                mask_exports = fiji_export_result.mask_exports
                label_exports = fiji_export_result.label_exports
                export_engine = "fiji"
                export_mode = "batch"
                if not mask_exports and not label_exports:
                    checkpoint = time.perf_counter()
                    organized = organize_cellprofiler_tiffs_for_fiji(
                        directories["raw"],
                        directories["masks"],
                        directories["labels"],
                        pattern=config.fiji_image_pattern,
                    )
                    mask_exports = organized.masks
                    label_exports = organized.labels
                    timing["fiji_postprocess_seconds"] += elapsed_since(checkpoint)
                    if mask_exports or label_exports:
                        export_warnings.append(
                            "Fiji batch export found no mask/label TIFFs by filename; "
                            "classified CellProfiler TIFF outputs in Python."
                        )
                        export_engine = "fiji+python_classify"
                        export_mode = "batch+in_process"
            else:
                if fiji_export_result is not None:
                    export_warnings.append(
                        "Fiji batch export failed; using Python TIFF fallback. "
                        + format_fiji_error_summary(fiji_export_result)
                    )
                if not config.fiji_fallback_to_python:
                    raise RuntimeError(
                        "Fiji batch export failed and Python fallback is disabled."
                    )
                checkpoint = time.perf_counter()
                organized = organize_cellprofiler_tiffs_for_fiji(
                    directories["raw"],
                    directories["masks"],
                    directories["labels"],
                    pattern=config.fiji_image_pattern,
                )
                mask_exports = organized.masks
                label_exports = organized.labels
                timing["fiji_postprocess_seconds"] += elapsed_since(checkpoint)
                export_engine = "python_fallback"
                export_mode = "in_process"

            if export_warnings:
                export_warning_log = directories["logs"] / "fiji_export_warning.log"
                export_warning_log.write_text(
                    "\n".join(export_warnings),
                    encoding="utf-8",
                )

    qc_artifacts: dict[str, dict[str, Path]] = {}
    qc_image_names = processed_images
    if config.adaptive_threshold and adaptive_summary is not None:
        qc_image_names = adaptive_summary.get("processed", processed_images)
    if config.generate_qc and qc_image_names:
        checkpoint = time.perf_counter()
        qc_artifacts = generate_qc_for_cellprofiler_results(
            cellprofiler_input_dir,
            directories["masks"],
            directories["labels"],
            directories["qc"],
            qc_image_names,
        )
        timing["qc_seconds"] = elapsed_since(checkpoint)

    checkpoint = time.perf_counter()
    log_files = dict(run_result.log_files)
    if oir_projection_log is not None:
        log_files["oir_projection"] = oir_projection_log
    if fiji_export_result is not None:
        log_files.update(
            {f"fiji_{key}": path for key, path in fiji_export_result.log_files.items()}
        )
    if export_warning_log is not None:
        log_files["fiji_warning"] = export_warning_log
    _attach_oir_projection_lifecycle_logs(log_files, lifecycle)

    result = CellProfilerWorkflowResult(
        results_dir=directories["results"].resolve(),
        raw_output_dir=directories["raw"].resolve(),
        measurements_dir=directories["measurements"].resolve(),
        masks_dir=directories["masks"].resolve(),
        labels_dir=directories["labels"].resolve(),
        qc_dir=directories["qc"].resolve(),
        logs_dir=directories["logs"].resolve(),
        processed_images=processed_images,
        tables=tables,
        table_summary=summarize_cellprofiler_tables(tables),
        measurements=measurements,
        mask_exports=mask_exports,
        label_exports=label_exports,
        qc_artifacts=qc_artifacts,
        log_files=log_files,
        cellprofiler_run=run_result,
        timing=timing,
        export_engine=export_engine,
        export_mode=export_mode,
        fiji_export_result=fiji_export_result,
        export_warnings=export_warnings or None,
        adaptive_threshold_summary=adaptive_summary,
        import_warnings=import_warnings or None,
    )
    summary_path = _write_workflow_summary(directories["logs"], result)
    result.log_files["workflow_summary"] = summary_path
    timing["final_cleanup_seconds"] = elapsed_since(checkpoint)
    _finalize_and_log_timing(timing, total_started, directories["logs"])
    result.timing = timing
    return result


def run_analysis_from_config(config: AnalysisConfig) -> dict[str, Any]:
    """Run analysis from an :class:`AnalysisConfig` instance."""
    validated = AnalysisConfig(
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        analysis_engine=_validate_engine(config.analysis_engine),
        pattern=config.pattern,
        pipeline=config.pipeline,
        blur_sigma=config.blur_sigma,
        min_object_size=config.min_object_size,
        labeling_method=_validate_labeling_method(config.labeling_method),
        clean_mask_before_labeling=config.clean_mask_before_labeling,
        watershed_min_distance=config.watershed_min_distance,
        watershed_min_peak_ratio=config.watershed_min_peak_ratio,
        cppipe_path=config.cppipe_path,
        cellprofiler_executable=config.cellprofiler_executable,
        cellprofiler_extra_args=config.cellprofiler_extra_args,
        merge_measurements=config.merge_measurements,
    )

    if validated.analysis_engine == "python":
        return _run_python_analysis(validated)
    return _run_cellprofiler_analysis(validated)
