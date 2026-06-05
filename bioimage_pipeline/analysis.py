"""Unified analysis entry point for Python and CellProfiler engines."""

from __future__ import annotations

import json
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
    load_cellprofiler_measurements,
    merge_cellprofiler_tables,
    run_cellprofiler_pipeline,
    run_cellprofiler_pipeline_logged,
    summarize_cellprofiler_tables,
)
from bioimage_pipeline.export import (
    export_measurements_csv,
    organize_cellprofiler_tiffs_for_fiji,
)
from bioimage_pipeline.qc import generate_qc_for_cellprofiler_results
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

AnalysisEngine = Literal["python", "cellprofiler"]
LabelingMethod = Literal["connected", "watershed"]
_SUPPORTED_ENGINES = frozenset({"python", "cellprofiler"})
_SUPPORTED_LABELING = frozenset({"connected", "watershed"})


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
    """

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
    tables = load_cellprofiler_measurements(output_dir)
    measurements = (
        merge_cellprofiler_tables(tables) if config.merge_measurements else None
    )

    return {
        "analysis_engine": "cellprofiler",
        "output_dir": output_dir,
        "processed": None,
        "failed": None,
        "tables": tables,
        "measurements": measurements,
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


def _prepare_workflow_directories(results_dir: Path) -> dict[str, Path]:
    directories = {
        "results": results_dir,
        "raw": results_dir / RESULTS_RAW_DIR,
        "measurements": results_dir / RESULTS_MEASUREMENTS_DIR,
        "masks": results_dir / RESULTS_MASKS_DIR,
        "labels": results_dir / RESULTS_LABELS_DIR,
        "qc": results_dir / RESULTS_QC_DIR,
        "logs": results_dir / RESULTS_LOGS_DIR,
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
) -> CellProfilerWorkflowResult:
    """Run the CellProfiler-to-Fiji workflow and organize all outputs.

    Steps: headless CellProfiler run → capture logs → collect CSV/TIFF outputs
    → write Fiji-compatible masks and labels → generate QC overlays.

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
    )
    return run_cellprofiler_workflow_from_config(config)


def run_cellprofiler_workflow_from_config(
    config: CellProfilerWorkflowConfig,
) -> CellProfilerWorkflowResult:
    """Run :func:`run_cellprofiler_workflow` from a config object."""
    directories = _prepare_workflow_directories(Path(config.output_dir))

    run_result = run_cellprofiler_pipeline_logged(
        cppipe_path=config.cppipe_path,
        input_dir=config.input_dir,
        output_dir=directories["raw"],
        extra_args=config.cellprofiler_extra_args,
        cellprofiler_executable=config.cellprofiler_executable,
        log_dir=directories["logs"],
    )
    if not run_result.succeeded:
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
                log_files=run_result.log_files,
                cellprofiler_run=run_result,
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

    copy_cellprofiler_measurements(
        directories["raw"],
        directories["measurements"],
    )
    tables = load_cellprofiler_measurements(directories["measurements"])
    measurements = (
        merge_cellprofiler_tables(tables) if config.merge_measurements else None
    )
    if measurements is not None:
        export_measurements_csv(
            directories["measurements"] / "merged_measurements.csv",
            measurements,
        )

    processed_images = extract_processed_image_names(tables)

    mask_exports: list[Path] = []
    label_exports: list[Path] = []
    if config.export_fiji_tiffs:
        organized = organize_cellprofiler_tiffs_for_fiji(
            directories["raw"],
            directories["masks"],
            directories["labels"],
            pattern=config.fiji_image_pattern,
        )
        mask_exports = organized.masks
        label_exports = organized.labels

    qc_artifacts: dict[str, dict[str, Path]] = {}
    if config.generate_qc and processed_images:
        qc_artifacts = generate_qc_for_cellprofiler_results(
            config.input_dir,
            directories["masks"],
            directories["labels"],
            directories["qc"],
            processed_images,
        )

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
        log_files=run_result.log_files,
        cellprofiler_run=run_result,
    )
    summary_path = _write_workflow_summary(directories["logs"], result)
    result.log_files["workflow_summary"] = summary_path
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
