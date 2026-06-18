"""Thin GUI controller helpers over workflow library functions.

Scientific orchestration stays in :mod:`bioimage_pipeline.analysis` and related
library modules. This module only maps form values, persisted settings, and
display helpers for the Tk shell.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bioimage_pipeline.analysis import resolve_workflow_output_dir
from bioimage_pipeline.cellprofiler_runner import RESULTS_LOGS_DIR
from bioimage_pipeline.cppipe_io import advise_pipeline_for_run
from bioimage_pipeline.gui.run_settings import (
    CPPIPE_PATH_KEY,
    EXPORT_FIJI_TIFFS_KEY,
    FIJI_MACRO_PATH_KEY,
    FORCE_OIR_REPROJECT_KEY,
    GENERATE_QC_KEY,
    INPUT_DIR_KEY,
    OIR_PROJECTION_ENGINE_KEY,
    OIR_PROJECTION_METHOD_KEY,
    OUTPUT_DIR_KEY,
    parse_bool_setting,
)
from bioimage_pipeline.gui.workflow_shell import (
    GuiWorkflowConfig,
    GuiWorkflowSummary,
    load_imported_pipeline,
    load_measurements_preview,
    read_log_tail,
    resolve_imported_pipeline_path,
    validate_workflow_config,
)
from bioimage_pipeline.gui.workflow_editor import resolve_workflow_input_dir_from_string
from bioimage_pipeline.z_projection import iter_oir_files


@dataclass(frozen=True)
class WorkflowFormValues:
    """User-editable workflow shell field values."""

    input_dir: str
    output_dir: str
    cppipe_path: str
    cellprofiler_executable: str
    fiji_executable: str
    fiji_macro_path: str
    export_fiji_tiffs: bool
    generate_qc: bool
    oir_projection_engine: str
    oir_projection_method: str
    force_oir_reproject: bool


@dataclass(frozen=True)
class WorkflowRunPreparation:
    """Validated inputs ready for :func:`run_gui_workflow`."""

    config: GuiWorkflowConfig
    advisories: tuple[str, ...]
    requires_oir_method_dialog: bool


def workflow_form_values_from_settings(
    settings: dict[str, str],
    *,
    default_oir_projection_engine: str,
) -> WorkflowFormValues:
    """Hydrate workflow form defaults from persisted GUI settings."""
    return WorkflowFormValues(
        input_dir=settings.get(INPUT_DIR_KEY, ""),
        output_dir=settings.get(OUTPUT_DIR_KEY, ""),
        cppipe_path=settings.get(CPPIPE_PATH_KEY, ""),
        cellprofiler_executable=settings.get("cellprofiler_executable", "cellprofiler"),
        fiji_executable=settings.get("fiji_executable", ""),
        fiji_macro_path=settings.get(FIJI_MACRO_PATH_KEY, ""),
        export_fiji_tiffs=parse_bool_setting(
            settings.get(EXPORT_FIJI_TIFFS_KEY),
            default=True,
        ),
        generate_qc=parse_bool_setting(settings.get(GENERATE_QC_KEY), default=True),
        oir_projection_engine=settings.get(
            OIR_PROJECTION_ENGINE_KEY,
            default_oir_projection_engine,
        ),
        oir_projection_method=settings.get(OIR_PROJECTION_METHOD_KEY, "max"),
        force_oir_reproject=parse_bool_setting(
            settings.get(FORCE_OIR_REPROJECT_KEY),
            default=False,
        ),
    )


def build_gui_workflow_config(values: WorkflowFormValues) -> GuiWorkflowConfig:
    """Map workflow form values to :class:`GuiWorkflowConfig`."""
    def optional_text(text: str) -> str | None:
        stripped = text.strip()
        return stripped or None

    return GuiWorkflowConfig(
        input_dir=values.input_dir.strip(),
        output_dir=values.output_dir.strip(),
        cppipe_path=values.cppipe_path.strip(),
        cellprofiler_executable=values.cellprofiler_executable.strip() or "cellprofiler",
        fiji_executable=optional_text(values.fiji_executable),
        fiji_macro_path=optional_text(values.fiji_macro_path),
        export_fiji_tiffs=values.export_fiji_tiffs,
        generate_qc=values.generate_qc,
        oir_projection_engine=values.oir_projection_engine.strip() or "fiji",
        oir_projection_method=values.oir_projection_method.strip() or "max",
        force_oir_reproject=values.force_oir_reproject,
    )


def prepare_workflow_run(values: WorkflowFormValues) -> WorkflowRunPreparation:
    """Validate workflow form values and resolve paths for a GUI run."""
    cppipe_path = resolve_imported_pipeline_path(values.cppipe_path)
    input_path = resolve_workflow_input_dir_from_string(values.input_dir)
    imported = load_imported_pipeline(cppipe_path)
    advisories = tuple(advise_pipeline_for_run(imported.pipeline))

    resolved_output_dir = resolve_workflow_output_dir(values.output_dir)
    config = build_gui_workflow_config(
        WorkflowFormValues(
            input_dir=str(input_path),
            output_dir=str(resolved_output_dir),
            cppipe_path=str(cppipe_path),
            cellprofiler_executable=values.cellprofiler_executable,
            fiji_executable=values.fiji_executable,
            fiji_macro_path=values.fiji_macro_path,
            export_fiji_tiffs=values.export_fiji_tiffs,
            generate_qc=values.generate_qc,
            oir_projection_engine=values.oir_projection_engine,
            oir_projection_method=values.oir_projection_method,
            force_oir_reproject=values.force_oir_reproject,
        )
    )
    errors = validate_workflow_config(config)
    if errors:
        raise ValueError("\n".join(errors))

    return WorkflowRunPreparation(
        config=config,
        advisories=advisories,
        requires_oir_method_dialog=bool(list(iter_oir_files(input_path))),
    )


def workflow_logs_dir(output_dir: str | Path) -> Path:
    """Return the workflow logs directory for an output folder."""
    return resolve_workflow_output_dir(output_dir) / RESULTS_LOGS_DIR


def workflow_log_paths(output_dir: str | Path) -> dict[str, Path]:
    """Return predictable log file paths under a workflow output folder."""
    logs_dir = workflow_logs_dir(output_dir)
    return {
        "stdout": logs_dir / "cellprofiler_stdout.log",
        "stderr": logs_dir / "cellprofiler_stderr.log",
        "workflow_summary": logs_dir / "workflow_summary.json",
        "oir_projection_summary": logs_dir / "oir_projection_summary.json",
        "prepare_input_profile": logs_dir / "prepare_input_profile.json",
    }


def format_workflow_log_tail(output_dir: str | Path, *, max_lines: int = 80) -> str:
    """Format stdout/stderr tails for the GUI log panel."""
    paths = workflow_log_paths(output_dir)
    sections: list[str] = []
    for label, key in (("stdout", "stdout"), ("stderr", "stderr")):
        path = paths[key]
        if not path.is_file():
            continue
        content = read_log_tail(path, max_lines=max_lines)
        if content:
            sections.append(f"=== CellProfiler {label} ===\n{content}")
    return "\n\n".join(sections)


def format_measurements_preview_text(
    measurement_files: list[Path],
    *,
    max_rows: int = 20,
) -> str:
    """Format a small measurements CSV preview for the GUI."""
    if not measurement_files:
        return "No measurement CSV files found."
    target = measurement_files[0]
    try:
        preview = load_measurements_preview(target, max_rows=max_rows)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return f"Could not load measurements preview from {target.name}: {exc}"
    return preview.to_string(index=False)


def output_shortcut_targets(
    summary: GuiWorkflowSummary,
) -> dict[str, Path]:
    """Return named output folders for GUI shortcut buttons."""
    return {
        "results": summary.results_dir,
        "measurements": summary.measurements_dir,
        "qc": summary.qc_dir,
        "logs": summary.logs_dir,
        "masks": summary.masks_dir,
        "labels": summary.labels_dir,
    }
