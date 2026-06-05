"""Phase 15.0 — temporary Streamlit workflow test UI (prototype, not final GUI)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from bioimage_pipeline.analysis import (
    CellProfilerWorkflowResult,
    run_cellprofiler_workflow,
)
from bioimage_pipeline.workflow_ui import (
    list_qc_pngs,
    load_measurements_for_display,
    read_text_tail,
    save_uploaded_cppipe,
    validate_workflow_inputs,
)

APP_TITLE = "Bioimage Pipeline — Workflow Test UI"
PROTOTYPE_NOTICE = (
    "**Prototype test UI (Phase 15.0).** This is not the final GUI. "
    "It wraps existing workflow APIs so you can click-run pipelines and "
    "inspect logs, overlays, measurements, and outputs."
)


def _init_session_state() -> None:
    defaults = {
        "workflow_result": None,
        "workflow_error": None,
        "cached_cppipe_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _resolve_cppipe_path(
    uploaded_file,
    pipeline_path_text: str,
    cache_dir: Path,
) -> Path | None:
    if uploaded_file is not None:
        cached = save_uploaded_cppipe(
            uploaded_file,
            uploaded_file.name,
            cache_dir,
        )
        st.session_state["cached_cppipe_path"] = str(cached)
        return cached

    if pipeline_path_text.strip():
        return Path(pipeline_path_text.strip()).resolve()

    if st.session_state.get("cached_cppipe_path"):
        cached_path = Path(st.session_state["cached_cppipe_path"])
        if cached_path.is_file():
            return cached_path
    return None


def _render_log_tabs(result: CellProfilerWorkflowResult) -> None:
    log_files = result.log_files
    tab_labels = []
    tab_contents: list[str] = []

    for label in ("stdout", "stderr", "workflow_summary", "command"):
        path = log_files.get(label)
        if path is None:
            continue
        tab_labels.append(label)
        if label == "workflow_summary" and Path(path).is_file():
            try:
                summary = json.loads(Path(path).read_text(encoding="utf-8"))
                tab_contents.append(json.dumps(summary, indent=2))
            except json.JSONDecodeError:
                tab_contents.append(read_text_tail(path))
        else:
            tab_contents.append(read_text_tail(path))

    if not tab_labels:
        st.info("No log files were captured for this run.")
        return

    tabs = st.tabs(tab_labels)
    for tab, label, content in zip(tabs, tab_labels, tab_contents, strict=True):
        with tab:
            language = "json" if label == "workflow_summary" else "text"
            st.code(content, language=language)


def _render_qc_gallery(result: CellProfilerWorkflowResult) -> None:
    png_files = list_qc_pngs(result.qc_dir)
    if not png_files:
        st.info("No QC overlay PNGs found in the output folder.")
        return

    columns = st.columns(min(3, len(png_files)))
    for index, png_path in enumerate(png_files):
        with columns[index % len(columns)]:
            st.image(str(png_path), caption=png_path.name, use_container_width=True)


def _render_import_warnings(result: CellProfilerWorkflowResult) -> None:
    warnings = result.import_warnings or []
    if not warnings:
        return
    st.subheader("Import warnings")
    for warning in warnings:
        st.warning(warning)


def _render_measurements(result: CellProfilerWorkflowResult) -> None:
    measurements = result.measurements
    if measurements is None:
        measurements = load_measurements_for_display(result.measurements_dir)

    if measurements is not None and not measurements.empty:
        st.markdown("**Merged measurements**")
        st.dataframe(measurements, use_container_width=True, hide_index=True)
        return

    if result.tables:
        st.info("Merged measurements unavailable; showing individual CellProfiler tables.")
        for table_name, dataframe in sorted(result.tables.items()):
            st.markdown(f"**{table_name}**")
            st.dataframe(dataframe, use_container_width=True, hide_index=True)
        return

    st.info("No measurement tables were found for this run.")


def _render_output_folders(result: CellProfilerWorkflowResult) -> None:
    folders = {
        "Results root": result.results_dir,
        "CellProfiler raw": result.raw_output_dir,
        "Measurements": result.measurements_dir,
        "Masks": result.masks_dir,
        "Labels": result.labels_dir,
        "QC": result.qc_dir,
        "Logs": result.logs_dir,
    }
    for label, folder in folders.items():
        st.text(f"{label}: {folder}")


def _render_result(result: CellProfilerWorkflowResult) -> None:
    st.success("Workflow completed.")
    st.subheader("Summary")
    st.json(result.to_dict())

    st.subheader("Output folders")
    _render_output_folders(result)

    st.subheader("Logs")
    _render_log_tabs(result)

    st.subheader("QC overlays")
    _render_qc_gallery(result)

    _render_import_warnings(result)

    st.subheader("Measurements")
    _render_measurements(result)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _init_session_state()

    st.title(APP_TITLE)
    st.markdown(PROTOTYPE_NOTICE)

    with st.form("workflow_form", clear_on_submit=False):
        col_left, col_right = st.columns(2)
        with col_left:
            input_dir = st.text_input(
                "Input image folder",
                placeholder=r"C:\data\images",
                help="Folder passed to CellProfiler as -i",
            )
            pipeline_path_text = st.text_input(
                "Pipeline file path (optional if uploading)",
                placeholder=r"C:\pipelines\nuclei.cppipe",
            )
            cellprofiler_executable = st.text_input(
                "CellProfiler executable",
                value="cellprofiler",
                help="Command name or full path to CellProfiler.exe",
            )
        with col_right:
            output_dir = st.text_input(
                "Output folder",
                placeholder=r"C:\data\workflow_results",
                help="Root folder for measurements/, masks/, labels/, qc/, logs/",
            )
            uploaded_cppipe = st.file_uploader(
                "Upload .cppipe pipeline",
                type=["cppipe"],
            )

        export_fiji_tiffs = st.checkbox("Export Fiji-compatible TIFFs (Python fallback)", value=True)
        generate_qc = st.checkbox("Generate QC overlays", value=True)
        adaptive_threshold = st.checkbox(
            "Experimental adaptive threshold (Phase 17 prototype)",
            value=False,
            help="Opt-in self-adaptive Python threshold staging before CellProfiler.",
        )

        submitted = st.form_submit_button("Run Workflow", type="primary")

    if submitted:
        cache_dir = Path(tempfile.gettempdir()) / "bioimage_pipeline_ui"
        cppipe_path = _resolve_cppipe_path(uploaded_cppipe, pipeline_path_text, cache_dir)
        errors = validate_workflow_inputs(input_dir, output_dir, cppipe_path)
        if errors:
            st.session_state["workflow_result"] = None
            st.session_state["workflow_error"] = "\n".join(errors)
        else:
            assert cppipe_path is not None
            with st.spinner("Running CellProfiler workflow... This may take several minutes."):
                try:
                    result = run_cellprofiler_workflow(
                        input_dir,
                        output_dir,
                        cppipe_path,
                        cellprofiler_executable=cellprofiler_executable.strip() or "cellprofiler",
                        export_fiji_tiffs=export_fiji_tiffs,
                        generate_qc=generate_qc,
                        adaptive_threshold=adaptive_threshold,
                    )
                    st.session_state["workflow_result"] = result
                    st.session_state["workflow_error"] = None
                except Exception as exc:  # noqa: BLE001 — surface workflow failures in UI
                    st.session_state["workflow_result"] = None
                    st.session_state["workflow_error"] = str(exc)

    if st.session_state["workflow_error"]:
        st.error(st.session_state["workflow_error"])
        logs_dir = Path(output_dir) / "logs" if output_dir.strip() else None
        if logs_dir and logs_dir.is_dir():
            st.subheader("Partial logs")
            for log_name in ("cellprofiler_stdout.log", "cellprofiler_stderr.log"):
                log_path = logs_dir / log_name
                if log_path.is_file():
                    st.markdown(f"**{log_name}**")
                    st.code(read_text_tail(log_path))

    result = st.session_state.get("workflow_result")
    if isinstance(result, CellProfilerWorkflowResult):
        _render_result(result)


if __name__ == "__main__":
    main()
