# GUI Direction (Phase 15 — 15.0 → 15.1 → 15.2)

The GUI is a **front-end and workflow manager** for CellProfiler and Fiji. It
exposes CellProfiler functionality through pipeline configuration and orchestration
— it does **not** reimplement CellProfiler algorithms.

Phase 15 ships in three steps: **15.0** (temporary Streamlit test UI — visible
now), **15.1** (proper workflow shell), **15.2** (pipeline builder).

## Product goal

| Goal | Yes / No |
|------|----------|
| Expose CellProfiler modules and pipelines through our application | **Yes** |
| Reimplement CellProfiler segmentation/measurement algorithms in Python | **No** |
| Run CellProfiler headlessly as the analysis engine | **Yes** |
| Provide visual pipeline building and parameter configuration | **Yes** |
| Replace CellProfiler desktop app for execution | **No** — we delegate to CP CLI |

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│  GUI (Phase 15)                                             │
│  • pipeline builder / module browser                        │
│  • workflow controls, logs, previews                          │
└──────────────────────────┬──────────────────────────────────┘
                           │ generates / loads .cppipe
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  This project — orchestration (Phases 13–14)                  │
│  • run_cellprofiler_workflow()                              │
│  • run_fiji_batch_export() (planned)                        │
│  • QC, logs, results folders                                │
└──────────────┬─────────────────────────┬────────────────────┘
               │ headless subprocess      │ headless subprocess
               ▼                          ▼
     ┌─────────────────┐       ┌─────────────────┐
     │  CellProfiler   │       │  Fiji/ImageJ    │
     │  analysis engine│       │  export/QC eng. │
     └─────────────────┘       └─────────────────┘
```

## Conceptual flow

```text
GUI
  → generates / manages pipeline configuration (.cppipe)
  → calls CellProfiler (headless, one run per folder)
  → collects outputs
  → optionally calls Fiji/ImageJ (headless batch export)
  → displays results (masks, labels, measurements, QC, logs)
```

## Phase 15 sub-phases

### Phase 15.0 — Temporary Streamlit workflow test UI

**Not the final GUI.** Fast prototype so you can click-run existing workflows and
see logs, overlays, measurements, and outputs without writing scripts.

| Feature | Description |
|---------|-------------|
| Input folder selection | Choose image directory for batch run |
| Pipeline load | Pick or upload existing `.cppipe` file |
| Executable config | CellProfiler and Fiji paths (sidebar / session) |
| Run workflow | Button → `run_cellprofiler_workflow()` (+ optional Fiji) |
| Logs | Stream from `logs/` (stdout/stderr, workflow summary) |
| Result preview | QC overlay PNGs, measurement CSV tables |
| Outputs | Link or download organized result folders |

Launch: `pip install -e ".[ui]"` then `streamlit run app/workflow_test_ui.py`.

### Phase 15.1 — GUI workflow shell

Proper workflow shell — may evolve from or replace 15.0. Run saved pipelines and
review results without Python scripting.

| Feature | Description |
|---------|-------------|
| Input folder selection | Choose image directory for batch run |
| Pipeline load | Open existing `.cppipe` file |
| Executable config | CellProfiler and Fiji paths (persisted) |
| Run workflow | Batch CP run + optional Fiji export |
| Logs & progress | Tail stdout/stderr, show stage timings |
| Result preview | QC overlays, measurement tables |
| Export results | Open or copy organized output folders |

### Phase 15.2 — Pipeline builder & module exposure

Configure and compose CellProfiler pipelines in the GUI.

| Feature | Description |
|---------|-------------|
| Module browser | Search/browse CP modules by name and category |
| Parameter panels | Edit module settings → persist to `.cppipe` |
| Pipeline editor | Add, remove, reorder modules |
| Load / save | Read and write standard `.cppipe` JSON |
| Run from editor | Hand off to 15.1 workflow shell |

## CellProfiler modules — expose, do not reimplement

The GUI surfaces existing CellProfiler capabilities as configurable pipeline
steps. Examples of modules to **expose** (not rewrite):

- `IdentifyPrimaryObjects`
- `IdentifySecondaryObjects`
- `MeasureObjectSizeShape`
- `MeasureObjectIntensity`
- `MeasureTexture`
- `RelateObjects`
- `ClassifyObjects`
- `ExportToSpreadsheet`
- `SaveImages`

Implementation approach:

1. **Catalog** — module names, categories, parameter keys (from `.cppipe` schema
   or curated metadata aligned with CellProfiler).
2. **Editor** — read/write `.cppipe` JSON; validate structure before run.
3. **Execution** — always delegate to `cellprofiler -c -r -p ...`; never run
   parallel Python implementations of the same algorithms for production.

## Fiji integration in the GUI

Fiji is optional in the workflow UI:

- Toggle Fiji headless export on/off per run.
- Configure Fiji executable and batch macro path.
- Preview final exported TIFFs and QC overlays after export completes.
- Fall back to Python TIFF export when Fiji is unavailable (logged in UI).

## Out of scope

- Rewriting any CellProfiler module logic in Python or in the GUI backend.
- Reimplementing Fiji plugins or Bio-Formats import pipelines.
- Per-image CellProfiler or Fiji subprocess launches (see batch-first rules in
  [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md)).

## Relationship to Python analysis engine

Phases 0–11 built a **lightweight Python pipeline** for teaching, tests, and
prototyping. The GUI **primary path** is CellProfiler. The Python engine may
appear later as an optional "simple mode" but must not become the main product
surface or duplicate CellProfiler module functionality.

## Planned files

| File | Phase | Purpose |
|------|-------|---------|
| `app/workflow_test_ui.py` | 15.0 | Temporary Streamlit test UI |
| `bioimage_pipeline/workflow_ui.py` | 15.0 | UI helpers (logs, QC, validation) |
| `bioimage_pipeline/gui/` or `app/` | 15.1 | GUI application package |
| `bioimage_pipeline/cppipe_io.py` | 15.2 | Load/save/parse `.cppipe` |
| `bioimage_pipeline/pipeline_catalog.py` | 15.2 | Module metadata for browser |
| `examples/run_gui.py` | 15.1 | Launch script |
| `tests/test_streamlit_workflow_ui.py` | 15.0 | Test UI smoke tests |
| `tests/test_cppipe_io.py` | 15.2 | Pipeline I/O tests |
| `tests/test_gui_workflow.py` | 15.1 | Workflow shell tests (mocked CP) |

## Technology options

| Option | Pros | Notes |
|--------|------|-------|
| **Streamlit** | Fast prototype | **15.0 first**; may evolve into 15.1 |
| **PyQt / PySide** | Native desktop, rich pipeline editor | Candidate for 15.1 / 15.2 |
| **Web UI + local API** | Flexible layout | Longer setup |

All options must call the same orchestration APIs (`analysis.py`, `cellprofiler_runner.py`, future `fiji_runner.py`).

## Related docs

- [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md) — Phase 15 acceptance criteria
- [cellprofiler_workflow.md](cellprofiler_workflow.md) — headless CP orchestration
- [fiji_headless_export.md](fiji_headless_export.md) — batch Fiji export plan
