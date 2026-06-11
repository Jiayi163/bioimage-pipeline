# GUI Direction (Phase 15 — import-only orchestration shell)

The GUI is a **front-end and workflow manager** for CellProfiler and Fiji. It
imports user-authored ``.cppipe`` files and orchestrates headless batch runs — it
does **not** reimplement CellProfiler algorithms or edit module settings in-app.

Phase 15 ships as an **import-only shell**: pipeline path, read-only module summary,
input/output folders, run controls, logs, and QC preview.

## Product goal

| Goal | Yes / No |
|------|----------|
| Expose CellProfiler modules and pipelines through our application | **Yes** (import + read-only summary) |
| Reimplement CellProfiler segmentation/measurement algorithms in Python | **No** |
| Run CellProfiler headlessly as the analysis engine | **Yes** |
| Provide visual pipeline building and parameter configuration | **No** — use native CellProfiler |
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
│  • run_fiji_batch_export()                                  │
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

### Phase 15.1 — GUI workflow shell

Proper workflow shell for running saved pipelines and reviewing results without
Python scripting or showing Fiji/CellProfiler windows.

Implemented as a standard-library Tkinter shell in
`bioimage_pipeline/gui/workflow_shell.py`, launched by `examples/run_gui.py`.

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

Configure and compose CellProfiler pipelines in the GUI. The initial builder is
implemented in the Tkinter shell with text-preserving `.cppipe` I/O and a curated
module catalog.

| Feature | Description |
|---------|-------------|
| Module browser | Search/browse CP modules by name and category |
| Parameter panels | Edit module settings → persist to `.cppipe` |
| Pipeline editor | Add, remove, reorder modules |
| Load / save | Read and write standard `.cppipe` text |
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
2. **Editor** — read/write text `.cppipe` files conservatively; validate
   structure before run.
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

## Files

| File | Phase | Purpose |
|------|-------|---------|
| `bioimage_pipeline/gui/workflow_shell.py` | 15.1 | GUI workflow shell and testable helpers |
| `bioimage_pipeline/cppipe_io.py` | 15.2 | Load/save/parse `.cppipe` |
| `bioimage_pipeline/pipeline_catalog.py` | 15.2 | Module metadata for browser |
| `examples/run_gui.py` | 15.1 | Launch script |
| `tests/test_cppipe_io.py` | 15.2 | Pipeline I/O tests |
| `tests/test_gui_workflow.py` | 15.1 | Workflow shell tests (mocked CP) |

## Technology options

| Option | Pros | Notes |
|--------|------|-------|
| **Tkinter** | No new dependency, available in standard Python installs | Used for 15.1 shell |
| **PyQt / PySide** | Native desktop, rich pipeline editor | Candidate for future richer shell/editor |
| **Web UI + local API** | Flexible layout | Candidate for future richer shell/editor |

All options must call the same orchestration APIs (`analysis.py`,
`cellprofiler_runner.py`, `fiji_runner.py`).

## Related docs

- [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md) — Phase 15 acceptance criteria
- [cellprofiler_workflow.md](cellprofiler_workflow.md) — headless CP orchestration
- [fiji_headless_export.md](fiji_headless_export.md) — batch Fiji export
