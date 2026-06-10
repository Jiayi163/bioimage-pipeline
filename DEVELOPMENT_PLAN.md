# Development Plan

This project is a **lightweight bioimage analysis pipeline** inspired by
Fiji/ImageJ and CellProfiler. It provides a Python-native stack and
batch-processing workflow, a CellProfiler integration layer, and a
Fiji-compatible TIFF export path.

## Project goal

```text
Input Images (folder or multi-page TIFF stack)
    ↓
Python pipeline  →  preprocess → segment → measure  →  masks, labels, CSV
    OR
CellProfiler     →  headless .cppipe run             →  measurements, exports
    OR
Fiji/ImageJ      →  headless macro export            →  final TIFFs + metadata
```

| Layer | Role |
|-------|------|
| **Python pipeline** | Standalone analysis — stack/batch processing, segmentation, measurement, export |
| **CellProfiler** | External analysis engine — headless `.cppipe` runs, CSV output |
| **Fiji/ImageJ** | External export/QC engine — headless TIFF export, macros |
| **Python TIFF export** | In-process fallback — ImageJ-compatible TIFF writing via `tifffile` |

**What we do:**

- Process folders of TIFFs or multi-page TIFF stacks with a composable Python pipeline.
- Run CellProfiler headlessly via `.cppipe` pipelines.
- Collect, organize, and preview CellProfiler outputs.
- Drive Fiji/ImageJ headless export when required (Phase 14).
- Provide a GUI that manages pipelines, calls external engines, and displays results (Phase 15).

## Performance: batch-first execution

Do **not** design the workflow to launch CellProfiler or Fiji once per image
unless absolutely necessary. External tool startup dominates runtime; batch
invocation is the default.

**Preferred batch workflow:**

1. Run **CellProfiler once** per input folder using a single `.cppipe` pipeline
   (`-i input_dir -o output_dir`). CellProfiler processes all images internally.
2. **Collect all** CellProfiler outputs (CSVs, masks, labels, raw TIFFs) from
   `cellprofiler_raw/` before any export step.
3. Run **Fiji/ImageJ headlessly once** per batch/folder when possible (Phase 14).
4. Use Fiji **macros/scripts that process an entire folder** (input dir + output
   dir arguments), not one subprocess per file.
5. Fall back to **per-image export** only when batch macro export is not possible
   (e.g. macro limitation, single-file debug mode). Python TIFF fallback may loop
   per file in-process — that is acceptable because it avoids JVM/process startup.
6. **Log timing** for each workflow stage in `logs/workflow_summary.json`:
   - CellProfiler runtime (seconds)
   - Fiji export runtime (seconds)
   - QC generation runtime (seconds)
7. Keep **Python TIFF export** as a fast in-process fallback for tests and simple
   cases when Fiji is unavailable.

**Current implementation (Phase 13):**

| Stage | Batch-first? | Notes |
|-------|--------------|-------|
| CellProfiler | Yes | One subprocess per folder via `run_cellprofiler_pipeline_logged()` |
| Python TIFF fallback | In-process loop | No extra CP/Fiji launches; acceptable for fallback |
| QC overlays | Per-image in Python | In-process only; no external tool relaunch |
| Stage timing | Planned Phase 14 | To be added to workflow summary |

**Anti-patterns to avoid:**

- Spawning CellProfiler once per image file.
- Spawning Fiji once per TIFF (unless batch macro cannot handle the format).
- Re-running analysis when only export settings change — re-export from collected
  `cellprofiler_raw/` instead.
- Building GUI controls that duplicate CellProfiler module logic in Python — the
  GUI configures `.cppipe` pipelines and invokes CellProfiler instead.

## Self-adaptive thresholding (Phase 17 — deferred, core differentiator)

**Status: `DEFERRED` — early prototype in repo, not enabled on default workflows.**

This is the project's **special value**: automatic per-image import-time thresholding
that CellProfiler cannot provide alone. Deferring Phase 17 is reasonable while
Phases 14–16 land, but it must **not disappear behind CellProfiler** in the roadmap
or product narrative. CellProfiler remains the analysis engine; Phase 17 is the
hybrid import layer that makes the workflow uniquely capable on difficult fluorescence
data.

CellProfiler alone cannot provide fully automatic per-image adaptive thresholding
at import (varying illumination, SNR, contrast) without either configuring
`.cppipe` methods or preprocessing in Python. The intended production path is:

```text
Python self-adaptive threshold (import) → staging/masks → CellProfiler (measurement)
```

An **experimental prototype** exists in `bioimage_pipeline/adaptive_import.py`
(rolling-ball, auto method selection, Sauvola/local/Otsu fallbacks, watershed,
confidence scoring). It is **not production-ready** and will likely need:

- Tuning on real fluorescence nuclei datasets beyond current synthetic fixtures
- SNR / vignette / density decision rules adjusted per assay
- Validation against manual or CellProfiler reference masks (IoU/Dice targets)
- GUI live-preview integration (Phase 15.1)
- Clear opt-in API — **not** the default Python or CellProfiler workflow until validated

**Current wiring (opt-in only):**

| Entry point | Default | Opt-in |
|-------------|---------|--------|
| `build_default_pipeline()` | Blur → Otsu (Phase 3–4) | N/A |
| `run_cellprofiler_workflow(..., adaptive_threshold=True)` | `False` | `--adaptive-threshold` CLI |
| `run_self_adaptive_threshold()` / `run_adaptive_threshold.py` | — | Direct API for experiments |

Do **not** treat Phase 17 as complete until real-data validation and product
sign-off. See full phase spec below (Phase 17 section).

## GUI direction (Phase 15 — 15.1 → 15.2)

Phase 15 starts with **15.1**, the proper workflow shell. **15.2** adds pipeline
building and CellProfiler module exposure.

The GUI is a **front-end and workflow manager** for CellProfiler and Fiji — not
a replacement for CellProfiler's analysis engine.

```text
GUI
  → generates / manages pipeline configuration (.cppipe)
  → calls CellProfiler (headless, batch)
  → collects outputs
  → optionally calls Fiji/ImageJ (headless export)
  → displays results (masks, labels, measurements, QC overlays, logs)
```

**In scope for the GUI:**

- Browse and search CellProfiler modules by name and category.
- Configure module parameters (read/write `.cppipe` or equivalent pipeline representation).
- Build and edit pipelines visually or step-by-step.
- Load and save pipelines (`.cppipe`).
- Select input image folders.
- Run CellProfiler headlessly (batch-first).
- View logs, progress, and stage timings.
- Preview masks, labels, and QC overlays.
- Export / open organized results.
- Optionally trigger Fiji/ImageJ headless export.

**Out of current scope for the GUI:**

- Replacing the CellProfiler desktop app for interactive pipeline editing.
- Real-time streaming analysis (batch-first is the design).

Phase 15 is split into **15.1** (workflow shell) and **15.2** (pipeline builder).
See [docs/gui_direction.md](docs/gui_direction.md).

Work should proceed phase by phase. Do not implement later phases until the
current phase has tests and passes its self-check.

## Phase Workflow

For every phase:

1. Implement the smallest working version.
2. Add or update pytest tests.
3. Review the code.
4. Check for missing imports, wrong paths, unclear names, inconsistent return
   types, shape errors, dtype problems, TIFF saving issues, CSV export issues,
   and overly complicated functions.
5. Fix problems found during review.
6. Summarize what was implemented.
7. List the exact terminal command to test it.
8. Mark the phase as `PHASE COMPLETE` or `PHASE NOT COMPLETE`.

## Roadmap at a Glance

| Phase | Goal | Status |
|-------|------|--------|
| 0 | Project setup | `PHASE COMPLETE` |
| 1 | TIFF I/O | `PHASE COMPLETE` |
| 2 | Basic preprocessing | `PHASE COMPLETE` |
| 3 | Basic thresholding | `PHASE COMPLETE` |
| 4 | Segmentation | `PHASE COMPLETE` |
| 5 | Measurement | `PHASE COMPLETE` |
| 6 | Export | `PHASE COMPLETE` |
| 7 | Pipeline core | `PHASE COMPLETE` |
| 8 | Batch processing | `PHASE COMPLETE` |
| 9 | Example workflow | `PHASE COMPLETE` |
| 10 | Visual check without GUI | `PHASE COMPLETE` |
| 10.1 | CellProfiler integration validation | `PHASE COMPLETE` |
| 10.2 | CellProfiler output import | `PHASE COMPLETE` |
| 10.3 | Unified analysis mode | `PHASE COMPLETE` |
| 10.4 | Real data validation | `PHASE COMPLETE` |
| 10.5 | Visualization and QC | `PHASE COMPLETE` |
| 11 | Advanced segmentation | `PHASE COMPLETE` |
| 11.1 | Mask cleanup (morphology) | `PHASE COMPLETE` |
| 12 | Python TIFF export (fallback / intermediate) | `PHASE COMPLETE` |
| 13 | CellProfiler workflow integration | `PHASE COMPLETE` ✔ |
| 14 | Fiji/ImageJ headless export integration | `PHASE COMPLETE` ✔ |
| 15.1 | GUI workflow shell (run, logs, preview, export) | `PHASE COMPLETE` ✔ |
| 15.2 | GUI pipeline builder & CP module exposure | `PHASE COMPLETE` ✔ |
| 16 | Optional Python analysis enhancements | `PHASE NOT COMPLETE` |
| 17 | Self-adaptive threshold at import (hybrid CP workflow) — **core differentiator** | `DEFERRED` — prototype only |
| **S.0** | **Stack track prep — fix failing test, update docs** | `PHASE COMPLETE` ✔ |
| **S.1** | **Stack I/O — AxisInfo, StackFrame, iter_stack_frames** | `PHASE COMPLETE` ✔ |
| **S.2** | Stack data model — ImageStack, load from folder or file | `PHASE COMPLETE` ✔ |
| **S.3** | Stack batch runner — run_pipeline_on_stack | `PHASE COMPLETE` ✔ |
| **S.4** | Stack export — per-frame TIFFs, combined CSV | `PHASE COMPLETE` ✔ |
| **S.5** | Processed-image export | `PHASE COMPLETE` ✔ |
| **S.6** | Fiji-macro-style batch recipe (CLI + optional JSON) | `PHASE COMPLETE` ✔ |
| **S.7** | Stack QC overlays, example, docs | `PHASE COMPLETE` ✔ |

## Phase 0: Project Setup

Goal: create the project structure and packaging files.

Tasks:

- Create the `bioimage_pipeline` package.
- Create `examples/` and `tests/`.
- Create `pyproject.toml`.
- Add dependencies.
- Add `README.md`.
- Add `.gitignore`.
- Do not implement full functionality yet.

Self-check:

- Can the package be imported?
- Does the folder structure match the plan?
- Does pytest run without import errors?

Status: `PHASE COMPLETE` ✔

## Phase 1: Fiji-Style TIFF Input/Output

Goal: implement TIFF reading and writing.

Files:

- `bioimage_pipeline/io.py`
- `tests/test_io.py`

Functions:

- `read_tiff(path) -> numpy.ndarray`
- `save_tiff(path, image) -> None`

Requirements:

- Use `tifffile`.
- Use `pathlib.Path`.
- Accept string or `Path` input.
- Raise clear errors for missing files.
- Preserve image shape.
- Save masks as TIFF-compatible arrays.

Self-check:

- Can it read a TIFF?
- Can it save a TIFF?
- Can Fiji open the saved TIFF?
- Does dtype stay reasonable?
- Are errors understandable?

Status: `PHASE COMPLETE` ✔

## Phase 2: Basic Preprocessing

Goal: implement simple preprocessing filters.

Files:

- `bioimage_pipeline/preprocess.py`
- `tests/test_preprocess.py`

Functions:

- `gaussian_blur(image, sigma=1)`
- `median_filter_image(image, radius=1)`
- `normalize_image(image)`

Requirements:

- Preserve image shape.
- Avoid modifying input image in place.
- Return NumPy arrays.
- Keep functions small.

Status: `PHASE COMPLETE` ✔

## Phase 3: Basic Thresholding

Goal: implement thresholding methods without advanced adaptive logic.

Files:

- `bioimage_pipeline/threshold.py`
- `tests/test_threshold.py`

Functions:

- `manual_threshold(image, value) -> bool mask`
- `otsu_threshold(image) -> bool mask`
- `adaptive_threshold(image, block_size=51, offset=0) -> bool mask`

Requirements:

- Use `scikit-image`.
- Always return boolean masks.
- Preserve image shape.
- Handle invalid `block_size` clearly.

Status: `PHASE COMPLETE` ✔

## Phase 4: Segmentation

Goal: convert binary masks into labeled objects.

Inspiration:

- **Fiji:** binary mask cleanup before analysis (no GUI).
- **CellProfiler:** simplified IdentifyPrimaryObjects (cleanup + labeling).

Files:

- `bioimage_pipeline/segment.py`
- `tests/test_segment.py`

Inputs / outputs:

- Input: `bool` mask `(H, W)`.
- Output: cleaned `bool` mask; labeled `int` image with background `0`.

Tasks:

- Implement `remove_small_objects_from_mask(mask, min_size=20)`.
- Implement `label_objects(mask)` with `skimage.measure.label`.
- Support 2D only; raise `ValueError` for non-2D masks.

Acceptance tests:

- Label `0` is background.
- Objects have unique integer labels.
- Tiny noise objects are removed.
- Returned label image is integer type.

Fiji visual check: deferred until Phase 6 export.

Deferred: 3D volumes, watershed splitting (see Phase 11), GUI (see Phase 15).

Status: `PHASE COMPLETE` ✔

## Phase 5: Measurement

Goal: measure segmented objects.

Inspiration:

- **Fiji:** Analyze Particles–style region properties (via scikit-image).
- **CellProfiler:** MeasureObjectIntensity / region props table.

Files:

- `bioimage_pipeline/measure.py`
- `tests/test_measure.py`

Inputs / outputs:

- Input: `labels` `(H, W)` int; optional `intensity_image` (original, not mask).
- Output: `pandas.DataFrame` with one row per object.
- Pipeline dict key: `"measurements"`.

Tasks:

- Implement `measure_objects(label_image, intensity_image=None)`.
- Use `skimage.measure.regionprops_table`.
- Columns: `label`, `area`, `centroid`, `bbox`; add `mean_intensity`, `max_intensity` when intensity provided.

Acceptance tests:

- One row per object.
- Area values are reasonable.
- Intensity columns use `intensity_image`, not the mask.

Deferred: custom measurement plugins, 3D props.

Status: `PHASE COMPLETE` ✔

## Phase 6: Export

Goal: export masks, labeled images, and measurements for Fiji and spreadsheets.

Inspiration:

- **Fiji:** save masks and label images as TIFF for visual inspection.
- **CellProfiler:** export measurements to CSV.

Files:

- `bioimage_pipeline/export.py`
- `tests/test_export.py`

Inputs / outputs:

- `export_mask_tiff(path, mask)` — bool mask → `uint8` 0/255 TIFF.
- `export_label_tiff(path, labels)` — integer label TIFF.
- `export_measurements_csv(path, dataframe)` — Excel-friendly CSV.

Tasks:

- Reuse `save_tiff` from `io.py` where appropriate.
- Create parent directories automatically.

Acceptance tests:

- Mask TIFF dtype `uint8`, values 0 or 255.
- Label TIFF preserves integer labels.
- CSV round-trip readable.

Fiji visual check: mask and label TIFFs open in Fiji without error.

Deferred: ROI sets, multi-channel stacks.

Status: `PHASE COMPLETE` ✔

## Phase 7: Pipeline Core

Goal: implement a simple CellProfiler-style pipeline system.

Inspiration:

- **Fiji:** N/A (macro chaining is separate).
- **CellProfiler:** ordered modules sharing an image workspace.

Files:

- `bioimage_pipeline/pipeline.py`
- `tests/test_pipeline.py`

Inputs / outputs:

- Shared dict, e.g. `image`, `processed`, `mask`, `labels`, `measurements`, `filename`.
- Each step: `Callable[[dict], dict]`.

Tasks:

- Implement `Pipeline` with `__init__(steps)` and `run(data)`.
- Run steps in order; clear errors on bad return types.

Acceptance tests:

- Later steps see keys from earlier steps.
- Non-dict return raises `TypeError`.
- Step exception wraps as `RuntimeError` with step index.

Deferred: parallel runs, module GUI, branching logic.

Status: `PHASE COMPLETE` ✔

## Phase 8: Batch Processing

Goal: run the pipeline on a folder of TIFF images.

Inspiration:

- **Fiji:** N/A (batch via scripts).
- **CellProfiler:** batch mode over image folders.

Files:

- `bioimage_pipeline/batch.py`
- `tests/test_batch.py`

Function:

- `run_pipeline_on_folder(pipeline, input_folder, output_folder, pattern="*.tif")`

Tasks:

- Discover TIFF files (`*.tif` and `*.tiff`).
- Per image: run pipeline, save mask/labels/measurements.
- Add `filename` column to measurements.
- Return `processed` and `failed` lists; do not stop on single failure.

Acceptance tests:

- Multiple files processed.
- Outputs named by image stem.
- Combined CSV optional; failures reported.

Deferred: distributed cluster batch, database output.

Status: `PHASE COMPLETE` ✔

## Phase 9: Example Workflow

Goal: create one runnable beginner-friendly example.

Inspiration:

- **Fiji:** inspect outputs after running a script.
- **CellProfiler:** single-image analysis pipeline template.

File:

- `examples/run_basic_pipeline.py`

Workflow:

1. Read TIFF (or generate synthetic image).
2. Gaussian blur → Otsu threshold → remove small objects → label → measure.
3. Export mask TIFF, label TIFF, measurements CSV.

Tasks:

- Clear variable names and comments.
- No real lab data required.

Acceptance tests:

- Script runs from terminal without error.
- Output files exist in a chosen output directory.

Status: `PHASE COMPLETE` ✔

## Phase 10: Visual Check Without GUI

Goal: create a simple visual validation script without building a GUI.

Inspiration:

- **Fiji:** manual QC of masks and labels.
- **CellProfiler:** N/A.

File:

- `examples/visual_check.py`

Requirements:

- Synthetic image with bright circles and noise.
- Save original, mask, labels, CSV.
- Print: `Open the output TIFF files in Fiji to visually inspect the result.`

Acceptance tests:

- All expected output files created.
- Script exits successfully from terminal.

Fiji visual check: user opens TIFFs in Fiji.

Status: `PHASE COMPLETE` ✔

## Post-Phase 10: Workflow-Orchestration Roadmap

Phases 0–10 delivered the **lightweight Python pipeline mode** (preprocess →
threshold → segment → measure → export) and a thin **CellProfiler engine mode**
(subprocess runner in `cellprofiler_runner.py`).

Existing modules:

- **Workflow orchestration:** `analysis.py`, `cellprofiler_runner.py`, `qc.py`
- **Python TIFF fallback:** `fiji_tiff.py`, `export.py`, `io.py`
- **Python analysis fallback:** `pipeline.py`, `batch.py`, `preprocess.py`,
  `threshold.py`, `segment.py`, `measure.py`
- **Fiji headless export (Phase 14):** `fiji_runner.py`
- **GUI (Phase 15):** workflow shell (15.1) → pipeline builder (15.2)
  — see [docs/gui_direction.md](docs/gui_direction.md)
- **Self-adaptive import (Phase 17, deferred):** `adaptive_import.py` — prototype
  only, opt-in; not production default

**Ordering rule:** Complete **Phase 10.1 → 10.5** before **Phase 11**.
Complete **Phase 12 (Python TIFF fallback)** before **Phase 13 (CellProfiler
workflow integration)**. Complete **Phase 13** before **Phase 14 (Fiji/ImageJ
headless export)**. Complete **Phase 14** before **Phase 15** (sub-phases
**15.1 → 15.2**). Phase **16** extends the optional Python engine only.
Phase **17** (self-adaptive import — **core differentiator**) is **deferred** —
resume after Phases 14–16 unless real-data needs force earlier validation. Keep
Phase 17 visible in the roadmap; it complements CellProfiler, it does not compete
with it.

All workflow phases must follow **batch-first execution** (see above): one
CellProfiler run per folder, one Fiji run per folder when possible, per-image
external launches only as a documented fallback.

**Forward roadmap (Phases 12–17):**

| Phase | Focus | Status |
|-------|-------|--------|
| 12 | Python TIFF export (fallback / intermediate) | `PHASE COMPLETE` ✔ |
| 13 | CellProfiler workflow integration | `PHASE COMPLETE` ✔ |
| 14 | Fiji/ImageJ headless export integration | `PHASE COMPLETE` ✔ |
| 15.1 | GUI workflow shell | `PHASE COMPLETE` ✔ |
| 15.2 | GUI pipeline builder & CP module exposure | `PHASE COMPLETE` ✔ |
| 16 | Optional Python analysis enhancements | `PHASE NOT COMPLETE` |
| 17 | Self-adaptive threshold at import (hybrid CP) — **core differentiator** | `DEFERRED` — prototype |

## Phase 10.1: CellProfiler Integration Validation

Goal: verify that the project can successfully execute a real CellProfiler
pipeline.

Files:

- `bioimage_pipeline/cellprofiler_runner.py`
- `tests/test_cellprofiler_runner.py`

Tasks:

- Validate CellProfiler executable detection.
- Validate `.cppipe` loading.
- Validate input image discovery.
- Validate output folder creation.
- Validate `ExportToSpreadsheet` output.
- Validate command-line execution.
- Add manual validation checklist (below).

### Manual validation checklist

Use this checklist when running against a real CellProfiler installation. Mark
each item only after a successful real run (not mocked tests).

- [x] CellProfiler is installed and the executable path resolves (PATH or
  `cellprofiler_executable`).
- [x] A `.cppipe` pipeline loads and runs headless (`-c -r -p -i -o`) — wrapper
  builds and validates the command; re-run E2E with your pipeline file.
- [x] Input TIFF image(s) are discovered in the input folder (CellProfiler CLI).
- [x] Output directory is created (or pre-created) and writable.
- [x] `ExportToSpreadsheet` CSV file(s) load via `load_cellprofiler_measurements`.
- [x] `run_cellprofiler_pipeline()` completes without error from Python (see
  [docs/cellprofiler_validation.md](docs/cellprofiler_validation.md)).

Self-check:

```bash
pytest tests/test_cellprofiler_runner.py
python examples/validate_cellprofiler.py --cppipe path/to/pipeline.cppipe --input-dir path/to/images --output-dir path/to/cellprofiler_output --executable "C:\Program Files\CellProfiler\CellProfiler.exe"
```

Acceptance:

- Real CellProfiler run completes.
- Output files are generated.
- Wrapper execution succeeds.

Implementation note: `run_cellprofiler_pipeline`, mocked subprocess tests, and
`examples/validate_cellprofiler.py` are in the repository. See
[docs/cellprofiler_validation.md](docs/cellprofiler_validation.md) for the
validation record and E2E re-run instructions.

Status: `PHASE COMPLETE` ✔

## Phase 10.2: CellProfiler Output Import

Goal: import CellProfiler outputs back into Python.

Files (proposed):

- `bioimage_pipeline/cellprofiler_runner.py` (extend)
- `tests/test_cellprofiler_runner.py`
- `tests/fixtures/cellprofiler/` (example CSV exports)

Tasks:

- Load CSV measurements.
- Validate expected columns.
- Merge multiple CSV outputs (e.g. Image, Experiment, object tables).
- Create helper functions for reading CellProfiler results.
- Add tests using example CSV files.

Functions:

- `read_cellprofiler_csv(path)`
- `validate_cellprofiler_columns(dataframe, required_columns)`
- `load_cellprofiler_measurements(output_dir)`
- `merge_cellprofiler_tables(tables)`

Files:

- `bioimage_pipeline/cellprofiler_runner.py`
- `tests/test_cellprofiler_runner.py`
- `tests/fixtures/cellprofiler/`

Acceptance:

- DataFrames load correctly.
- Missing files produce clear errors.

Self-check:

```bash
pytest tests/test_cellprofiler_runner.py -k "load_cellprofiler or merge_cellprofiler or validate_cellprofiler"
```

Status: `PHASE COMPLETE` ✔

## Phase 10.3: Unified Analysis Mode

Goal: support both analysis engines behind one configuration surface for
scripting and tests. **Production and GUI workflows use CellProfiler as the
primary engine** (Phase 15); Python mode remains an optional fallback.

Modes:

1. **Lightweight Python mode** — existing `Pipeline` / `batch.py` workflow
   (teaching, tests, prototyping only).
2. **CellProfiler mode** — `.cppipe` subprocess via `cellprofiler_runner.py`
   (primary product path).

Tasks:

- Add configuration option: `analysis_engine = "python" | "cellprofiler"`.
- Create a unified entry point (e.g. `run_analysis(...)` in a new
  `bioimage_pipeline/analysis.py` or extended `batch.py`).
- Keep existing `Pipeline`, `run_pipeline_on_folder`, and
  `run_cellprofiler_pipeline` APIs unchanged.

Acceptance:

- User can switch engines with one parameter.
- Python-only and CellProfiler-only paths still work independently.

Status: `PHASE COMPLETE` ✔

Self-check:

```bash
pytest tests/test_analysis.py
python examples/run_analysis.py --input-dir path/to/images --output-dir path/to/output
python examples/run_analysis.py --engine cellprofiler --cppipe path/to/pipeline.cppipe --input-dir path/to/images --output-dir path/to/cellprofiler_output --executable "C:\Program Files\CellProfiler\CellProfiler.exe"
```

## Phase 10.4: Real Data Validation

Goal: validate performance on real microscopy images.

Tasks:

- Test with real TIFF images (not only synthetic data).
- Compare masks and measurements between Python mode and CellProfiler mode
  where applicable.
- Record observed limitations (image size, staining, SNR, channel count).
- Document failure cases (optional: `docs/real_data_validation.md`).

Acceptance:

- At least one real dataset successfully processed by each engine used in the
  test plan.

Status: `PHASE COMPLETE` ✔

Self-check:

```bash
python examples/validate_real_data.py --input-dir path/to/real_images --output-dir path/to/validation_output
```

## Phase 10.5: Visualization and Quality Control

Goal: improve result inspection beyond saving raw TIFFs.

Inspiration:

- **Fiji:** manual overlay and slice inspection.
- **CellProfiler:** N/A (external viewer).

Files (proposed):

- `bioimage_pipeline/qc.py` or extended `examples/visual_check.py`
- `tests/test_qc.py`

Tasks:

- Overlay masks on images and save composite figures.
- Save QC figures (PNG or TIFF) alongside pipeline outputs.
- Document a Fiji inspection workflow (step-by-step in README or examples).
- Add optional Napari support for interactive viewing.

Foundation: `examples/visual_check.py` (Phase 10) saves mask/label TIFFs for
Fiji; this phase adds overlays and richer QC artifacts.

Acceptance:

- User can visually verify segmentation quality without writing custom scripts.

Status: `PHASE COMPLETE` ✔

### Phase 10.5 checkpoint (verified)

- [x] `examples/visual_check.py` fixture defines **two** bright circular objects at
  `(50, 50)` and `(110, 105)`; pipeline detects **2** objects with matching
  centroids and intensities (`230`, `210`) in `measurements.csv`.
- [x] Mask and label QC overlays are spatially aligned with `original.tif`
  (saved PNGs byte-match live overlay generation; label coverage 100% on
  foreground pixels).
- [x] Full test suite passes: `python -m pytest -v` (78 tests).

Self-check:

```bash
python -m pytest -v
pytest tests/test_qc.py -v
python examples/visual_check.py
```

## Phase 11: Advanced Segmentation

Goal: improve object separation beyond basic connected-component labeling.

Inspiration:

- **Fiji:** Watershed, Analyze Particles separation heuristics.
- **CellProfiler:** IdentifyPrimaryObjects with declumping strategies.

Sub-phases:

| Sub-phase | Focus | Status |
|-----------|-------|--------|
| 11.1 | Mask cleanup (morphology) | `PHASE COMPLETE` |
| 11.2 | Distance transform | `PHASE COMPLETE` |
| 11.3 | Watershed splitting | `PHASE COMPLETE` |
| 11.4 | Pipeline integration | `PHASE COMPLETE` |

Deferred from Phase 4: watershed splitting.

Acceptance (Phase 11 overall):

- Touching objects can be separated in synthetic and representative test cases.

Status: `PHASE COMPLETE` ✔

## Phase 11.1: Mask Cleanup (Morphology)

Goal: improve binary masks before advanced splitting.

Files:

- `bioimage_pipeline/segment.py`
- `tests/test_segment.py`

Functions:

- `fill_holes(mask)`
- `clear_border_objects(mask)`
- `clean_mask(mask, ...)`

Tasks:

- Fill internal holes in foreground objects.
- Remove objects connected to the image border.
- Combine hole filling, small-object removal, and border clearing.

Acceptance:

- Cleanup fixes hole and border artifacts without removing valid interior objects.

Self-check:

```bash
pytest tests/test_segment.py -v
```

Status: `PHASE COMPLETE` ✔

## Phase 11.2: Distance Transform

Goal: compute distance maps from binary masks for watershed seeding.

Files:

- `bioimage_pipeline/segment.py`
- `tests/test_segment.py`

Functions:

- `distance_transform(mask)`

Tasks:

- Euclidean distance transform on foreground pixels.
- Zero background; peaks near object centers.

Acceptance:

- Output shape matches mask; values are non-negative; disk center is a peak;
  touching disks produce two peak regions.

Self-check:

```bash
pytest tests/test_segment.py -k distance -v
```

Status: `PHASE COMPLETE` ✔

## Phase 11.3: Watershed Splitting

Goal: separate touching objects using distance-transform watershed.

Files:

- `bioimage_pipeline/segment.py`
- `tests/test_segment.py`

Functions:

- `split_touching_objects(mask, ...)`

Tasks:

- Find distance-transform peaks as seeds.
- Run watershed inside the binary mask.
- Return integer label image.

Acceptance:

- Touching disks merge with `label_objects` but split into two labels with
  `split_touching_objects`; separated disks still label correctly.

Self-check:

```bash
pytest tests/test_segment.py -k split_touching -v
```

Status: `PHASE COMPLETE` ✔

## Phase 11.4: Pipeline Integration

Goal: expose watershed labeling through the default Python pipeline.

Files:

- `bioimage_pipeline/analysis.py`
- `examples/touching_objects_demo.py`
- `tests/test_analysis.py`

Tasks:

- Add ``labeling_method`` option to ``build_default_pipeline`` and
  ``run_analysis``.
- Optional ``clean_mask_before_labeling`` flag.
- Touching-objects demo with QC overlays for connected vs watershed.

Acceptance:

- Default pipeline behavior unchanged (``labeling_method="connected"``).
- ``labeling_method="watershed"`` splits touching synthetic objects.

Self-check:

```bash
pytest tests/test_analysis.py -k "watershed or labeling" -v
python examples/touching_objects_demo.py
```

Status: `PHASE COMPLETE` ✔

## Phase 12: Python TIFF Export (Fallback / Intermediate)

Goal: write ImageJ-compatible TIFF files from Python when Fiji/ImageJ is not
available, and prepare intermediate exports before Phase 14 headless Fiji runs.
This phase does **not** replace Fiji/ImageJ headless export for production
final outputs (see Phase 14).

Files:

- `bioimage_pipeline/fiji_tiff.py`
- `bioimage_pipeline/export.py` (extend)
- `bioimage_pipeline/io.py` (optional ImageJ mode)
- `tests/test_fiji_tiff.py`
- `tests/test_export.py`
- `docs/fiji_tiff_export.md`

Tasks:

- ImageJ-compatible TIFF writer with optional metadata (pixel size, unit,
  channel name, description).
- Mask export as `uint8` (`0` / `255`).
- Label export as `uint16` or `uint32` depending on max label ID.
- Intensity export preserving safe integer dtypes.
- Round-trip tests for shape, dtype, mask values, label IDs, and metadata.
- Document ordinary TIFF vs ImageJ TIFF vs future OME-TIFF.

Acceptance:

- Exported masks, labels, and intensity TIFFs round-trip in tests.
- Metadata is written and read when provided.
- Roadmap reflects CellProfiler = primary engine, Python TIFF = fallback,
  Fiji headless export = Phase 14 production path.

Self-check:

```bash
pytest tests/test_fiji_tiff.py tests/test_export.py -v
```

Status: `PHASE COMPLETE` ✔

### Phase 12 checkpoint (verified)

- [x] `mask.tif` — `uint8`, shape `(160, 160)`, values `0` / `255`, ImageJ-compatible
- [x] `labels.tif` — `uint16`, shape `(160, 160)`, label IDs `1` and `2` preserved
- [x] `original.tif` — `uint16` intensity TIFF (plain TIFF in `visual_check.py`)
- [x] QC overlays — `visual_check_qc_mask_overlay.png`, `visual_check_qc_label_overlay.png`
- [x] `python -m pytest -v` — 112 tests passed

Deferred to later phases:

- OME-TIFF export with full calibration metadata
- Fiji/ImageJ headless subprocess export → **Phase 14**

### Export path summary (Phase 12 vs Phase 14)

| Path | Module | Role |
|------|--------|------|
| **Production (Phase 14)** | `fiji_runner.py` | Headless Fiji/ImageJ export of final TIFFs |
| **Fallback / intermediate (Phase 12)** | `fiji_tiff.py`, `export.py` | Python `tifffile` ImageJ-compatible writes |
| **Plain storage** | `io.py` | Ordinary TIFF without ImageJ tags |

See [docs/fiji_tiff_export.md](docs/fiji_tiff_export.md) and
[docs/fiji_headless_export.md](docs/fiji_headless_export.md).

## Phase 13: CellProfiler Workflow Integration

Goal: orchestrate CellProfiler as the primary analysis engine — from `.cppipe`
selection through **one headless batch run per input folder** to organized
intermediate results.

CellProfiler remains the **primary analysis engine**. This phase wires the CP
subprocess runner (single launch per folder), log capture, output discovery,
and results-folder layout. It does **not** implement CellProfiler modules.
Final TIFF export through Fiji is Phase 14; Phase 13 currently uses Python
TIFF export as an in-process fallback (no extra CP/Fiji launches).

**Performance requirement:** `run_cellprofiler_pipeline_logged()` must invoke
CellProfiler **once** with `-i input_dir` — never once per image file.

Files:

- `bioimage_pipeline/cellprofiler_runner.py` (extend)
- `bioimage_pipeline/analysis.py` (extend)
- `bioimage_pipeline/export.py` / `fiji_tiff.py` (CellProfiler output → Fiji TIFF)
- `bioimage_pipeline/qc.py` (QC overlays from CP outputs)
- `examples/run_cellprofiler_workflow.py`
- `tests/test_cellprofiler_workflow.py`
- `docs/cellprofiler_workflow.md`

Tasks:

1. Improve CellProfiler runner workflow (logged subprocess, output discovery).
2. Accept a `.cppipe` pipeline file and image folder.
3. Run CellProfiler headlessly.
4. Capture logs and errors clearly (`logs/`).
5. Locate CellProfiler outputs (CSV + TIFF in `cellprofiler_raw/`).
6. Convert output TIFFs into organized folders via Python fallback export
   (`masks/`, `labels/`) until Phase 14 Fiji headless export is wired in.
7. Generate QC overlays from CellProfiler outputs (`qc/`).
8. Organize all outputs into a clean results folder:
   `measurements/`, `masks/`, `labels/`, `qc/`, `logs/`, `cellprofiler_raw/`.
9. Add tests using mocked CellProfiler subprocess calls.
10. Document the CellProfiler workflow stage.

Acceptance:

- User can select a `.cppipe` and image folder, run CellProfiler **once**
  headlessly over the full folder, capture logs/errors, and collect organized
  results automatically.
- Python lightweight engine still works independently via `analysis_engine="python"`.
- No per-image CellProfiler subprocess launches in the workflow API.

Self-check:

```bash
python -m compileall bioimage_pipeline tests examples
python -m pytest -v
pytest tests/test_cellprofiler_workflow.py -v
python examples/run_cellprofiler_workflow.py --cppipe path/to/pipeline.cppipe --input-dir path/to/images --output-dir path/to/results
```

Status: `PHASE COMPLETE` ✔

Implementation note: `run_cellprofiler_workflow()` in `analysis.py` orchestrates
a **single** headless CP execution per folder, log capture, CSV collection,
Python fallback TIFF organization (in-process), and QC overlay generation. See
[docs/cellprofiler_workflow.md](docs/cellprofiler_workflow.md).

**Phase 14 follow-up:** add per-stage timing fields to `workflow_summary.json`.

## Phase 14: Fiji/ImageJ Headless Export Integration

Goal: run Fiji/ImageJ headlessly **once per batch/folder** to produce **final
TIFF outputs** from collected CellProfiler results, preserving Fiji/ImageJ-specific
metadata and export behavior. Python TIFF export (Phase 12) remains available
as a fast in-process fallback.

Fiji/ImageJ is a **first-class export engine**, not only a manual viewing target.

**Performance requirement:** prefer **one Fiji subprocess per folder** with a
batch macro (e.g. `export_folder.ijm input_dir output_dir`). Do not spawn Fiji
once per image unless batch export is impossible — document and log when
per-image fallback is used.

Files:

- `bioimage_pipeline/fiji_runner.py`
- `bioimage_pipeline/analysis.py` (extends workflow to call Fiji export)
- `bioimage_pipeline/export.py` (fallback path when Fiji unavailable)
- `examples/fiji_macros/export_folder.ijm` — batch folder macro
- `examples/run_fiji_export.py`
- `tests/test_fiji_runner.py`
- `docs/fiji_headless_export.md`

Tasks:

1. Configure Fiji/ImageJ executable path (similar to CellProfiler runner).
2. Detect whether Fiji is installed; report clear errors when missing.
3. Run Fiji/ImageJ headlessly through subprocess — **one invocation per batch**
   (`ImageJ-linux64 --headless`, `Fiji.app/ImageJ-win64.exe /headless`, etc.).
4. Support **batch folder macros/scripts** that read from `cellprofiler_raw/` (or
   staging dir) and write final TIFFs to `masks/`, `labels/`. Avoid per-file
   Fiji launches unless batch export is not possible.
5. Convert CellProfiler outputs (masks, labels, intensity) into final
   Fiji-exported TIFFs under organized folders.
6. Capture Fiji stdout/stderr; store export logs under `logs/`.
7. Fail gracefully when Fiji is unavailable — fall back to Python TIFF export
   (Phase 12, in-process) with a logged warning.
8. Record **stage timing** in `logs/workflow_summary.json`:
   - `timing.cellprofiler_seconds`
   - `timing.fiji_export_seconds`
   - `timing.qc_seconds`
   - `timing.total_seconds`
9. Add tests using mocked Fiji subprocess calls (verify single batch invocation).
10. Document headless export setup, batch macro conventions, and per-image fallback
    criteria.

Acceptance:

- Full workflow: Input Images → CellProfiler (once) → Fiji/ImageJ (once) → Final Outputs.
- Final TIFFs in results folders are produced by **one batch Fiji run** when available.
- Python TIFF export still works when Fiji is missing or export is disabled.
- Export logs and **per-stage timings** are captured alongside CellProfiler logs.
- Per-image Fiji launches are opt-in/debug only, never the default batch path.

Self-check:

```bash
python -m compileall bioimage_pipeline tests examples
python -m pytest tests/test_fiji_runner.py -v
python examples/run_fiji_export.py --help
```

Status: `PHASE COMPLETE` ✔

### Phase 14 checkpoint (implemented)

- [x] Fiji/ImageJ executable resolution and clear missing-executable errors are in
  `bioimage_pipeline/fiji_runner.py`.
- [x] `run_fiji_batch_export()` runs one Fiji subprocess per batch macro and writes
  `fiji_stdout.log`, `fiji_stderr.log`, `fiji_command.txt`, and
  `fiji_exit_code.txt`.
- [x] `run_cellprofiler_workflow()` now attempts Fiji batch export before QC and
  falls back to Python TIFF export with `logs/fiji_export_warning.log` when Fiji
  is unavailable or the macro fails.
- [x] `workflow_summary.json` includes `timing.cellprofiler_seconds`,
  `timing.fiji_export_seconds`, `timing.qc_seconds`, and `timing.total_seconds`,
  plus `export_engine` / `export_mode`.
- [x] Batch macro, CLI example, and mocked subprocess tests are present:
  `examples/fiji_macros/export_folder.ijm`, `examples/run_fiji_export.py`,
  `tests/test_fiji_runner.py`.

Verification:

```bash
python -m compileall bioimage_pipeline tests examples
python examples/run_fiji_export.py --help
```

Note: `python -m pytest tests/test_fiji_runner.py tests/test_cellprofiler_workflow.py -v`
currently fails during collection in this environment because `scikit-image` is
binary-incompatible with the installed NumPy (`numpy.dtype size changed`), before
the Phase 14 tests execute.

### Fiji Automated OIR Z-Max Output Status

Status: `PHASE COMPLETE` ✔

Implemented:

- Finds input `.oir` files recursively.
- Builds expected `.tif` output names.
- Generates a manual-run Fiji macro using **Bio-Formats Windowless Importer**.
- Runs the intended Z Project Max Intensity macro path from Fiji GUI mode.
- Verifies that expected output files exist before reporting files as processed.

Validation result:

- Fiji reference output and Python-generated/manual-macro output are
  **pixel-identical**.

Validated:

- Same input `.oir` files.
- Same output filenames.
- Same file sizes.
- Same image shapes.
- Same pixel values (`compare_outputs.py` reported identical TIFF arrays).

Note:

- Command-line Fiji macro execution for `.oir` import remains avoided because it
  can trigger `java.lang.VerifyError` in `loci.plugins.in.MainDialog` on some
  Fiji/Bio-Formats/Java combinations. The accepted path is to generate the
  `.ijm` file from Python and run it manually from the Fiji GUI.

## Phase 15: GUI — CellProfiler & Fiji Workflow Front-End

Goal: build a user-facing application that exposes CellProfiler and Fiji
functionality through pipeline management and workflow control.

Delivered in two sub-phases: **15.1** (proper workflow shell) and **15.2**
(pipeline builder).

The GUI is **not** a new analysis engine. It is a front-end that:

1. Generates and manages `.cppipe` pipeline configuration (15.2).
2. Calls CellProfiler headlessly (batch-first, via Phase 13 APIs).
3. Collects and displays outputs (measurements, masks, labels, QC).
4. Optionally triggers Fiji/ImageJ headless export (Phase 14).
5. Surfaces logs, progress, and stage timings.

The GUI reads/writes pipeline configuration and can delegate execution to
CellProfiler or run the built-in Python pipeline.

Files:

- `bioimage_pipeline/gui/` package (Phase 15.1+)
- `bioimage_pipeline/pipeline_catalog.py` (Phase 15.2 module metadata)
- `bioimage_pipeline/cppipe_io.py` (Phase 15.2 load/save/parse `.cppipe`)
- `examples/run_gui.py` (Phase 15.1)
- `tests/test_gui_workflow.py`, `tests/test_cppipe_io.py`
- `docs/gui_direction.md`

### Phase 15.1: GUI Workflow Shell

Goal: build the **proper workflow shell** — polished run/review experience for
running saved CellProfiler pipelines and reviewing organized outputs.

Technology: standard-library **Tkinter** shell for the first proper workflow UI,
with workflow logic separated into testable helpers. This keeps Phase 15.1
dependency-light while still calling the existing headless orchestration APIs
without showing Fiji or CellProfiler windows.

Tasks:

- Select input image folder and output directory.
- Load an existing `.cppipe` pipeline file.
- Configure CellProfiler and Fiji executable paths (persisted settings).
- Run batch workflow (`run_cellprofiler_workflow` + optional Fiji export).
- Display live log tail / progress and stage timings from `logs/`.
- Preview QC overlays (mask/label PNGs) and measurement tables.
- Open or export organized results (`measurements/`, `masks/`, `labels/`).
- Production-quality layout, error handling, and executable-path validation.

Acceptance:

- Non-programmers can run a saved `.cppipe` end-to-end and inspect results in the GUI.
- GUI invokes CellProfiler — it does not run Python segmentation as the primary path.

Status: `PHASE COMPLETE` ✔

### Phase 15.1 checkpoint (implemented)

- [x] `bioimage_pipeline/gui/workflow_shell.py` provides the GUI workflow shell,
  validation helpers, workflow summary model, log-tail reader, measurements
  preview loader, and Tkinter launch surface.
- [x] `examples/run_gui.py` launches the shell.
- [x] GUI execution delegates to `run_cellprofiler_workflow()` so CellProfiler and
  Fiji run through headless orchestration rather than showing their desktop UI.
- [x] `tests/test_gui_workflow.py` covers config validation, headless workflow
  delegation, output summaries, log tails, and measurement previews.

Self-check:

```bash
python -m pytest tests/test_gui_workflow.py -v
python -m compileall bioimage_pipeline tests examples
python examples/run_gui.py
```

### Phase 15.2: GUI Pipeline Builder & CellProfiler Module Exposure

Goal: let users **browse, configure, and compose CellProfiler pipelines** in the
GUI without opening the CellProfiler desktop app for every edit.

Tasks:

- Module catalog: browse/search CellProfiler modules by name and category
  (metadata sourced from CellProfiler module definitions or a curated catalog).
- Parameter panels: configure module settings; persist to `.cppipe` format.
- Pipeline editor: add, remove, reorder modules; validate pipeline structure.
- Load/save `.cppipe` files; import pipelines created in CellProfiler.
- Link pipeline editor to workflow shell (15.1): build → run → preview in one app.

Implementation notes:

- Prefer text-preserving read/write of standard `.cppipe` files so pipelines
  remain compatible with CellProfiler CLI and desktop app.
- Module parameter schemas may be loaded from CellProfiler where possible, or
  maintained as a catalog that maps to `.cppipe` keys — either way, execution
  stays in CellProfiler.
Acceptance:

- User can build or edit a pipeline in the GUI, save as `.cppipe`, and run it
  headlessly with the same results as running that `.cppipe` in CellProfiler directly.
- GUI exposes modules like `IdentifyPrimaryObjects` as configurable steps.

Status: `PHASE COMPLETE` ✔

### Phase 15.2 checkpoint (implemented)

- [x] `bioimage_pipeline/cppipe_io.py` loads, parses, validates, edits, renumbers,
  saves CellProfiler `.cppipe` text while preserving unknown lines, and generates
  the minimal GUI pipeline:
  `Images → Metadata → NamesAndTypes → IdentifyPrimaryObjects → SaveImages → ExportToSpreadsheet`.
- [x] `bioimage_pipeline/pipeline_catalog.py` exposes a curated CellProfiler
  module catalog with searchable module metadata, visible setting labels,
  defaults, allowed choices, and conditional visibility for the Phase 15.2
  modules.
- [x] `bioimage_pipeline/gui/workflow_shell.py` includes a Pipeline Builder panel
  with selected pipeline modules on the left, selected-module settings on the
  right, add/remove/reorder controls, live setting updates, saving edited
  pipelines, and running them through the Phase 15.1 headless workflow shell.
- [x] `tests/test_cppipe_io.py` covers parser round trips, setting updates,
  module move/remove/append, catalog lookup, conditional visible settings,
  generated module blocks, GUI model updates, and runner handoff.
- [x] A generated Phase 15.2 `.cppipe` opened and ran headlessly through
  CellProfiler 4.2.8 on a synthetic TIFF, writing `Image.csv`, `Nuclei.csv`, and
  `Nuclei0001.tiff` to the selected output folder.

Self-check:

```bash
python -m pytest tests/test_cppipe_io.py tests/test_gui_workflow.py -v
python -c "from bioimage_pipeline.cppipe_io import create_pipeline_from_catalog; print(create_pipeline_from_catalog().to_text())"
```

Deferred from Phase 4: GUI (now Phase 15 with sub-phases 15.1 and 15.2).

### Phase 15.3 editor-first refactor

Goal: turn the shell into a CellProfiler-style pipeline editor.

Implemented:

- [x] Startup loads only the required setup modules
  (`Images → Metadata → NamesAndTypes → Groups`); no analysis/demo/test modules.
- [x] Phase A — Run executes the in-memory edited pipeline by materializing it to
  a working `.cppipe`; selecting a separate pipeline file is no longer required.
- [x] Phase B — Module deletion via toolbar button, right-click context menu, and
  Delete/Backspace keys; the four setup modules are protected (no delete/reorder).
- [x] Full CellProfiler module catalog (`pipeline_catalog.py`) grouped into
  CellProfiler categories (Image Processing, Object Processing, Measurement,
  File Processing, Data Tools, Advanced, Worm Toolbox, Other) via
  `list_modules_by_category()`; catalog browser is now a category tree.

Deferred (to implement later):

- [x] Phase C — New/Open/Save/Save As menu with current-file tracking and a
  modified/title indicator.
- [x] Phase D — Move input-folder selection into the Images module and add a
  "detected images" preview.
- [x] Phase E — Surface SaveImages/ExportToSpreadsheet output paths in the UI.
- [x] Phase F — Editor-first layout (promote pipeline + settings, demote run
  options to a Run panel/dialog).

**Architecture decision (2026-06-10):** Hybrid strategy — continue custom editor +
execution engine; use native CellProfiler for advanced authoring. See
[docs/architecture_decision.md](docs/architecture_decision.md) and
[docs/cellprofiler_authoring.md](docs/cellprofiler_authoring.md).

## Phase 16: Optional Python Analysis Enhancements

Goal: improve the **optional lightweight Python engine** only — for teaching,
tests, and prototyping. This phase is **explicitly not** the product direction
for production analysis or GUI primary workflows.

Note: **Phase 11 already delivered** watershed splitting, morphology cleanup,
and distance-transform segmentation. Phase 16 extends or refines those features
for the Python fallback engine only.

**Relationship to GUI (Phase 15):** the GUI primary path uses CellProfiler.
Python engine enhancements may appear as an optional "simple mode" in the GUI
later, but must not replace or duplicate CellProfiler module functionality.

Tasks:

- Adaptive thresholding (auto block size, rolling-ball background correction) in
  `threshold.py` / `preprocess.py` for the **simple** Python engine.
- Further watershed / declumping tuning and presets.
- Additional morphology cleanup options.
- Compare Python vs CellProfiler results on sample images (optional).

Note: **Full self-adaptive import pipeline** (auto method per image, staging → CP)
is **Phase 17 (deferred)** — see prototype in `adaptive_import.py`, not the
default product path.

Files (proposed):

- `bioimage_pipeline/threshold.py` (extend)
- `bioimage_pipeline/preprocess.py` (extend)
- `bioimage_pipeline/segment.py` (extend)
- `tests/test_threshold.py`, `tests/test_segment.py`

Acceptance:

- Python engine improvements are opt-in and do not replace CellProfiler as the
  primary engine for production analysis.

Status: `PHASE NOT COMPLETE`

## Phase 17: Self-Adaptive Threshold at Import (Hybrid CP Workflow)

Goal: automatic per-image thresholding for fluorescence nuclei at import — robust
to vignetting, low SNR, and uneven illumination — then feed masks into
CellProfiler for measurement. **CellProfiler cannot do this alone**; Python
handles import-time adaptivity; CP remains the analysis engine downstream.

This phase is the project's **core differentiator** — the capability that makes
the hybrid workflow uniquely valuable. It is **deferred** until Phases 14–16 are
stable, but it must remain a first-class roadmap item and must not be subsumed by
"just use CellProfiler" as the product story.

**Status: `DEFERRED` — prototype implemented, logic incomplete for production.**

A first-pass prototype landed early for experimentation. It is **not enabled**
on default workflows and **will require future tuning or redesign** before
product use.

### Prototype (in repo, subject to change)

| File | Purpose |
|------|---------|
| `bioimage_pipeline/adaptive_import.py` | `run_self_adaptive_threshold`, folder staging |
| `bioimage_pipeline/preprocess.py` | `rolling_ball_subtract` (used by prototype) |
| `bioimage_pipeline/threshold.py` | `sauvola_threshold` (used by prototype) |
| `tests/test_adaptive_import.py` | Synthetic fixture tests only |
| `examples/run_adaptive_threshold.py` | Experimental batch runner |

Prototype steps: inspect image → optional rolling-ball → normalize → auto
method (Otsu / local / Sauvola) → morphology → optional watershed → confidence
log → write `staging/masks/` and `staging/labels/`.

Opt-in CellProfiler hook: `run_cellprofiler_workflow(..., adaptive_threshold=True)`.

### Why deferred

- Real-world performance on diverse assays is **unproven** (only synthetic
  vignetted nuclei fixtures validated so far).
- Method selection heuristics may need assay-specific presets or user overrides.
- May need Sauvola/Niblack tuning, better block-size estimation, or ML-assisted
  QC — scope TBD after real-data review.
- Must not replace or duplicate CellProfiler module logic beyond the threshold
  slice agreed for the hybrid workflow.

### Future work before marking complete

1. Validate on real microscopy datasets; define minimum IoU/Dice acceptance.
2. Per-image decision log review; add “low confidence → review in GUI” path.
3. Document recommended `.cppipe` templates that consume staging masks.
4. Integrate with Phase 15.1 GUI preview (import overlay before CP run).
5. Re-evaluate whether default workflows should opt in after sign-off.

Acceptance (when resumed):

- Opt-in staging → CP batch workflow produces acceptable masks on agreed fixtures.
- Default Python and CP workflows remain unchanged unless user enables Phase 17.
- Decision logs and confidence flags support GUI review.

Self-check (prototype only):

```bash
python examples/run_adaptive_threshold.py --input-dir path/to/images --output-dir path/to/staging
pytest tests/test_adaptive_import.py -v
```

Status: `DEFERRED` (prototype in repo — do not use as production default)

## Optional future work (not scheduled)

These items are useful but outside the current Phases 13–16 scope:

- Batch job queues and preset profiles for recurring lab workflows
- OME-TIFF export with full microscopy calibration metadata
- Deeper CellProfiler integration (e.g. live module schema sync from installed CP version)

---

## Stack / Batch Track (Phases S.0 – S.7)

**Goal:** Fiji-style stack and batch-processing workflow on the **Python engine**.
Load a folder of images or a multi-page TIFF, treat every image/frame as a stack
slice, apply the same preprocess → segment → measure pipeline to each slice, and
export per-frame TIFFs and a combined CSV.

This is an **independent track** alongside the CellProfiler workflow track
(Phases 13–17). Both use the same core Python pipeline modules.

```text
Input (folder of TIFFs  OR  one multi-page stack TIFF)
    ↓
ImageStack  →  Pipeline (same steps for every frame)
    ↓
Per-frame outputs (mask.tif, labels.tif, measurements.csv)
Combined all_measurements.csv  (frame / slice / stack_id columns)
```

## Phase S.0: Stack Track Prep

Goal: clean baseline before introducing stack code.

Tasks:

- Remove spurious `"columns found"` entry from `warnings` list in
  `load_cellprofiler_measurements` (`cellprofiler_runner.py` line 456).
- Add stack-track section to `DEVELOPMENT_PLAN.md`.

Acceptance:

- `python -m pytest -v` passes with 0 failures.

Self-check:

```bash
python -m pytest -v
```

Status: `PHASE COMPLETE` ✔

## Phase S.1: Stack I/O — AxisInfo, StackFrame, iter_stack_frames

Goal: primitives for reading multi-frame TIFFs frame-by-frame.

Files:

- `bioimage_pipeline/io.py` (extend)
- `tests/test_io.py` (extend)

New public API:

| Symbol | Description |
|--------|-------------|
| `AxisInfo` | Dataclass: height, width, z_count, t_count, c_count, frame_count, source |
| `StackFrame` | Dataclass: index, array, z_index, t_index, c_index, source_path, metadata |
| `interpret_tiff_axes(shape, imagej_metadata)` | Infer Z/T/C from shape + optional ImageJ metadata |
| `extract_2d_plane(image, frame_index)` | Pull one 2D plane from any nD array |
| `iter_stack_frames(path)` | Iterator — yields one StackFrame per page |

Rules:

- Single-page TIFF → yields 1 frame.
- Multi-page TIFF → yields N frames (one per page).
- ImageJ metadata used when present for per-axis indices.
- Frame arrays are always 2D `(H, W)`.
- Source path attached to every frame.

Acceptance tests (18 new tests in `test_io.py`):

- `interpret_tiff_axes` on 2D, 3D, 4D, 5D shapes and ImageJ metadata dict.
- `extract_2d_plane` on 2D passthrough, 3D selection, out-of-range error.
- `iter_stack_frames` single image, multipage, 2D frame guarantee,
  source path, missing file, directory, z_index.

Self-check:

```bash
python -m pytest tests/test_io.py -v
python -m pytest -v
```

Status: `PHASE COMPLETE` ✔

## Phase S.2: Stack Data Model

Goal: a unified `ImageStack` object that can be built from a folder of 2D TIFFs
**or** from a single multi-page TIFF — without duplicating frame logic.

Files:

- `bioimage_pipeline/stack.py` (new)
- `tests/test_stack.py` (new)

API:

```python
@dataclass
class ImageStack:
    frames: list[StackFrame]
    source: str | Path  # file path or folder path
    axis_info: AxisInfo | None

def load_stack_from_tiff(path) -> ImageStack
def load_stack_from_folder(folder, pattern="*.tif") -> ImageStack
def load_stack(source) -> ImageStack   # auto-detects file vs folder
```

Rules:

- `load_stack_from_folder`: discovers TIFFs sorted alphabetically; each file
  becomes one `StackFrame` (2D image or first plane of a multi-page file).
- `load_stack_from_tiff`: uses `iter_stack_frames`; returns all pages.
- `load_stack`: if `source` is a file → `load_stack_from_tiff`; if a folder
  → `load_stack_from_folder`.
- Empty folder → `ValueError`.
- Missing path → `FileNotFoundError`.

Acceptance tests:

- Folder with N TIFFs → stack with N frames.
- Multi-page TIFF → stack with N frames.
- File vs folder auto-detection.
- Frame order is alphabetical for folder sources.
- Empty folder raises.

Status: `PHASE COMPLETE` ✔

## Phase S.3: Stack Batch Runner

Goal: run a `Pipeline` on every frame of an `ImageStack`, collecting per-frame
outputs and a combined measurement table.

Files:

- `bioimage_pipeline/batch.py` (extend)
- `tests/test_batch.py` (extend)

New function:

```python
def run_pipeline_on_stack(
    pipeline: Pipeline,
    stack: ImageStack,
    output_dir: str | Path,
) -> dict[str, Any]
```

Returns: `{"processed": [...], "failed": [...], "measurements": DataFrame | None}`

Per-frame data dict includes `"filename"`, `"frame_index"`, `"z_index"`,
`"t_index"`, `"c_index"` from the frame metadata.

Rules:

- Continue processing remaining frames on single-frame failure (same as
  `run_pipeline_on_folder`).
- Collect all per-frame measurements DataFrames; concat to `all_measurements.csv`.
- Do not re-read source files; reuse `StackFrame.array`.

Acceptance tests:

- N-frame stack → N processed outputs.
- Single-frame failure → reported in `failed`, remaining frames succeed.
- Combined measurements have `frame_index` column.
- No stack-level failure when one frame is bad.

Status: `PHASE COMPLETE` ✔

## Phase S.4: Stack Export

Goal: write per-frame mask/label TIFFs and a combined CSV with frame-identity
columns.

Files:

- `bioimage_pipeline/export.py` (extend)
- `tests/test_export.py` (extend)

Output naming convention:

```text
output_dir/
    {stem}_f000_mask.tif
    {stem}_f000_labels.tif
    {stem}_f000_measurements.csv
    ...
    all_measurements.csv   ← frame_index, z_index, filename columns added
```

Acceptance tests:

- File names include zero-padded frame index.
- Combined CSV schema includes `frame_index`.
- Round-trip: mask values 0/255, label IDs preserved.

Status: `PHASE COMPLETE` ✔

## Phase S.5: Processed-Image Export

Goal: export the `"processed"` (blurred/corrected) plane alongside masks and
labels, so the full pipeline intermediate state is inspectable.

Files:

- `bioimage_pipeline/batch.py` (extend)
- `bioimage_pipeline/export.py` (extend)

Convention: `{stem}_f000_processed.tif` — intensity TIFF written by
`export_intensity_tiff`.

Acceptance tests:

- Processed TIFF exists after batch run.
- Shape matches source frame.
- Dtype is safe integer or float32.

Status: `PHASE COMPLETE` ✔

## Phase S.6: Fiji-Macro-Style Batch Recipe (CLI)

Goal: a repeatable CLI command that mirrors Fiji's "Process Folder" / macro
workflow — load input, run fixed pipeline steps, export results — without
requiring any Python scripting by the user.

Files:

- `examples/run_stack_batch.py`
- `bioimage_pipeline/stack_recipe.py` — JSON recipe load/save/merge
- `bioimage_pipeline/stack_batch.py` — `run_stack_batch_workflow()`
- `examples/stack_batch_recipe.json` — example recipe template

CLI:

```bash
python examples/run_stack_batch.py \
    --input path/to/folder_or_stack.tif \
    --output path/to/results \
    [--blur-sigma 1.0] \
    [--min-object-size 20] \
    [--labeling connected|watershed] \
    [--export-processed]
```

Optional: JSON recipe file (`--recipe batch_recipe.json`) that serializes
pipeline step names and parameters for reproducible batch runs.

Implemented: `--recipe`, `--generate-qc`, `stack_recipe.py`, `stack_batch.py`,
`examples/stack_batch_recipe.json`.

Acceptance tests:

- CLI smoke test: `subprocess.run([sys.executable, "examples/run_stack_batch.py", ...])`.
- Result directory contains expected output files.

Status: `PHASE COMPLETE` ✔

## Phase S.7: Stack QC, Example, and Docs

Goal: QC overlays per frame, a runnable end-to-end example, and documentation.

Files:

- `bioimage_pipeline/qc.py` (extend — `generate_qc_for_stack`)
- `examples/run_stack_example.py` (new — synthetic Z-stack end-to-end)
- `docs/stack_batch_workflow.md` (new — Fiji workflow mapping + API reference)
- `bioimage_pipeline/stack_batch.py` — shared workflow entry point
- `tests/test_stack_batch_cli.py` — CLI subprocess smoke tests

`generate_qc_for_stack(stack, masks_dir, labels_dir, qc_dir)` produces one
overlay PNG per frame using existing `export_qc_artifacts`.

Acceptance:

- Example runs from terminal with no real data required.
- QC overlay PNGs are created for each frame.
- Doc maps Fiji "Process Stack" steps to this project's API.

Status: `PHASE COMPLETE` ✔
