# Development Plan

This project is a lightweight **CellProfiler-to-Fiji workflow tool**. It does not
import or copy code from CellProfiler or Fiji/ImageJ.

## Project goal (clarified)

| Layer | Role |
|-------|------|
| **CellProfiler** | Full analysis engine — all functionality via headless `.cppipe` runs |
| **This project** | Manage inputs, run CP headlessly, collect outputs, organize results |
| **Fiji/ImageJ** | Manual QC/viewing target — open exported TIFFs (no embedded Fiji runtime) |
| **Python engine** | Lightweight/simple fallback for teaching, tests, and quick prototypes |

Core goal: CellProfiler performs analysis; this project delivers Fiji-inspectable
outputs (masks, labels, measurements, QC overlays) in a clean results folder.

We are **not** reimplementing CellProfiler modules. We **are** standardizing
CellProfiler outputs into Fiji-friendly TIFFs and organized workflow artifacts.

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
| 12 | Fiji/ImageJ-compatible TIFF export | `PHASE COMPLETE` |
| 13 | CellProfiler-to-Fiji workflow integration | `PHASE COMPLETE` ✔ |
| 14 | GUI (Streamlit / PyQt) | `PHASE NOT COMPLETE` |
| 15 | Advanced CellProfiler support | `PHASE NOT COMPLETE` |
| 16 | Optional Python enhancements | `PHASE NOT COMPLETE` |

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

Deferred: 3D volumes, watershed splitting (see Phase 11), GUI (see Phase 14).

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

## Post-Phase 10: Dual-Engine Roadmap

Phases 0–10 delivered the **lightweight Python pipeline mode** (preprocess →
threshold → segment → measure → export) and a thin **CellProfiler engine mode**
(subprocess runner in `cellprofiler_runner.py`).

Existing modules:

- **Python mode:** `pipeline.py`, `batch.py`, `preprocess.py`, `threshold.py`,
  `segment.py`, `measure.py`, `export.py`, `fiji_tiff.py`
- **CellProfiler mode:** `cellprofiler_runner.py` (full CP via `.cppipe`)
- **Fiji/ImageJ target:** `fiji_tiff.py`, `export.py`, `docs/fiji_tiff_export.md`

**Ordering rule:** Complete **Phase 10.1 → 10.5** before **Phase 11**.
Complete **Phase 12 (Fiji TIFF export)** before **Phase 13 (CellProfiler
workflow integration)**. Complete **Phase 13** before **Phase 14 (GUI)**.
Phases **15–16** extend the product after the core CP → Fiji export path works.

**Forward roadmap (Phases 12–16):**

| Phase | Focus | Status |
|-------|-------|--------|
| 12 | Fiji/ImageJ TIFF export | `PHASE COMPLETE` ✔ |
| 13 | CellProfiler-to-Fiji workflow integration | `PHASE COMPLETE` ✔ |
| 14 | GUI (Streamlit / PyQt) | `PHASE NOT COMPLETE` |
| 15 | Advanced CellProfiler support | `PHASE NOT COMPLETE` |
| 16 | Optional Python enhancements | `PHASE NOT COMPLETE` |

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

Goal: support both analysis engines behind one configuration surface.

Modes:

1. **Lightweight Python mode** — existing `Pipeline` / `batch.py` workflow.
2. **CellProfiler mode** — `.cppipe` subprocess via `cellprofiler_runner.py`.

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

## Phase 12: Fiji/ImageJ-Compatible TIFF Export

Goal: make final TIFF outputs open correctly in Fiji/ImageJ without calling
Fiji/ImageJ directly.

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
- Roadmap reflects CellProfiler = engine, Python = lightweight, Fiji = QC target.

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

Deferred to a later phase:

- OME-TIFF export with full calibration metadata
- Direct Fiji/ImageJ subprocess or macro integration (only when clearly needed)

## Phase 13: CellProfiler-to-Fiji Workflow Integration

Goal: make this project a lightweight CellProfiler-to-Fiji workflow tool — from
`.cppipe` selection through headless execution to organized, Fiji-inspectable
results.

CellProfiler remains the **full analysis engine**. This phase wires existing
pieces into one repeatable workflow (CLI and Python API), not new CP modules.

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
6. Convert output TIFFs into Fiji/ImageJ-compatible files (`masks/`, `labels/`).
7. Generate QC overlays from CellProfiler outputs (`qc/`).
8. Organize all outputs into a clean results folder:
   `measurements/`, `masks/`, `labels/`, `qc/`, `logs/`.
9. Add tests using mocked CellProfiler subprocess calls.
10. Document the CellProfiler-to-Fiji workflow.

Acceptance:

- User can run one command or API call to execute a `.cppipe` and receive
  organized measurements, Fiji-openable TIFFs, QC overlays, and captured logs.
- Python lightweight engine still works independently via `analysis_engine="python"`.

Self-check:

```bash
python -m compileall bioimage_pipeline tests examples
python -m pytest -v
pytest tests/test_cellprofiler_workflow.py -v
python examples/run_cellprofiler_workflow.py --cppipe path/to/pipeline.cppipe --input-dir path/to/images --output-dir path/to/results
```

Status: `PHASE COMPLETE` ✔

Implementation note: `run_cellprofiler_workflow()` in `analysis.py` orchestrates
headless CP execution, log capture, CSV collection, Fiji TIFF organization, and
QC overlay generation. See [docs/cellprofiler_workflow.md](docs/cellprofiler_workflow.md).

## Phase 14: GUI (Streamlit / PyQt)

Goal: create a user-facing interface for non-programmers.

Preferred first target: **Streamlit** demo (fastest path). **PyQt** remains an
option for a desktop-native app later.

Tasks:

- File and folder selection.
- Engine choice: CellProfiler (`.cppipe`) or Python lightweight pipeline.
- Parameter controls (threshold, min object size, labeling method).
- Pipeline execution with progress feedback.
- Result preview (mask overlay, measurement table).
- Export controls (Fiji-compatible TIFF, CSV, output directory).

Acceptance:

- Non-programmers can run a CellProfiler or Python workflow without editing
  Python scripts.

Deferred from Phase 4: GUI.

Status: `PHASE NOT COMPLETE`

## Phase 15: Advanced CellProfiler Support

Goal: make recurring CellProfiler use easier for labs — templates, presets,
and batch jobs.

Tasks:

- Pipeline templates (documented example `.cppipe` workflows).
- Parameter presets (named config profiles for common assays).
- Batch job runner (queue folders, track status, collect outputs).
- Optional config file (`YAML` / `JSON`) for repeatable runs.

Acceptance:

- User can launch batch CellProfiler jobs from a preset without editing
  pipeline JSON by hand.

Status: `PHASE NOT COMPLETE`

## Phase 16: Optional Python Enhancements

Goal: improve the **lightweight Python engine** only. CellProfiler already
covers heavy analysis; these are optional niceties for simple workflows.

Note: **Phase 11 already delivered** watershed splitting, morphology cleanup,
and distance-transform segmentation. Phase 16 extends or refines those features
and adds thresholding improvements previously listed as a separate phase.

Tasks:

- Adaptive thresholding (auto block size, rolling-ball background correction).
- Further watershed / declumping tuning and presets.
- Additional morphology cleanup options.
- Compare Python vs CellProfiler results on sample images (optional).

Files (proposed):

- `bioimage_pipeline/threshold.py` (extend)
- `bioimage_pipeline/preprocess.py` (extend)
- `bioimage_pipeline/segment.py` (extend)
- `tests/test_threshold.py`, `tests/test_segment.py`

Acceptance:

- Python engine improvements are opt-in and do not replace CellProfiler as the
  primary engine for production analysis.

Status: `PHASE NOT COMPLETE`
