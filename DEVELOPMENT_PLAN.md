# Development Plan

This project is a lightweight Python bioimage analysis package inspired by
Fiji/ImageJ and CellProfiler. It does not import or copy code from either
project.

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
| 10.3 | Unified analysis mode | `PHASE NOT COMPLETE` |
| 10.4 | Real data validation | `PHASE NOT COMPLETE` |
| 10.5 | Visualization and QC | `PHASE NOT COMPLETE` |
| 11 | Advanced segmentation | `PHASE NOT COMPLETE` |
| 12 | Adaptive thresholding improvements | `PHASE NOT COMPLETE` |
| 13 | GUI | `PHASE NOT COMPLETE` |

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

Deferred: 3D volumes, watershed splitting (see Phase 11), GUI (see Phase 13).

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
  `segment.py`, `measure.py`, `export.py`
- **CellProfiler mode:** `cellprofiler_runner.py`

**Ordering rule:** Complete **Phase 10.1 → 10.5** in order before starting
**Phase 11**. Do not skip to advanced thresholding (**Phase 12**) until
CellProfiler integration, unified mode, real-data validation, and QC are in place.

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

Status: `PHASE NOT COMPLETE`

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

Status: `PHASE NOT COMPLETE`

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

Status: `PHASE NOT COMPLETE`

## Phase 11: Advanced Segmentation

Goal: improve object separation beyond basic connected-component labeling.

Inspiration:

- **Fiji:** Watershed, Analyze Particles separation heuristics.
- **CellProfiler:** IdentifyPrimaryObjects with declumping strategies.

Files (proposed):

- `bioimage_pipeline/segment.py` (extend)
- `tests/test_segment.py`

Tasks:

- Watershed segmentation on distance transforms.
- Distance transform from binary masks.
- Touching object splitting.
- Improved object cleanup (morphology, hole filling, border clearing).

Deferred from Phase 4: watershed splitting.

Acceptance:

- Touching objects can be separated in synthetic and representative test cases.

Status: `PHASE NOT COMPLETE`

## Phase 12: Adaptive Thresholding Improvements

Goal: improve threshold robustness across varied image types. Do not start until
Phases 10.1–11 are complete.

Inspiration:

- **Fiji:** Auto Threshold, rolling-ball background correction.
- **CellProfiler:** advanced threshold modules.

Files (proposed):

- `bioimage_pipeline/threshold.py` (extend)
- `bioimage_pipeline/preprocess.py` (background correction)
- `tests/test_threshold.py`

Tasks:

- Auto parameter selection (block size, offset, sensitivity).
- Background correction before thresholding.
- Rolling-ball subtraction.
- Multi-scale thresholding.

Deferred examples (from former Phase 11 plan):

- Auto-tuned block size / offset.
- Rolling-ball or morphological background subtraction before threshold.
- Multi-scale or learned thresholds.

Acceptance:

- Improved performance across at least two image types (e.g. high vs low
  background, uneven illumination).

Status: `PHASE NOT COMPLETE`

## Phase 13: GUI

Goal: create a user-facing interface for non-programmers.

Tasks:

- File and folder selection.
- Parameter controls (threshold, min object size, engine choice).
- Pipeline execution with progress feedback.
- Result preview (mask overlay, measurement table).
- Export controls (TIFF, CSV, output directory).

Acceptance:

- Non-programmers can run the workflow without editing Python scripts.

Deferred from Phase 4: GUI.

Status: `PHASE NOT COMPLETE`
