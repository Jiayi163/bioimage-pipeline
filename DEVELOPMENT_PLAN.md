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

Status: `PHASE NOT COMPLETE`

## Phase 4: Segmentation

Goal: convert binary masks into labeled objects.

Files:

- `bioimage_pipeline/segment.py`
- `tests/test_segment.py`

Functions:

- `remove_small_objects_from_mask(mask, min_size=20) -> bool mask`
- `label_objects(mask) -> labeled image`

Requirements:

- Background should be `0`.
- Each object should have a unique integer label.
- Support 2D images first.

Status: `PHASE NOT COMPLETE`

## Phase 5: Measurement

Goal: measure segmented objects.

Files:

- `bioimage_pipeline/measure.py`
- `tests/test_measure.py`

Function:

- `measure_objects(label_image, intensity_image=None) -> pandas.DataFrame`

Measurements:

- `label`
- `area`
- `centroid`
- `bbox`
- `mean_intensity` if `intensity_image` is provided
- `max_intensity` if `intensity_image` is provided

Requirements:

- Use `skimage.measure.regionprops_table`.
- Return a pandas `DataFrame`.
- Each object should be one row.
- Intensity measurements must come from the original image.

Status: `PHASE NOT COMPLETE`

## Phase 6: Export

Goal: export masks, labeled images, and measurements.

Files:

- `bioimage_pipeline/export.py`
- `tests/test_export.py`

Functions:

- `export_mask_tiff(path, mask)`
- `export_label_tiff(path, labels)`
- `export_measurements_csv(path, dataframe)`

Requirements:

- Boolean masks should save as `0/255` `uint8` TIFF.
- Labeled images should save as integer TIFF.
- CSV should be readable by Excel.
- Create output folders if needed.

Status: `PHASE NOT COMPLETE`

## Phase 7: Pipeline Core

Goal: implement a simple CellProfiler-style pipeline system.

Files:

- `bioimage_pipeline/pipeline.py`
- `tests/test_pipeline.py`

Design:

Each pipeline step receives and returns a dictionary.

Example data dictionary:

```python
{
    "image": image,
    "processed": processed_image,
    "mask": mask,
    "labels": labels,
    "measurements": dataframe,
}
```

Class:

- `Pipeline`

Methods:

- `__init__(steps)`
- `run(data)`

Status: `PHASE NOT COMPLETE`

## Phase 8: Batch Processing

Goal: run the pipeline on a folder of TIFF images.

Files:

- `bioimage_pipeline/batch.py`
- `tests/test_batch.py`

Function:

- `run_pipeline_on_folder(pipeline, input_folder, output_folder, pattern="*.tif")`

Requirements:

- Process all TIFF images in a folder.
- Save output masks.
- Save one CSV per image or one combined CSV.
- Track image filename in results.
- Do not crash the entire batch if one image fails.

Status: `PHASE NOT COMPLETE`

## Phase 9: Example Workflow

Goal: create one runnable beginner-friendly example.

File:

- `examples/run_basic_pipeline.py`

Workflow:

1. Read TIFF.
2. Gaussian blur.
3. Otsu threshold.
4. Remove small objects.
5. Label objects.
6. Measure objects.
7. Save mask TIFF.
8. Save CSV.

Requirements:

- Easy for a beginner to read.
- Use clear variable names.
- Include comments.
- Do not require real lab data.
- Optionally generate a synthetic test image.

Status: `PHASE NOT COMPLETE`

## Phase 10: Visual Check Without GUI

Goal: create a simple visual validation script without building a GUI.

Requirements:

- Generate a synthetic image with bright circular objects and noise.
- Save the original image as TIFF.
- Run the basic pipeline.
- Save mask TIFF.
- Save labeled TIFF.
- Save measurements CSV.
- Print this instruction:
  `"Open the output TIFF files in Fiji to visually inspect the result."`

Self-check:

- Does the script run from terminal?
- Are TIFF files created?
- Can the TIFF files be opened in Fiji?
- Does the mask show the expected objects?
- Does the CSV contain object measurements?

Expected result:

- `PHASE COMPLETE` only if the example creates all expected output files.

Status: `PHASE NOT COMPLETE`
