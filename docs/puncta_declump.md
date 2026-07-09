# Puncta Declumping Workflow

This document describes the Python prototype for detecting and declumping tiny fluorescent puncta using local background correction, local maxima detection, elliptical Gaussian fitting, and Gaussian mixture model (GMM) selection.

## Overview

The pipeline treats each punctum as a Gaussian-like intensity peak in the raw grayscale image. Thresholding produces a coarse foreground mask; maxima detection and Gaussian fitting always run on the **raw intensity image** (after local background correction inside each object ROI), not on the binary mask.

```text
raw grayscale image
  → threshold mask (or external mask)
  → connected-component labeling
  → per-object background correction (ring around mask)
  → local maxima detection (all objects)
  → route: single elliptical Gaussian OR joint GMM (large / multi-peak)
  → model comparison (1 vs M±1 components via BIC)
  → quality filtering + fit_status labeling
  → CSV + overlay + seed image + optional diagnostic PNGs
```

Thresholding finds candidate mask regions only. Gaussian fitting refines spot count and subpixel coordinates afterward.

## Quick start

```bash
pip install -e ".[dev]"

python examples/run_puncta_declump.py \
  --input path/to/image.tif \
  --output-dir results/puncta \
  --single-spot-max-diameter 7 \
  --threshold-method otsu
```

With an external mask:

```bash
python examples/run_puncta_declump.py \
  --input path/to/image.tif \
  --mask path/to/mask.tif \
  --output-dir results/puncta
```

For Z-stacks or multi-page TIFFs, select a plane with `--frame-index`:

```bash
python examples/run_puncta_declump.py \
  --input path/to/stack.tif \
  --mask path/to/mask_stack.tif \
  --output-dir results/puncta \
  --frame-index 0
```

Run tests:

```bash
pytest tests/test_puncta_declump.py -v
```

## Per-object workflow

For each connected mask object:

1. **Local background correction** — estimate background from a ring around the object and subtract it inside an expanded ROI patch.
2. **Local maxima** — detect reliable peaks on the corrected patch (even for small objects).
3. **Routing**
   - **Single path** — one peak and object diameter ≤ `single_spot_max_diameter`: fit one elliptical Gaussian.
   - **GMM path** — two or more reliable peaks, or object larger than `single_spot_max_diameter`: joint mixture fit with BIC model selection among 1, M−1, M, and M+1 components.
4. **Validation** — sigma bounds, amplitude, center shift, center inside mask, duplicate separation, relative residual, R².
5. **Fallback** — if fitting fails and `accept_brightest_on_fit_failure` is true, record brightest-pixel coordinates with `fit_status=fit_failed_fallback` (no subpixel fitted coordinates).

## Key parameters

| Parameter | Default | Purpose |
|---|---|---|
| `single_spot_max_diameter` | 7.0 | Objects above this diameter use the GMM path |
| `min_reliable_peaks_for_gmm` | 2 | Two or more maxima also trigger GMM, even if the object is small |
| `expected_single_spot_diameter` | 5.0 | Guides initial Gaussian sigma guess |
| `background_ring_width` | 3 | Ring width (px) for local background estimation |
| `background_margin` | 4 | Extra margin around object for background ring |
| `smoothing_sigma` | 0.75 | Mild blur before local maxima detection |
| `min_peak_distance` | 3 | Minimum spacing between maxima (pixels) |
| `peak_noise_tolerance` | 0.0 | Added to local median for maxima threshold |
| `fit_roi_radius` | 5 | Half-size of square ROI for single-component fits |
| `min_sigma` / `max_sigma` | 0.5 / 4.0 | Allowed fitted spot spread |
| `max_center_shift` | 4.0 | Reject fits whose center moves too far from the seed |
| `min_amplitude` | 10.0 | Minimum fitted peak amplitude above background |
| `max_fit_residual_relative` | 0.25 | Max fit RMSE / amplitude (scale-invariant) |
| `min_r_squared` | 0.3 | Minimum coefficient of determination |
| `gmm_max_components` | 5 | Maximum mixture components per object |
| `min_center_separation` | 3.0 | Minimum distance between accepted puncta |

## Outputs

Each run writes:

- `puncta_measurements.csv` — per-component measurements and `fit_status`
- `puncta_summary.json` — object counts, fit quality medians, diagnostic paths
- `puncta_seeds.tif` — accepted centers as point labels
- `puncta_mask.tif` — foreground mask
- `puncta_labels.tif` — connected mask objects
- `puncta_overlay.png` — accepted puncta with fit-quality coloring
- `diagnostics/` (when enabled) — corrected / predicted / residual PNGs for suspicious fits

Use `--show-rejected` to also draw rejected candidates in red.

### CSV columns (selected)

| Column | Meaning |
|---|---|
| `fit_status` | `fit_ok`, `fit_failed_fallback`, `rejected_*` |
| `path` | `single`, `gmm`, or `fallback` |
| `component_id` | Index within the object (1 for single-spot objects) |
| `x_fit` / `y_fit` | Subpixel fitted coordinates (null for fallback) |
| `sigma_x` / `sigma_y` | Elliptical Gaussian widths (columns / rows) |
| `r_squared` / `model_score` | Goodness of fit |
| `n_components_in_model` | Components in the selected mixture model |

Only rows with `fit_status=fit_ok` should be treated as true Gaussian subpixel coordinates.

### fit_status values

| Status | Meaning |
|---|---|
| `fit_ok` | Accepted Gaussian fit with subpixel coordinates |
| `fit_failed_fallback` | Fit failed; brightest pixel used instead |
| `rejected_bad_fit` | Failed validation (residual, shift, R², etc.) |
| `rejected_duplicate` | Too close to another accepted punctum |
| `rejected_outside_mask` | Fitted center outside object mask |
| `rejected_low_amplitude` | Amplitude below threshold |

### Overlay legend

| Visual | Meaning |
|---|---|
| Green cross + circle | `fit_ok` Gaussian fit |
| Cyan line | Seed-to-fit center shift (>0.25 px) |
| Yellow cross | `fit_failed_fallback` (brightest pixel) |
| Red cross | Rejected candidate (`--show-rejected`) |

## Python API

```python
import numpy as np
from bioimage_pipeline.puncta import PunctaDeclumpConfig, run_puncta_declump

image = ...  # 2D grayscale array
config = PunctaDeclumpConfig(single_spot_max_diameter=7.0)
result = run_puncta_declump(image, config, diagnostics_dir="results/diagnostics")

for punctum in result.gaussian_fitted:
    print(punctum.fitted_row, punctum.fitted_col, punctum.sigma_row, punctum.sigma_col)
```

## Design notes

- Mask objects gate ROIs only; fitting uses background-corrected raw intensity.
- Small objects with multiple reliable peaks still enter the GMM path.
- Large single-peak objects run BIC model selection (1 vs M components) on the GMM path.
- Fallback coordinates are explicitly marked and do not populate `fitted_row` / `fitted_col`.
- This prototype is intended as a reference for a future native Fiji Java plugin.

## Related modules

- `bioimage_pipeline/puncta/background.py` — local background correction
- `bioimage_pipeline/puncta/object_processor.py` — per-object GMM workflow
- `bioimage_pipeline/puncta/gaussian_fitter.py` — elliptical + mixture fitting, BIC selection
- `bioimage_pipeline/puncta/candidate_filter.py` — validation and deduplication
- `bioimage_pipeline/puncta/diagnostics.py` — residual / fit diagnostic images
- `bioimage_pipeline/puncta/pipeline.py` — orchestrator
