# Puncta Declumping Workflow

This document describes the Python prototype for detecting and declumping tiny fluorescent puncta using size-gated local maxima detection and 2D circular Gaussian fitting.

## Overview

The pipeline treats each punctum as a Gaussian-like intensity peak in the raw grayscale image. Thresholding produces a coarse foreground mask; maxima detection and Gaussian fitting always run on the **raw intensity image**, not on the binary mask.

```text
raw grayscale image
  → threshold mask (or external mask)
  → connected-component labeling
  → size gate (single vs clump path)
  → local maxima + Gaussian fitting (clumps)
  → quality filtering
  → CSV + overlay + seed image export
```

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

If the mask stack uses a different slice index:

```bash
python examples/run_puncta_declump.py \
  --input path/to/stack.tif \
  --mask path/to/mask_stack.tif \
  --output-dir results/puncta \
  --frame-index 2 \
  --mask-frame-index 2
```

Run tests:

```bash
pytest tests/test_puncta_declump.py -v
```

## Key parameters

| Parameter | Default | Purpose |
|---|---|---|
| `single_spot_max_diameter` | 7.0 | Objects at or below this equivalent diameter use the single-punctum path |
| `expected_single_spot_diameter` | 5.0 | Guides initial Gaussian sigma guess |
| `smoothing_sigma` | 0.75 | Mild blur before local maxima detection inside clumps |
| `min_peak_distance` | 3 | Minimum spacing between maxima (pixels) |
| `peak_noise_tolerance` | 0.0 | Added to local median for maxima threshold; increase to suppress weak peaks |
| `fit_roi_radius` | 5 | Half-size of square ROI used for Gaussian fitting |
| `min_sigma` / `max_sigma` | 0.5 / 4.0 | Allowed fitted spot spread |
| `max_center_shift` | 3.0 | Reject fits whose center moves too far from the seed maximum |
| `min_amplitude` | 10.0 | Minimum fitted peak amplitude above background |
| `max_fit_residual_relative` | 0.25 | Max fit RMSE / amplitude (scale-invariant for 16-bit images) |
| `max_fit_residual` | (none) | Optional absolute RMSE cap; omit for intensity-scale invariance |
| `min_center_separation` | 3.0 | Minimum distance between accepted puncta |

Measure obvious single puncta in Fiji first, then set `expected_single_spot_diameter` and `single_spot_max_diameter` slightly above the typical single-spot size.

## Outputs

Each run writes:

- `puncta_measurements.csv` — per-candidate measurements and acceptance status
- `puncta_summary.json` — object counts and threshold metadata
- `puncta_seeds.tif` — accepted centers as point labels (for watershed / CellProfiler)
- `puncta_mask.tif` — foreground mask
- `puncta_labels.tif` — connected mask objects
- `puncta_overlay.png` — accepted puncta shown as green crosses

Use `--show-rejected` to also draw rejected candidates in red.

### Overlay legend

The overlay PNG now shows whether Gaussian fitting worked:

| Visual | Meaning |
|---|---|
| Green cross + green circle | Gaussian fit accepted (circle ≈ FWHM spot width) |
| Cyan line | Seed-to-fit center shift (>0.25 px) |
| Yellow cross | Brightest-pixel fallback (no Gaussian fit) |
| Red cross | Rejected candidate (only with `--show-rejected`) |

Check `puncta_summary.json` → `fit_quality` for counts and median σ.

## Viewing in Fiji

1. Open the original image in Fiji.
2. Open `puncta_overlay.png` as a reference, or import `puncta_seeds.tif` and use **Process > Binary > Options** if needed for point display.
3. Open `puncta_measurements.csv` in Excel or Fiji's table tools for inspection.

For downstream seeded splitting, use `puncta_seeds.tif` as seed markers with watershed or IdentifySecondaryObjects-style workflows.

## Python API

```python
import numpy as np
from bioimage_pipeline.puncta import PunctaDeclumpConfig, run_puncta_declump

image = ...  # 2D grayscale array
config = PunctaDeclumpConfig(single_spot_max_diameter=7.0)
result = run_puncta_declump(image, config)

for punctum in result.accepted:
    print(punctum.final_row, punctum.final_col, punctum.sigma)
```

## Design notes

- Small objects skip declumping and use the brightest pixel as the seed.
- Large objects detect intensity maxima inside the mask, fit a circular Gaussian at each peak, and reject poor fits.
- If no maxima survive in a large object, the pipeline falls back to the brightest pixel with a warning.
- This prototype is intended as a reference for a future native Fiji Java plugin.

## Related modules

- `bioimage_pipeline/puncta/threshold_mask.py` — mask generation
- `bioimage_pipeline/puncta/maxima_detector.py` — intensity peak detection
- `bioimage_pipeline/puncta/gaussian_fitter.py` — 2D circular Gaussian fitting
- `bioimage_pipeline/puncta/pipeline.py` — orchestrator
