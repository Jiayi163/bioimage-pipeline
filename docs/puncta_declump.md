# Puncta Declumping Workflow

This document describes the selective puncta declumping pipeline: cheap geometry screening for all objects, image-level candidate detection once per image, Gaussian fitting only on suspicious objects, optional watershed splitting from accepted centers, and limited diagnostic PNG export.

## Architecture

```text
raw grayscale image
  → threshold mask (or external mask)
  → connected-component labeling
  → image-level candidate detection ONCE (Python LoG default)
  → assign peaks to objects by mask containment
  → route each object:
        ordinary  → fast path (assigned peak or brightest pixel, no Gaussian fit)
        suspicious → patch background correction + Gaussian/GMM fit
  → optional watershed split using accepted fit centers
  → CSV + JSON + labels + optional diagnostic PNGs (suspicious only by default)
```

**Performance principle:** expensive fitting runs only on a small suspicious subset. Ordinary small, round, isolated objects stay on the fast path.

## Candidate detectors

One detector per run (`--candidate-detector`):

| Mode | Fiji calls | Notes |
|---|---|---|
| `python_log` (default) | 0 | In-memory DoG/LoG + `peak_local_max` |
| `fiji_find_maxima` | 1 per image or batch | ImageJ Find Maxima macro |
| `trackmate` | 1 per image or batch | TrackMate LoG via headless Groovy |
| `comparison` | 1–N Fiji | Benchmark all detectors; primary = Python LoG |

Candidate coordinates are cached under `{output_dir}/.puncta_cache/` keyed by source path, mtime, detector name, and settings. Use `--force-redetect` to bypass cache.

## Quick start

Single image:

```bash
python examples/run_puncta_declump.py \
  --input path/to/image.tif \
  --output-dir results/puncta \
  --candidate-detector python_log \
  --single-spot-max-diameter 7 \
  --threshold-method otsu
```

Batch folder:

```bash
python examples/run_puncta_batch.py \
  --input-dir path/to/images \
  --output-dir results/puncta_batch \
  --candidate-detector python_log
```

With external mask:

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

Legacy behavior (fit every object):

```bash
python examples/run_puncta_declump.py \
  --input path/to/image.tif \
  --output-dir results/puncta \
  --no-selective-routing
```

Run tests:

```bash
pytest tests/test_puncta_declump.py tests/test_puncta_selective.py -v
```

## Routing rules

**Ordinary (fast path)** when all of:
- `equivalent_diameter <= single_spot_max_diameter`
- `area <= pi * (single_spot_max_diameter/2)^2 * ordinary_area_factor`
- fewer than `min_reliable_peaks_for_routing` separated assigned peaks (default 3)
- shape/solidity flags alone do **not** trigger routing

**Suspicious** when any of:
- 3+ separated assigned peaks
- 2 separated peaks on oversized or irregular objects
- `equivalent_diameter > single_spot_max_diameter`
- `area` above ordinary max
- manual diagnostic object ID

On real puncta masks, expect roughly **10–20% suspicious** after balanced routing (not 80%+).

## Key parameters

| Parameter | Default | Purpose |
|---|---|---|
| `candidate_detector` | `python_log` | Image-level peak detector |
| `enable_selective_routing` | true | Fast path for ordinary objects |
| `min_reliable_peaks_for_routing` | 3 | Separated peak count → suspicious |
| `single_spot_max_diameter` | 7.0 | Size gate for fast path |
| `ordinary_area_factor` | 2.0 | Max ordinary area vs max single-spot area |
| `gmm_max_components` | 3 | Default max mixture components |
| `gmm_max_components_large` | 5 | Max components for large objects |
| `gmm_bic_improvement_margin` | 2.0 | Early-stop BIC threshold |
| `gmm_aic_improvement_margin` | 2.0 | Early-stop AIC threshold |
| `enable_watershed_declump` | true | Split multi-center objects post-fit |
| `diagnostic_mode` | `balanced` | PNG export policy |

## Timing metrics

Each run records stage timings in `{stem}_summary.json`:

- `preprocessing_time`
- `connected_component_time`
- `candidate_detection_time`
- `gaussian_fit_time`
- `watershed_time`
- `diagnostic_export_time`
- `total_time`
- `number_of_objects`, `number_of_suspicious_objects`, `number_of_fitted_objects`

## Outputs

| File | Description |
|---|---|
| `{stem}_measurements.csv` | All candidates with routing/fit debug columns |
| `{stem}_summary.json` | Counts, fit quality, timing, detector info |
| `{stem}_labels.tif` | Label image (watershed-updated when applicable) |
| `{stem}_seeds.tif` | Accepted center seeds |
| `{stem}_overlay.png` | Visual QC overlay |
| `diagnostics/` | Limited PNG panels (balanced mode) |

## TrackMate integration

TrackMate is optional/benchmark. Requires Fiji with TrackMate installed. Uses one Fiji subprocess per image (or batch macro for folders). Spot CSV is parsed into the shared coordinate table; peaks are assigned to connected components in Python — no per-object Fiji calls.

Reference macro: `examples/fiji_macros/trackmate_log_detect.groovy`

## Expected performance

| Configuration | Relative speed |
|---|---|
| Python LoG + selective routing | Fastest (default) |
| Fiji Find Maxima batch | Moderate |
| TrackMate batch | Moderate to slower |
| Fit every object (`--no-selective-routing`) | Slowest (legacy) |
