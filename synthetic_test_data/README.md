# Synthetic Puncta Validation Data

This folder holds **synthetic** microscopy images used to validate the Gaussian/GMM
puncta pipeline. It is separate from real laboratory images.

## Install dependencies

From the project root:

```powershell
pip install -e .
```

Required packages (already listed in `pyproject.toml`):

- `numpy`
- `scipy`
- `tifffile`

## Generate synthetic cases

From the project root:

```powershell
python scripts\generate_synthetic_puncta.py
```

Generate one case:

```powershell
python scripts\generate_synthetic_puncta.py --case case3_overlapping
```

List built-in cases:

```powershell
python scripts\generate_synthetic_puncta.py --list-cases
```

Preview the full batch grid (manifest only):

```powershell
python scripts\generate_synthetic_puncta.py --batch --dry-run
```

Generate a small batch subset (default: 5 conditions):

```powershell
python scripts\generate_synthetic_puncta.py --batch --batch-limit 5
```

## Output layout

For each case `{case_name}`:

| Path | File | Meaning |
|------|------|---------|
| `images/{case_name}/` | `synthetic_noisy.tif` | Simulated raw image (Poisson + read noise) — **pipeline input** |
| `images/{case_name}/` | `synthetic_clean.tif` | Noise-free signal (background + Gaussians) |
| `masks/{case_name}/` | `synthetic_mask.tif` | Binary external mask (0/255) — **pipeline input** |
| `ground_truth/{case_name}/` | `synthetic_true_seeds.tif` | One labeled pixel per true punctum center |
| `ground_truth/{case_name}/` | `synthetic_ground_truth.json` | Known spot count, coordinates, sigmas, generation params |

Pipeline run outputs go under `results/{case_name}/` (gitignored).

## Coordinate system

Ground truth uses the same convention as `{stem}_measurements.csv`:

- **`x`** = column (`final_col`, `x_fit`)
- **`y`** = row (`final_row`, `y_fit`)

## Run the existing puncta pipeline

Do **not** modify production puncta code for validation. Pass the noisy image and mask:

```powershell
python examples/run_puncta_declump.py `
  --input "synthetic_test_data/images/case1_isolated/synthetic_noisy.tif" `
  --mask "synthetic_test_data/masks/case1_isolated/synthetic_mask.tif" `
  --output-dir "synthetic_test_data/results/case1_isolated" `
  --stem "case1_isolated" `
  --candidate-detector python_log `
  --diagnostic-mode summary `
  --no-fiji-tiffs
```

For overlapping cases (merged mask, multiple true spots), consider forcing full fitting:

```powershell
python examples/run_puncta_declump.py `
  --input "synthetic_test_data/images/case3_overlapping/synthetic_noisy.tif" `
  --mask "synthetic_test_data/masks/case3_overlapping/synthetic_mask.tif" `
  --output-dir "synthetic_test_data/results/case3_overlapping" `
  --stem "case3_overlapping" `
  --no-selective-routing `
  --candidate-detector python_log `
  --diagnostic-mode summary `
  --no-fiji-tiffs
```

## Compare predictions to ground truth

1. Open `ground_truth/{case_name}/synthetic_ground_truth.json` for true spot count and `(x, y)` positions.
2. Open `results/{case_name}/{stem}_measurements.csv` and filter rows where `accepted == True`.
3. Compare:
   - **Count accuracy**: `len(accepted)` vs `true_spot_count`
   - **Position error**: nearest-neighbor distance from each true spot to an accepted `(final_col, final_row)` within tolerance (e.g. 1 px)

A future `scripts/compare_synthetic_puncta.py` may automate this step.

## Built-in validation cases

| Case | True spots | Separation | Mask | Expected |
|------|------------|------------|------|----------|
| `case1_isolated` | 1 | — | 1 object | predicted count = 1 |
| `case2_separated` | 2 | 6 px | 2 objects | predicted count = 2 |
| `case3_overlapping` | 2 | 2 px | 1 merged object | predicted count = 2 (GMM declump) |

## Batch grid (future systematic benchmark)

The generator defines a grid varying:

- spot count: 1, 2, 3
- center separation: 1–6 px
- sigma: 1.5, 2.0, 2.5, 3.0
- brightness ratio: 1:1, 1:1.5, 1:2
- noise: low, medium, high
- random seeds: 3 per condition

Use `--batch --dry-run` to inspect the full manifest without generating thousands of files.

## Evaluate pipeline results

After running the puncta pipeline on synthetic cases, compare predictions to ground truth:

```powershell
python scripts\evaluate_synthetic_puncta.py --evaluate-basic
python scripts\evaluate_synthetic_puncta.py --separation-benchmark
python scripts\evaluate_synthetic_puncta.py --run-pipelines --evaluate-basic
```

Outputs:

- `results/synthetic_validation_summary.csv`
- `results/synthetic_validation_summary.json`
- `results/separation_benchmark_summary.csv` (when separation benchmark is evaluated)

Pass criterion (default 2 px tolerance): exact count match, zero false positives/false negatives, all matched centers within tolerance.

Case 3 is evaluated twice against the same ground truth (`case3_overlapping`):

- `case3_overlapping_normal` — selective routing enabled
- `case3_overlapping_forced_gmm` — `--no-selective-routing`

Generate separation benchmark cases (2 spots, merged mask, separation 1–6 px):

```powershell
python scripts\generate_synthetic_puncta.py --separation-benchmark
```

Separation benchmark pipeline runs use forced GMM (`--no-selective-routing`) to measure fitting resolution independent of routing.

## GMM debugging and ablation (synthetic)

Focused instrumentation lives outside the production puncta path in
`bioimage_pipeline/puncta/validation/gmm_probe.py`. It records per-model
diagnostics (n=1/2/3), multi-start initialization strategies, and filter
ablation modes without changing runtime behavior of `run_puncta_declump`.

Run on the four basic validation cases:

```powershell
python scripts\run_gmm_synthetic_debug.py --basic-cases --print-case3-report
```

Run a single case:

```powershell
python scripts\run_gmm_synthetic_debug.py --case case3_overlapping_forced_gmm --print-case3-report
```

Outputs per run folder (`results/{run_name}/`):

| File | Contents |
|------|----------|
| `gmm_model_diagnostics.json` | Full per-model records (init, fit params, RSS, R², AIC/BIC, merge notes, rejection reasons) |
| `gmm_model_diagnostics.csv` | Flattened summary of model attempts |

Global ablation summary: `results/gmm_ablation_summary.json`

Direct fitting unit tests (no router, no LoG, progressive layer re-enable):

```powershell
python -m pytest tests/test_gmm_direct_fit.py -q
```

Key diagnostic fields:

- `model_selection.balanced_model_attempted_n2` — whether production `select_balanced_model` attempts n=2 (independent of peak count)
- `model_selection.candidate_component_counts_legacy_select_best` — legacy path tied to peak count
- `model_selection.exact_second_component_rejection` — precise rejection reason (not a generic summary)
- `model_attempts[].model_level_rejection_reason` — e.g. `post_merge_collapsed_2_to_1`
- `model_attempts[].component_rejection_reasons` — e.g. `component_2:duplicate_center_too_close`

Ablation modes compared for overlapping cases:

- `detector_based` — production-style init from detected peaks only
- `symmetric_two_component` — symmetric offset init
- `multi_start_best` — best BIC across all init strategies
- `multi_start_no_filters` — multi-start with acceptance filters disabled
- `multi_start_duplicate_only` — only duplicate-distance filter enabled

## Calibrating from real puncta (later)

When tuning synthetic parameters to match laboratory data, measure from real runs:

- median fitted `sigma` from `{stem}_measurements.csv`
- typical spot amplitude above local background
- local background mean and standard deviation
- approximate signal-to-noise ratio
- typical punctum diameter (`single_spot_max_diameter`)
- image bit depth

Plug those values into `scripts/generate_synthetic_puncta.py` CLI flags (`--background`, `--read-noise`, spot amplitudes in case JSON, etc.).
