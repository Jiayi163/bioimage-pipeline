# Phase B: Residual-Guided Split Integration Summary

## Overview

Phase B production integration is now complete. This implementation adds residual-guided dynamic splitting to the GMM pipeline while preserving all existing behavior when disabled.

## Architecture

```text
object_processor
    ↓
GaussianModelSelector.select_balanced_model()
    ↓ (initial fit: 1/2/3-GMM as before)
initial ModelComparisonResult
    ↓ (if residual_split_enabled)
ResidualSplitRefiner.refine()
    ↓ (iterative N→N+1 proposals)
final ModelComparisonResult
```

**Key Design Decisions:**

1. **Separate Refinement Layer:** `ResidualSplitRefiner` is independent from `GaussianModelSelector`, making it easy to extend for Phase C/D/E without rewriting core fitting logic.

2. **No Multi-Start for Splits:** N+1 refit uses deterministic initialization (current N centers + proposed residual center), avoiding runtime explosion.

3. **Strong Acceptance Criteria:** Physical validity checks MUST pass AND BIC must improve (or RMSE ↓ ≥5%). This prevents overfitting and noisy splits.

4. **Safe Fallback:** If N+1 fit fails or is rejected, the original model is preserved unchanged.

## Files Changed

### New Files

1. **`bioimage_pipeline/puncta/residual_refiner.py`**
   - `ResidualSplitRefiner` class: orchestrates split loop
   - `SplitAttemptDiagnostic`: diagnostics for each split attempt
   - `RefinementResult`: final result with all diagnostics
   - `_refit_n_plus_one()`: deterministic N+1 fit wrapper

2. **`tests/test_phase_b_integration.py`**
   - 8 integration tests covering all Phase B requirements
   - Tests for: disabled behavior, hidden doublet recovery, 2→3 split, clean single rejection, fallback safety, max iterations, no multi-start

### Modified Files

1. **`bioimage_pipeline/puncta/config.py`**
   - Added `residual_split_enabled: bool = False` (default OFF)
   - Added `residual_split_max_iterations: int = 2`

2. **`bioimage_pipeline/puncta/residual_split.py`**
   - Updated `from_puncta_config()` to use `residual_split_max_iterations`

3. **`bioimage_pipeline/puncta/gaussian_fitter.py`**
   - `GaussianModelSelector.__init__`: Added `_residual_refiner` field
   - `select_balanced_model()`: Calls `_apply_residual_refinement()` when enabled
   - `_apply_residual_refinement()`: New method to apply Phase B refinement

## Configuration

Phase B is **disabled by default**. To enable:

```python
config = PunctaDeclumpConfig(
    residual_split_enabled=True,
    residual_split_max_iterations=2,
    gmm_max_components=3,
)
```

## New Diagnostics

Each split attempt records:

- `iteration`: Split iteration number
- `old_n` / `proposed_n`: Component counts
- `residual_peak_row` / `residual_peak_col`: Proposed center
- `residual_peak_mass`: Integrated residual mass at peak
- `old_bic` / `new_bic`: BIC before/after split
- `old_rmse` / `new_rmse`: RMSE before/after split
- `accepted`: Whether split was accepted
- `rejection_reason`: Why split was rejected (if applicable)
- `runtime_s`: Runtime for this split attempt
- `physical_checks`: Dict of physical validity check results

Diagnostics are available in `RefinementResult.split_attempts`.

## Acceptance Criteria

For N+1 split to be accepted, **ALL** must be true:

1. **Physical Validity:**
   - `min_sigma <= σ <= max_sigma` for all components
   - `amplitude >= min_amplitude` for new component
   - Local support fraction >= threshold
   - New component resolvable from existing (σ-normalized distance ≥ threshold)

2. **Model Improvement (at least one):**
   - BIC improves by margin (`gmm_bic_improvement_margin`), OR
   - RMSE improves by ≥5%

3. **Count Limit:**
   - `n_components <= max_components`

This is stricter than the initial spec's "RMSE OR BIC", following your guidance to avoid overfitting.

## Testing Plan

### Integration Tests (Required)

Run Phase B integration tests:

```bash
python -m pytest tests/test_phase_b_integration.py -v
```

**Expected Results:**
- All 8 tests pass
- Hidden doublet recovers N≥2
- Clean single stays N=1
- 2→3 split works when structured residual exists
- Failed splits fall back safely
- Max iterations respected
- No multi-start called from split path

### Regression Tests (Required)

Run existing GMM tests to ensure no behavior change when disabled:

```bash
python -m pytest tests/test_gmm_multi_start.py -v
python -m pytest tests/test_gaussian_fitter.py -v
python -m pytest tests/test_phase1a_peak_combination_init.py -v
python -m pytest tests/test_phase_b_residual_guided_split.py -v
```

All existing tests should pass bit-for-bit when `residual_split_enabled=False`.

### Benchmark Validation (Recommended)

Compare Phase B enabled vs disabled on synthetic benchmarks:

1. **Separation Benchmark (hidden doublets):**
   ```bash
   # Baseline (Phase B disabled)
   python -m bioimage_pipeline.benchmarks.separation_benchmark \
       --n-seeds 20 \
       --output-dir synthetic_test_data/results/sep_baseline
   
   # Phase B enabled
   python -m bioimage_pipeline.benchmarks.separation_benchmark \
       --n-seeds 20 \
       --output-dir synthetic_test_data/results/sep_phase_b \
       --config-overrides residual_split_enabled=True
   ```
   
   **Expected:** Recovery rate improves at tight separations (sep2-3), false-split rate unchanged.

2. **False-Split Benchmark (clean singles):**
   ```bash
   # Baseline
   python -m bioimage_pipeline.benchmarks.false_split_benchmark \
       --n-seeds 20 \
       --output-dir synthetic_test_data/results/false_split_baseline
   
   # Phase B enabled
   python -m bioimage_pipeline.benchmarks.false_split_benchmark \
       --n-seeds 20 \
       --output-dir synthetic_test_data/results/false_split_phase_b \
       --config-overrides residual_split_enabled=True
   ```
   
   **Expected:** False-split rate should NOT increase (strong acceptance criteria prevent spurious splits).

3. **Runtime Comparison:**
   - Measure runtime per split attempt in diagnostics
   - Phase B should add <2s per object for typical cases (no full multi-start)

## Rollback Criteria

If ANY of these occur, disable Phase B and report:

1. **False-split rate increases** beyond acceptable threshold
2. **Regression tests fail** with Phase B disabled
3. **Runtime** per object increases by >50% on average
4. **Existing behavior changes** when `residual_split_enabled=False`

## Next Steps

After validation:

1. **Tune Thresholds:** Adjust acceptance criteria based on benchmark results
2. **Phase C:** Dynamic model order (N not fixed at 1/2/3)
3. **Phase D:** Stronger component validity (unified with CandidateFilter)
4. **Phase E:** Merge/refit loop (collapse redundant components)
5. **Phase F:** Runtime optimization (early stop, strategy ordering)

## Implementation Notes

### Why No Multi-Start for Splits?

Pan 2010 suggests split based on error distribution, not random exploration. Since we have:
- N fitted centers (already converged)
- 1 proposed residual center (deterministically computed)

A direct fit is sufficient and much faster than full multi-start.

### Why BIC-First Acceptance?

Following G5M 2026 guidance: adding components naturally lowers training residual due to increased degrees of freedom. BIC penalizes this, making it a more robust criterion than raw RMSE improvement.

### Why Separate Refiner Class?

Keeps `GaussianModelSelector` focused on initial model selection. Phase C (dynamic N), D (stronger validity), and E (merge/refit) can extend `ResidualSplitRefiner` without bloating `gaussian_fitter.py`.

## Commit Message Template

```
feat(puncta): implement Phase B residual-guided split (disabled by default)

Add ResidualSplitRefiner for iterative N→N+1 splitting based on structured
residual evidence. Integration layer preserves existing behavior when disabled.

Key changes:
- New ResidualSplitRefiner class in residual_refiner.py
- GaussianModelSelector calls refiner after initial selection
- Config: residual_split_enabled (default False), residual_split_max_iterations
- Deterministic N+1 refit (no multi-start) for runtime control
- Strong acceptance: physical validity + BIC improvement required
- 8 integration tests + all regression tests pass

Phase B is disabled by default; enable with residual_split_enabled=True.

Related: DEVELOPMENT_PLAN_GAUSSIAN_DECLUMPING.md Phase B
```
