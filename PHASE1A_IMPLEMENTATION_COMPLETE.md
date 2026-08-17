# Phase 1A Implementation Complete

**Date:** 2026-08-17  
**Status:** ✅ Implemented, awaiting test verification

---

## Implementation Summary

Phase 1A adds **peak-combination initialization** for GMM multi-start search. The implementation generates ranked initialization strategies directly from detected peak pairs, prioritizing bright and well-separated combinations.

---

## Code Changes

### 1. Config Parameter (bioimage_pipeline/puncta/config.py)

**Added after line 100:**
```python
gmm_peak_combination_max: int = 6
```

**Added validation after line 185:**
```python
if self.gmm_peak_combination_max < 0:
    raise ValueError("gmm_peak_combination_max must be non-negative")
```

**Rationale:**
- Default of 6 covers C(4,2) = 6 pairs from 4 peaks (typical case)
- Conservative to avoid combinatorial explosion
- Value of 0 disables peak-pair generation entirely

---

### 2. Peak-Pair Ranking Function (bioimage_pipeline/puncta/gmm_multi_start.py)

**Added after line 58 (_detector_based_init_peaks):**

```python
def _rank_peak_pairs(
    peaks: list[PeakCandidate],
    *,
    n_components: int,
    min_separation: float,
) -> list[tuple[int, ...]]:
    """Return peak index combinations ranked by intensity, filtered by separation.
    
    Ranking strategy:
    1. Reject pairs below min_separation threshold
    2. Rank remaining pairs primarily by combined peak intensity
    3. Use separation as tie-breaker (prefer well-separated over close)
    4. Return empty list if no valid pairs exist
    
    Returns empty list if len(peaks) < n_components or no pairs meet separation.
    """
    if len(peaks) < n_components:
        return []
    
    if n_components != 2:
        return []
    
    valid_pairs: list[tuple[int, int]] = []
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            separation = math.hypot(
                peaks[i].col - peaks[j].col,
                peaks[i].row - peaks[j].row,
            )
            if separation >= min_separation:
                valid_pairs.append((i, j))
    
    if not valid_pairs:
        return []
    
    # Sort by: intensity sum (descending), then separation (descending as tie-breaker)
    def sort_key(indices: tuple[int, int]) -> tuple[float, float]:
        i, j = indices
        peak_i, peak_j = peaks[i], peaks[j]
        intensity_sum = peak_i.intensity + peak_j.intensity
        separation = math.hypot(peak_i.col - peak_j.col, peak_i.row - peak_j.row)
        return (-intensity_sum, -separation)
    
    ranked_pairs = sorted(valid_pairs, key=sort_key)
    
    return ranked_pairs
```

**Design Principles:**
- **No hard-coded constants:** Uses configured `gmm_acceptance_min_separation` (typically 1.5px)
- **Simple ranking:** Primary = intensity sum, secondary = separation (tie-breaker)
- **No fallback:** Returns empty list if no valid pairs, letting existing strategies handle it
- **Experimental clarity:** Isolates "use detected peak pairs" effect without introducing new scoring heuristics

---

### 3. Peak-Pair Strategy Generation (bioimage_pipeline/puncta/gmm_multi_start.py)

**Added after line 136 in generate_two_component_init_sets:**

```python
    # Phase 1A: Peak-combination initialization
    if len(peaks) >= 2 and config.gmm_peak_combination_max > 0:
        ranked_pairs = _rank_peak_pairs(
            peaks,
            n_components=2,
            min_separation=config.gmm_acceptance_min_separation,
        )
        for pair_index, (i, j) in enumerate(ranked_pairs[:config.gmm_peak_combination_max]):
            peak_i = peaks[i]
            peak_j = peaks[j]
            strategies[f"peak_pair_{i}_{j}"] = [
                PeakCandidate(row=peak_i.row, col=peak_i.col, intensity=peak_i.intensity),
                PeakCandidate(row=peak_j.row, col=peak_j.col, intensity=peak_j.intensity),
            ]
```

**Behavior:**
- Only generates strategies when ≥2 peaks available
- Respects `gmm_peak_combination_max` cap (default 6)
- Uses actual detected peak positions (not geometric offsets)
- Strategy names indicate source peaks: `peak_pair_0_1`, `peak_pair_0_2`, etc.

---

### 4. Strategy Ordering (bioimage_pipeline/puncta/gmm_multi_start.py)

**Replaced lines 221-239 (ordered_multi_start_strategies function):**

```python
def ordered_multi_start_strategies(
    init_sets: dict[str, list[PeakCandidate]],
    *,
    config: PunctaDeclumpConfig,
) -> list[str]:
    """Return strategy execution order: detector-based, peak-pairs, then others.
    
    Priority order:
    1. detector_based (top peaks by intensity)
    2. peak_pair_* (ranked combinations of detected peaks) [Phase 1A]
    3. residual_peak (single fit residual maximum)
    4. major_axis (object geometry)
    5. symmetric_* (geometric offsets)
    6. offset_* (geometric offsets)
    """
    available = set(init_sets.keys())
    ordered: list[str] = []
    
    # Stage 1: detector-based (existing)
    if "detector_based" in available:
        ordered.append("detector_based")
    
    # Stage 2: peak-pair combinations (Phase 1A - NEW)
    # Preserve ranking from _rank_peak_pairs by maintaining dict insertion order
    peak_pair_names = sorted(
        [name for name in available if name.startswith("peak_pair_")],
        key=lambda name: list(init_sets.keys()).index(name),
    )
    ordered.extend(peak_pair_names)
    
    # Stage 3: residual and geometry-based
    for key in ("residual_peak", "major_axis"):
        if key in available:
            ordered.append(key)
    
    # Stage 4: symmetric and offset strategies
    ordered.extend(sorted(name for name in available if name.startswith("symmetric_")))
    ordered.extend(sorted(name for name in available if name.startswith("offset_")))
    
    # Stage 5: any remaining strategies
    for name in sorted(available):
        if name not in ordered:
            ordered.append(name)
    
    # Global cap
    if config.gmm_max_multi_starts > 0:
        ordered = ordered[: config.gmm_max_multi_starts]
    
    return ordered
```

**Rationale:**
- Peak-pairs come immediately after `detector_based` (highest priority)
- Before `residual_peak` (which depends on single-fit, may duplicate peaks)
- Well before `symmetric_*/offset_*` geometric strategies
- Preserves ranking from `_rank_peak_pairs` via dict insertion order

---

## Files Modified

| File | Lines Added | Lines Changed | Purpose |
|------|-------------|---------------|---------|
| `bioimage_pipeline/puncta/config.py` | 2 | 2 | Add config parameter + validation |
| `bioimage_pipeline/puncta/gmm_multi_start.py` | 62 | 19 | Add ranking function, generation logic, ordering |
| **Total Production Code** | **64** | **21** | Phase 1A implementation |

---

## What Changed

### Behavior Changes:
1. When 2+ filtered peaks exist and are separated by ≥1.5px, GMM multi-start generates `peak_pair_*` strategies
2. Execution order becomes: `detector_based` → `peak_pair_*` → `residual_peak` → `major_axis` → geometric offsets
3. Number of peak-pair attempts capped by `gmm_peak_combination_max` (default 6)
4. Total strategies still capped by `gmm_max_multi_starts` (default 20)

### What Did NOT Change:
1. Fast path routing (unchanged)
2. Model selection logic (BIC, AIC, merge rules)
3. CandidateFilter acceptance thresholds
4. Single Gaussian fitting
5. Residual analysis
6. Diagnostic exports
7. Behavior when 0 or 1 peaks detected (falls back to detector_based + geometric)

---

## Verification Steps

### 1. Quick Verification (Smoke Test)

```bash
python verify_phase1a.py
```

Expected output:
```
Phase 1A Implementation Verification
==================================================

Test: Config parameter
✓ Config parameter exists with default value 6
✓ Validation works - negative values rejected

Test: Function signature
✓ _rank_peak_pairs function exists with correct signature

Test: Peak-pair generation
✓ Peak-pair strategies generated: ['peak_pair_0_1', 'peak_pair_0_2', ...]

Test: Strategy ordering
  Ordered strategies: ['detector_based', 'peak_pair_0_1', ...]
✓ Peak-pair strategies are ordered correctly

==================================================
Verification: 4/4 tests passed

✓ Phase 1A implementation verified!
```

### 2. Full Test Suite (Critical Tests 1-11)

```bash
pytest tests/test_phase1a_peak_combination_init.py -v
```

Expected: All 12 tests pass (tests 1-11 are critical, test 6 and 12 are aspirational)

### 3. Regression Tests (Existing GMM Tests)

```bash
pytest tests/test_gmm_multi_start.py -v
```

Expected: All existing tests continue to pass (no regressions)

---

## Benchmark Validation (Next Step)

To measure Phase 1A performance impact:

### Before/After Workflow

1. **Baseline (Before Phase 1A - if not already done):**
   ```bash
   # Revert Phase 1A changes temporarily
   git stash
   
   # Run benchmarks
   python scripts/run_synthetic_benchmarks.py --data-root synthetic_test_data --benchmark stage2_separation --num-seeds 20
   python scripts/run_synthetic_benchmarks.py --data-root synthetic_test_data --benchmark false_split --num-seeds 5
   
   # Save results
   cp -r synthetic_test_data/results synthetic_test_data/results_before_phase1a
   
   # Restore Phase 1A
   git stash pop
   ```

2. **After Phase 1A:**
   ```bash
   # Run same benchmarks (overwrites results)
   python scripts/run_synthetic_benchmarks.py --data-root synthetic_test_data --benchmark stage2_separation --num-seeds 20
   python scripts/run_synthetic_benchmarks.py --data-root synthetic_test_data --benchmark false_split --num-seeds 5
   ```

3. **Compare Results:**
   ```bash
   # Compare aggregate metrics
   diff synthetic_test_data/results_before_phase1a/benchmark_reports/stage2_separation_aggregate.json \
        synthetic_test_data/results/benchmark_reports/stage2_separation_aggregate.json
   
   diff synthetic_test_data/results_before_phase1a/benchmark_reports/false_split_aggregate.json \
        synthetic_test_data/results/benchmark_reports/false_split_aggregate.json
   ```

### Key Metrics to Compare

| Metric | File | Column/Field | Pass Condition |
|--------|------|--------------|----------------|
| False-split rate | `false_split_aggregate.json` | `false_split_rate` | ≤ baseline + 1% |
| sep3 recovery | `stage2_separation_aggregate.json` | `sep3.recovery_rate` | ≥ baseline + 5% |
| sep4 recovery | `stage2_separation_aggregate.json` | `sep4.recovery_rate` | ≥ baseline + 5% |
| Median converged | `stage2_separation_summary.csv` | `multi_start_converged` (median) | ≥ baseline |
| GMM runtime | `stage2_separation_summary.csv` | `gmm_runtime_s` (median) | ≤ baseline × 1.5 |

---

## Success Criteria

### Primary (All Must Pass)

- [x] Code implemented and compiles without errors
- [ ] Verification script passes (4/4 tests)
- [ ] All Phase 1A tests pass (12/12)
- [ ] All existing GMM tests pass (no regressions)
- [ ] False-split rate ≤ baseline + 1%
- [ ] sep3 or sep4 recovery ≥ baseline + 5%
- [ ] Runtime ≤ baseline × 1.5

### Secondary (Improvement Evidence)

- [ ] Peak-pair strategies appear in winning strategies
- [ ] Oracle gap decreases
- [ ] sep3_seed1010 converges (currently fails)

### Rollback Criteria (Immediate Revert If)

- [ ] False-split rate increases >2% absolute
- [ ] Runtime increases >2× (100%)
- [ ] Separation recovery decreases >5% absolute
- [ ] Any critical test (1-11) fails

---

## Next Steps

1. ✅ **Implementation complete** - Phase 1A code is in place
2. ⏳ **Run verification script** - `python verify_phase1a.py`
3. ⏳ **Run Phase 1A tests** - `pytest tests/test_phase1a_peak_combination_init.py -v`
4. ⏳ **Run regression tests** - `pytest tests/test_gmm_multi_start.py -v`
5. ⏳ **Run benchmarks** - Compare before/after performance
6. ⏳ **Analyze results** - Document metrics in comparison table
7. ⏳ **Decision** - Approve, tune, or rollback based on criteria

---

## Not Implemented (As Requested)

- ❌ Phase 1B: Triple initialization (for n=3 components)
- ❌ Phase 1C: Peak quality weighting
- ❌ Phase 1D: Adaptive cap tuning
- ❌ Advanced ranking models (scale-independent scoring, etc.)

These remain for future work if Phase 1A proves successful.

---

**Prepared by:** Cursor Agent  
**Date:** 2026-08-17  
**Status:** Implementation complete, awaiting validation
