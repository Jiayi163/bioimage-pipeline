# Phase 1A Implementation Specification: Peak-Combination Initialization

## Overview

This document specifies the exact code changes required to implement Phase 1A: peak-combination initialization for GMM multi-start search.

**Goal:** Generate initialization strategies directly from combinations of detected local maxima, ranked by peak quality (intensity + separation), to improve GMM search basin selection without threshold tuning.

**Status:** Specification only - awaiting approval before implementation.

---

## Proposed Code Changes

### Change 1: Add `gmm_peak_combination_max` config parameter

**File:** `bioimage_pipeline/puncta/config.py`

**Location:** After line 100 (after `gmm_multi_start_early_stop_bic_agreement`)

**Add:**
```python
gmm_peak_combination_max: int = 6
```

**Validation:** Add to `__post_init__` (after line 185):
```python
if self.gmm_peak_combination_max < 0:
    raise ValueError("gmm_peak_combination_max must be non-negative")
```

**Rationale:** Conservative default of 6 limits combinatorial explosion while covering common cases (C(4,2)=6). Value of 0 disables peak-pair generation.

---

### Change 2: Add peak-pair ranking helper function

**File:** `bioimage_pipeline/puncta/gmm_multi_start.py`

**Location:** After `_detector_based_init_peaks` function (after line 58)

**Add:**
```python
def _rank_peak_pairs(
    peaks: list[PeakCandidate],
    *,
    n_components: int,
) -> list[tuple[int, ...]]:
    """Return peak index combinations ranked by quality (intensity sum + separation).
    
    For n_components=2, returns pairs (i, j) where i < j.
    Ranking considers:
    - Combined peak intensity (higher is better)
    - Spatial separation (farther is better, up to a reasonable limit)
    
    Returns empty list if len(peaks) < n_components.
    """
    if len(peaks) < n_components:
        return []
    
    if n_components != 2:
        # Future: extend to triples for n=3
        # For now, only support pairs
        return []
    
    # Generate all pairs
    pairs: list[tuple[int, int]] = []
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            pairs.append((i, j))
    
    # Score each pair by: intensity_sum + separation_bonus
    def score_pair(indices: tuple[int, int]) -> float:
        i, j = indices
        peak_i, peak_j = peaks[i], peaks[j]
        
        # Intensity component (normalized to ~1000 scale)
        intensity_sum = peak_i.intensity + peak_j.intensity
        
        # Separation component (Euclidean distance)
        separation = math.hypot(
            peak_i.col - peak_j.col,
            peak_i.row - peak_j.row,
        )
        
        # Separation bonus: prefer well-separated pairs
        # Use sqrt to diminish returns for very distant peaks
        # (we want separation ~2-6 px, not 20+ px)
        separation_bonus = 100.0 * math.sqrt(separation)
        
        return intensity_sum + separation_bonus
    
    # Sort pairs by score (descending)
    ranked_pairs = sorted(pairs, key=score_pair, reverse=True)
    
    return ranked_pairs
```

**Rationale:** Explicit ranking function separates concerns, testable in isolation, extensible to triples.

---

### Change 3: Modify `generate_two_component_init_sets` to include peak-pair strategies

**File:** `bioimage_pipeline/puncta/gmm_multi_start.py`

**Location:** In `generate_two_component_init_sets` function, after line 88 (after `if not peaks: return strategies`)

**Insert after the `if not peaks: return strategies` block:**

```python
    # Phase 1A: Peak-combination initialization
    # Generate ranked peak-pair strategies from detected maxima
    if len(peaks) >= 2 and config.gmm_peak_combination_max > 0:
        ranked_pairs = _rank_peak_pairs(peaks, n_components=2)
        for pair_index, (i, j) in enumerate(ranked_pairs[:config.gmm_peak_combination_max]):
            peak_i = peaks[i]
            peak_j = peaks[j]
            strategies[f"peak_pair_{i}_{j}"] = [
                PeakCandidate(row=peak_i.row, col=peak_i.col, intensity=peak_i.intensity),
                PeakCandidate(row=peak_j.row, col=peak_j.col, intensity=peak_j.intensity),
            ]
```

**Rationale:** 
- Only generates peak-pairs when 2+ peaks available
- Respects `gmm_peak_combination_max` cap
- Uses actual detected peak positions (not geometric offsets)
- Names clearly indicate source peaks (e.g. `peak_pair_0_1`)

---

### Change 4: Modify `ordered_multi_start_strategies` to prioritize peak-pairs

**File:** `bioimage_pipeline/puncta/gmm_multi_start.py`

**Location:** In `ordered_multi_start_strategies` function, replace lines 164-173 with:

**Current code (lines 164-173):**
```python
def ordered_multi_start_strategies(
    init_sets: dict[str, list[PeakCandidate]],
    *,
    config: PunctaDeclumpConfig,
) -> list[str]:
    """Return strategy execution order: staged priority, then symmetric, then offset."""
    available = set(init_sets.keys())
    ordered: list[str] = []
    for key in ("detector_based", "residual_peak", "major_axis"):
        if key in available:
            ordered.append(key)
```

**New code:**
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
    # These are ranked by _rank_peak_pairs, so preserve that order
    peak_pair_names = sorted(
        [name for name in available if name.startswith("peak_pair_")],
        key=lambda name: (
            # Extract indices from "peak_pair_i_j" and use original ranking
            # The dict insertion order preserves ranking from generate_two_component_init_sets
            list(init_sets.keys()).index(name)
        ),
    )
    ordered.extend(peak_pair_names)
    
    # Stage 3: residual and geometry-based
    for key in ("residual_peak", "major_axis"):
        if key in available:
            ordered.append(key)
```

**Leave the rest of the function unchanged** (lines 169-176 handle symmetric/offset/remaining).

**Rationale:**
- Peak-pairs come immediately after detector_based (highest priority non-standard init)
- Preserves ranking from `_rank_peak_pairs` (brightest + well-separated first)
- Before residual_peak (which depends on single-fit, may duplicate peaks)
- Well before symmetric/offset geometric strategies

---

### Change 5: Add import for `math` module

**File:** `bioimage_pipeline/puncta/gmm_multi_start.py`

**Location:** After line 6 (after existing imports)

**Current imports:**
```python
import math
import time
from dataclasses import dataclass, field
```

**No change needed** - `math` is already imported.

---

## Summary of Files Modified

| File | Changes |
|------|---------|
| `bioimage_pipeline/puncta/config.py` | Add `gmm_peak_combination_max` parameter (default 6) |
| `bioimage_pipeline/puncta/gmm_multi_start.py` | Add `_rank_peak_pairs()`, modify `generate_two_component_init_sets()`, modify `ordered_multi_start_strategies()` |
| `tests/test_phase1a_peak_combination_init.py` | New comprehensive test suite (12 tests) |

**Total lines changed:** ~100 lines added (including tests, docs, comments)

**Total lines in production code:** ~60 lines

---

## Expected Behavior Changes

### What changes:
1. When 2+ filtered peaks exist, GMM multi-start will now try peak-pair combinations early in the search
2. Execution order becomes: `detector_based` → `peak_pair_*` → `residual_peak` → `major_axis` → geometric offsets
3. Number of peak-pair attempts is capped by `gmm_peak_combination_max` (default 6)

### What does NOT change:
1. Fast path routing (unchanged)
2. Model selection logic (BIC, AIC, merge rules)
3. CandidateFilter acceptance thresholds
4. Single Gaussian fitting
5. Residual analysis
6. Diagnostic exports
7. Behavior when 0 or 1 peaks detected (falls back to detector_based)

### Over-splitting risk mitigation:
1. Peak-pairs only added when multiple peaks detected (not for single spots)
2. Capped by `gmm_peak_combination_max` (default 6, not 20)
3. Global `gmm_max_multi_starts` cap still applies
4. Existing merge rules (`gmm_min_component_separation`) unchanged
5. Model selection BIC penalty unchanged
6. CandidateFilter duplicate rejection unchanged

---

## Validation Plan

### Test coverage:
- **Unit tests (12 tests):** Peak-pair generation, ranking, ordering, caps, edge cases
- **Regression tests:** sep3_seed101 should continue to work
- **Failure recovery tests:** sep3_seed1010 should show `multi_start_converged > 0`
- **False-split protection:** Clean single Gaussians should not over-split

### Benchmark validation (next step, after implementation):
1. Run full separation benchmark: `python scripts/run_synthetic_benchmarks.py --benchmark stage2_separation`
2. Run false-split benchmark: `python scripts/run_synthetic_benchmarks.py --benchmark false_split`
3. Run oracle experiment: `python scripts/run_gmm_oracle_experiment.py --case sep_benchmark_sep3_seed1010`
4. Compare before/after metrics (see PHASE1A_METRICS.md)

---

## Config Example

### Minimal config change to enable Phase 1A:
```python
config = PunctaDeclumpConfig(
    gmm_multi_start_enabled=True,
    gmm_peak_combination_max=6,  # NEW parameter (default)
)
```

### To disable peak-pair initialization (if regressions occur):
```python
config = PunctaDeclumpConfig(
    gmm_multi_start_enabled=True,
    gmm_peak_combination_max=0,  # Disables peak-pair generation
)
```

### To increase peak-pair search (for very complex objects):
```python
config = PunctaDeclumpConfig(
    gmm_multi_start_enabled=True,
    gmm_peak_combination_max=10,  # More aggressive search
    gmm_max_multi_starts=25,       # Increase global cap accordingly
)
```

---

## Open Questions (to resolve before implementation)

1. **Peak ranking weights:** Current scoring uses `intensity_sum + 100*sqrt(separation)`. Should we:
   - Tune the 100 multiplier?
   - Add peak prominence or local support as a factor?
   - Use a different separation function (e.g., sigmoid)?

2. **Cap default:** Is 6 the right default for `gmm_peak_combination_max`?
   - Too low: may miss the correct pair in 4+ peak objects
   - Too high: runtime cost, more local minima
   - Alternative: adaptive cap based on object complexity tier?

3. **Triple combinations:** Should Phase 1A include peak triples for n=3 component fits?
   - Defer to Phase 1 follow-up?
   - Or implement now since the framework is ready?

4. **Order preservation:** Should we use dict insertion order or explicit indices for ranking?
   - Current approach relies on Python 3.7+ dict ordering
   - Alternative: attach a `_rank` attribute to strategy names

---

## Next Steps

1. **Review this specification** - confirm approach is sound
2. **Approve/revise** - make any adjustments to ranking, caps, or ordering
3. **Run tests** - execute `tests/test_phase1a_peak_combination_init.py` to confirm current failures
4. **Implement** - apply the exact code changes specified above
5. **Validate** - run benchmark suite and compare metrics
6. **Document metrics** - record before/after results in PHASE1A_METRICS.md

---

**Prepared by:** Cursor Agent  
**Date:** 2026-08-17  
**Status:** Awaiting review and approval
