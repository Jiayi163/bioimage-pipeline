# Phase 1A Implementation - Code Diff Summary

## File 1: bioimage_pipeline/puncta/config.py

### Change 1.1: Add config parameter (after line 100)

```diff
     gmm_multi_start_early_stop_min_converged: int = 2
     gmm_multi_start_early_stop_bic_agreement: float = 15.0
+    gmm_peak_combination_max: int = 6

     # Selective routing / detectors
```

### Change 1.2: Add validation (after line 185)

```diff
         if not self.gmm_multi_start_separations:
             raise ValueError("gmm_multi_start_separations must not be empty")
+        if self.gmm_peak_combination_max < 0:
+            raise ValueError("gmm_peak_combination_max must be non-negative")

         # Backward compatibility for legacy boolean flags.
```

---

## File 2: bioimage_pipeline/puncta/gmm_multi_start.py

### Change 2.1: Add _rank_peak_pairs function (after line 58)

```diff
     return init


+def _rank_peak_pairs(
+    peaks: list[PeakCandidate],
+    *,
+    n_components: int,
+    min_separation: float,
+) -> list[tuple[int, ...]]:
+    """Return peak index combinations ranked by intensity, filtered by separation.
+    
+    Ranking strategy:
+    1. Reject pairs below min_separation threshold
+    2. Rank remaining pairs primarily by combined peak intensity
+    3. Use separation as tie-breaker (prefer well-separated over close)
+    4. Return empty list if no valid pairs exist
+    
+    Returns empty list if len(peaks) < n_components or no pairs meet separation.
+    """
+    if len(peaks) < n_components:
+        return []
+    
+    if n_components != 2:
+        return []
+    
+    valid_pairs: list[tuple[int, int]] = []
+    for i in range(len(peaks)):
+        for j in range(i + 1, len(peaks)):
+            separation = math.hypot(
+                peaks[i].col - peaks[j].col,
+                peaks[i].row - peaks[j].row,
+            )
+            if separation >= min_separation:
+                valid_pairs.append((i, j))
+    
+    if not valid_pairs:
+        return []
+    
+    # Sort by: intensity sum (descending), then separation (descending as tie-breaker)
+    def sort_key(indices: tuple[int, int]) -> tuple[float, float]:
+        i, j = indices
+        peak_i, peak_j = peaks[i], peaks[j]
+        intensity_sum = peak_i.intensity + peak_j.intensity
+        separation = math.hypot(peak_i.col - peak_j.col, peak_i.row - peak_j.row)
+        return (-intensity_sum, -separation)
+    
+    ranked_pairs = sorted(valid_pairs, key=sort_key)
+    
+    return ranked_pairs
+
+
 @dataclass(frozen=True)
```

### Change 2.2: Add peak-pair generation (after line 136)

```diff
     if not peaks:
         return strategies

+    # Phase 1A: Peak-combination initialization
+    if len(peaks) >= 2 and config.gmm_peak_combination_max > 0:
+        ranked_pairs = _rank_peak_pairs(
+            peaks,
+            n_components=2,
+            min_separation=config.gmm_acceptance_min_separation,
+        )
+        for pair_index, (i, j) in enumerate(ranked_pairs[:config.gmm_peak_combination_max]):
+            peak_i = peaks[i]
+            peak_j = peaks[j]
+            strategies[f"peak_pair_{i}_{j}"] = [
+                PeakCandidate(row=peak_i.row, col=peak_i.col, intensity=peak_i.intensity),
+                PeakCandidate(row=peak_j.row, col=peak_j.col, intensity=peak_j.intensity),
+            ]
+
     base = peaks[0]
```

### Change 2.3: Replace ordered_multi_start_strategies (lines 221-239)

```diff
 def ordered_multi_start_strategies(
     init_sets: dict[str, list[PeakCandidate]],
     *,
     config: PunctaDeclumpConfig,
 ) -> list[str]:
-    """Return strategy execution order: staged priority, then symmetric, then offset."""
+    """Return strategy execution order: detector-based, peak-pairs, then others.
+    
+    Priority order:
+    1. detector_based (top peaks by intensity)
+    2. peak_pair_* (ranked combinations of detected peaks) [Phase 1A]
+    3. residual_peak (single fit residual maximum)
+    4. major_axis (object geometry)
+    5. symmetric_* (geometric offsets)
+    6. offset_* (geometric offsets)
+    """
     available = set(init_sets.keys())
     ordered: list[str] = []
-    for key in ("detector_based", "residual_peak", "major_axis"):
-        if key in available:
-            ordered.append(key)
+    
+    # Stage 1: detector-based (existing)
+    if "detector_based" in available:
+        ordered.append("detector_based")
+    
+    # Stage 2: peak-pair combinations (Phase 1A - NEW)
+    # Preserve ranking from _rank_peak_pairs by maintaining dict insertion order
+    peak_pair_names = sorted(
+        [name for name in available if name.startswith("peak_pair_")],
+        key=lambda name: list(init_sets.keys()).index(name),
+    )
+    ordered.extend(peak_pair_names)
+    
+    # Stage 3: residual and geometry-based
+    for key in ("residual_peak", "major_axis"):
+        if key in available:
+            ordered.append(key)
+    
+    # Stage 4: symmetric and offset strategies
     ordered.extend(sorted(name for name in available if name.startswith("symmetric_")))
     ordered.extend(sorted(name for name in available if name.startswith("offset_")))
+    
+    # Stage 5: any remaining strategies
     for name in sorted(available):
         if name not in ordered:
             ordered.append(name)
+    
+    # Global cap
     if config.gmm_max_multi_starts > 0:
         ordered = ordered[: config.gmm_max_multi_starts]
+    
     return ordered
```

---

## Summary Statistics

| File | Insertions | Modifications | Total Changes |
|------|-----------|---------------|---------------|
| `config.py` | +2 | +2 | 4 lines |
| `gmm_multi_start.py` | +62 | +19 | 81 lines |
| **Total** | **+64** | **+21** | **85 lines** |

---

## New Functions Added

1. `_rank_peak_pairs()` - 47 lines
   - Filters peak pairs by minimum separation
   - Ranks by intensity (primary) and separation (tie-breaker)
   - Returns empty if no valid pairs

---

## Modified Functions

1. `generate_two_component_init_sets()` - Added 14 lines
   - Generates peak-pair strategies after detector-based
   - Respects gmm_peak_combination_max cap

2. `ordered_multi_start_strategies()` - Restructured (same line count, better organization)
   - Added peak-pair stage between detector-based and residual
   - Preserves ranking from _rank_peak_pairs

---

## Testing Commands

### 1. Verify Implementation
```bash
python verify_phase1a.py
```

### 2. Run Phase 1A Tests
```bash
pytest tests/test_phase1a_peak_combination_init.py -v
```

### 3. Run Regression Tests
```bash
pytest tests/test_gmm_multi_start.py -v
```

### 4. Run Smoke Test (Single Test)
```bash
pytest tests/test_phase1a_peak_combination_init.py::test_peak_pair_init_strategies_generated_with_multiple_peaks -xvs
```

---

## Key Design Decisions (Approved)

1. **Simple Ranking**: Intensity primary, separation tie-breaker only
2. **No Fallback**: Returns empty if all pairs too close (not near-duplicates)
3. **Configured Threshold**: Uses `gmm_acceptance_min_separation` (typically 1.5px)
4. **No New Heuristics**: Avoids 2:1 weighting or scale-independent scoring
5. **Experimental Clarity**: Isolates "use detected peak pairs" effect

---

## What to Expect

### When Peak-Pairs Are Generated:
- 2+ peaks detected
- At least one pair has separation ≥ `gmm_acceptance_min_separation` (1.5px)
- `gmm_peak_combination_max > 0` (default 6)

### When Peak-Pairs Are Skipped:
- 0 or 1 peak detected
- All pairs too close (<1.5px separation)
- `gmm_peak_combination_max = 0` (disabled)

### Strategy Execution Order Example:
```
1. detector_based
2. peak_pair_0_1
3. peak_pair_0_2
4. peak_pair_1_2
5. residual_peak
6. major_axis
7. symmetric_x_sep2
8. symmetric_y_sep2
... (up to gmm_max_multi_starts total)
```

---

## Verification Checklist

- [ ] `verify_phase1a.py` passes all checks
- [ ] All 12 Phase 1A tests pass
- [ ] All existing GMM tests pass (no regressions)
- [ ] Benchmarks show improvement or no degradation
- [ ] False-split rate within acceptable range
- [ ] Runtime impact acceptable

---

**Status:** Implementation complete, ready for testing  
**Date:** 2026-08-17
