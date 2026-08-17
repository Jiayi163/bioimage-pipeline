# Gaussian/GMM Puncta Declumping — Development Plan

This document summarizes the current state of the puncta declumping pipeline, confirmed limitations, and a phased roadmap for improving robustness on complex overlapping puncta and dense cloud-like objects.

**Status:** Planning only. No production behavior changes are authorized until Phase 1 is reviewed and approved.

---

# Current status

The pipeline already implements a full selective declumping workflow:

| Capability | Primary location |
|------------|------------------|
| Threshold / external-mask segmentation | `bioimage_pipeline/puncta/pipeline.py`, `segment.py` |
| Connected-object analysis | `bioimage_pipeline/puncta/connected_objects.py` |
| Image-level local maxima (TrackMate / python_log) | `bioimage_pipeline/puncta/candidate_detector.py` |
| Peak-to-object assignment | `bioimage_pipeline/puncta/peak_assignment.py` |
| ROI local maxima (fallback when no assigned peaks) | `bioimage_pipeline/puncta/maxima_detector.py` |
| Selective routing (fast vs suspicious) | `bioimage_pipeline/puncta/object_router.py`, `pipeline.py` |
| Single elliptical Gaussian fitting | `bioimage_pipeline/puncta/gaussian_fitter.py` (`EllipticalGaussianFitter`) |
| Multi-start 2-component GMM | `bioimage_pipeline/puncta/gmm_multi_start.py` |
| 2- and sometimes 3-component mixture fitting | `gaussian_fitter.py` (`GaussianModelSelector.select_balanced_model`) |
| BIC/AIC model comparison | `gaussian_fitter.py` |
| Post-fit component acceptance | `bioimage_pipeline/puncta/candidate_filter.py` |
| Under-split suspect flagging + report | `bioimage_pipeline/puncta/object_processor.py`, `under_split_report.py` |
| CSV / JSON / timing exports | `bioimage_pipeline/puncta/export.py` |
| Diagnostic PNG/TIFF overlays | `bioimage_pipeline/puncta/diagnostics.py` |
| Synthetic benchmark runner | `scripts/run_synthetic_benchmarks.py` |
| Single-case benchmark reruns (`--case`) | `scripts/run_synthetic_benchmarks.py` |
| Validation-only oracle experiments | `bioimage_pipeline/puncta/validation/gmm_oracle.py`, `scripts/run_gmm_oracle_experiment.py` |
| Ablation / probe tooling | `bioimage_pipeline/puncta/validation/gmm_probe.py` |

The project is no longer missing basic functionality. The main gap is **robustness** on difficult overlaps and dense clouds.

---

# Confirmed limitations

## 1. GMM initialization / search instability

**Evidence**

- Synthetic separation benchmark: cases such as `sep_benchmark_sep3_seed101` can succeed while `sep_benchmark_sep3_seed1010` fails despite similar separation.
- Failed multi-start runs often show many optimizer-converged attempts that **collapse to one component after merge**, not merely optimizer failure.
- Validation oracle (`gmm_oracle.py`): when initialized near known synthetic ground-truth centers, the same mixture model recovers a correct 2-component solution.

**Conclusion:** The objective/model can represent the answer; production search often lands in the wrong basin.

## 2. Complex real-image clouds remain under-split

Real microscopy diagnostics show:

- Multiple filtered maxima but 0–1 accepted `fit_ok` components
- Good GMM R² improvement over single Gaussian, yet only one accepted component
- Large / elongated objects with structured residuals

Under-split reporting (`under_split_report.py`) supports these patterns with categories such as `gmm_tried_but_only_one_accepted`, `gmm_tried_but_components_collapsed`, and `gmm_tried_but_second_component_rejected`.

## 3. Current model order is too limited for some dense objects

Production model selection (`select_balanced_model`) is essentially:

```
single → try 2 (multi-start) → maybe try 3 (detector init only)
```

Caps:

- `gmm_max_components = 3` (default)
- `gmm_max_components_large = 5` only when `equivalent_diameter > large_object_diameter_threshold`

Real objects with 4+ filtered maxima can therefore be structurally under-modeled (`gmm_candidate_components = 3` while 4 peaks present).

## 4. Model selection and component acceptance can disagree

The pipeline has three distinct gates after optimization:

1. **Post-fit merge** (`_merge_close_components`, threshold `gmm_min_component_separation`)
2. **Model selection** (`select_balanced_model`, BIC/AIC + spurious-split guard)
3. **CandidateFilter** (`evaluate_mixture_components`, amplitude/sigma/residual/duplicate rules)

A 2-component GMM can win model selection (large R² gain) while CandidateFilter accepts only one component. Current exports include `rejection_reason` per candidate, but not full component-level diagnostic fields (local support, contribution fraction, local residual improvement).

## 5. Real-image accuracy is not yet directly measurable

Synthetic benchmarks and oracle tooling exist, but there is no committed real ROI ground-truth set with precision/recall/count agreement metrics. High R² on real images is necessary but not sufficient for biological correctness.

---

# Development phases

## Phase 1 — GMM initialization / search robustness

**Goal:** Improve basin selection without threshold tuning as the primary fix.

Planned work:

- Peak-combination initialization from filtered local maxima
- Stronger residual-based initialization (mixture residual, not only single-fit residual)
- Collapse-aware bounded retry after merge collapse
- Optional coarse combination search for hard 2-component cases (runtime-gated)
- Benchmark before/after on separation + false-split controls
- Oracle-gap metric (normal init vs ground-truth init)

**Out of scope for Phase 1:** raising N>3 globally, lowering separation thresholds, increasing `max_nfev` everywhere.

## Phase 2 — Dynamic component count

- Evidence-driven progression: 1 → 2 → … up to configured cap (4–5 on complex objects only)
- Strong stopping rules (BIC plateau, residual structure absent, invalid bounds)
- False-split benchmark as gate

## Phase 3 — Residual-driven iterative splitting

- After each accepted N-component fit, analyze mixture `residual_patch`
- Test N+1 only when structured positive residual remains
- Integrate with under-split classification

## Phase 4 — Component-level acceptance diagnostics

- Explicit per-component validation fields in exports
- Determine whether CandidateFilter is too aggressive **after** reasons are visible
- No threshold tuning until diagnostics justify it

## Phase 5 — Real-data validation tooling

- ROI export, annotation schema, evaluation script
- 20–50 representative ROIs with count/center metrics
- Baseline before/after each algorithm phase

## Phase 6 — Runtime optimization

- Staged search and early stopping once accuracy work stabilizes
- Full multi-start only for hard suspicious objects

---

# Validation requirements

Every algorithmic phase must be evaluated against:

| Category | Tooling |
|----------|---------|
| Separation benchmark | `scripts/run_synthetic_benchmarks.py --benchmark stage2_separation` |
| Previously successful cases | Same benchmark, all seeds — not only failures |
| Previously failed cases | e.g. `sep_benchmark_sep3_seed1010` via `--case` |
| False-split controls | `false_split` benchmark |
| Simple singles | `test_gmm_multi_start.py::test_noisy_single_gaussian_mostly_not_oversplit`, `test_puncta_declump.py` |
| Oracle gap | `scripts/run_gmm_oracle_experiment.py` vs production multi-start on same case |
| Runtime | Benchmark CSV timing + multi-start profiling fields |
| Under/over-split rates | `evaluate_synthetic_puncta.py`, `_undersplit_report.csv` |

**Do not optimize for one synthetic failure case in isolation.**

---

# Important non-solutions

Do **not** use these as primary fixes:

1. **Lowering `gmm_min_component_separation` or `min_center_separation`** — accepts collapsed wrong minima (e.g. 1.4 px when truth is ~3 px).
2. **Increasing `gmm_multi_start_max_nfev` everywhere** — many failures already converge to collapsed solutions.
3. **Running 5-component GMM on every object** — runtime, overfitting, false splitting.
4. **Tuning only one failed benchmark case** — must preserve successes and false-split controls.
5. **Interpreting high R² on real images as ground truth** — need annotated ROI validation (Phase 5).

---

# Immediate next task

**Phase 1 planning review only.** See detailed Phase 1 appendix below and the separate review report in the PR/discussion. **Do not implement Phase 1 until approved.**

---

# Appendix A — Production control flow (reference)

```
pipeline.run()
  detect image-level peaks → assign_peaks_to_objects()
  for each object:
    ObjectRouter.classify()
      ordinary_single → process_fast() [no Gaussian fit]
      suspicious      → process_suspicious()
        build patch, peaks (assigned or MaximaDetector)
        fit single Gaussian on primary peak
        if GMM triggers → select_balanced_model()
          fit n=2 via fit_two_component_multi_start()
            for each init strategy:
              fit_mixture_from_init_peaks()
                least_squares → _merge_close_components()
                if n_components < 2: attempt discarded
          optionally fit n=3 (detector init, no multi-start)
          _compare_single_vs_mixture()
        CandidateFilter.evaluate_mixture_components() or single path
  build_under_split_report()
  export CSV/JSON/diagnostics
```

---

# Appendix B — Phase 1 technical plan (for review)

## B.1 Current initialization strategies

**File:** `gmm_multi_start.py`

| Function | Role |
|----------|------|
| `generate_two_component_init_sets()` | Builds named 2-comp init peak sets |
| `ordered_multi_start_strategies()` | Execution order, capped by `gmm_max_multi_starts` |
| `fit_two_component_multi_start()` | Loop strategies, pick lowest BIC converged fit |
| `fit_mixture_from_init_peaks()` | Shared optimizer + merge for each attempt |

**Existing strategy keys:**

- `detector_based` — top peaks by intensity (pads with offset clones if needed)
- `symmetric_x_sep{N}`, `symmetric_y_sep{N}`, `offset_x_sep{N}`, `offset_y_sep{N}` — anchored on **brightest peak only**
- `major_axis` — object major axis endpoints
- `residual_peak` — brightest peak + argmax of **single-fit** residual patch

**Execution order today:** `detector_based` → `residual_peak` → `major_axis` → symmetric → offset.

**Weak point:** When multiple distinct filtered peaks exist, geometric inits still anchor on `peaks[0]`. Peak combinations are not enumerated.

## B.2 Local maxima → GMM init trace

| Step | File / function |
|------|-----------------|
| Image detection | `candidate_detector.py` → `pipeline.run()` |
| Assignment | `peak_assignment.assign_peaks_to_objects()` |
| Routing | `object_router.classify()` |
| Peak source | `object_processor.process_suspicious()`: assigned peaks **or** `MaximaDetector.detect()` |
| Single fit (feeds residual_peak init) | `EllipticalGaussianFitter.fit_peak(peaks[0])` |
| Model selection | `GaussianModelSelector.select_balanced_model()` |
| 2-comp init | `generate_two_component_init_sets(peaks, ..., single_component=single)` |
| 3-comp init | `GaussianMixtureFitter._initial_peaks()` → top-N peaks only |

## B.3 Merge / collapse

**File:** `gaussian_fitter.py` → `_merge_close_components()`

Called inside `fit_mixture_from_init_peaks()` after every optimizer run.

Rules:

- Drop component if center distance `< gmm_min_component_separation` (default 1.5 px)
- Drop weaker component if amplitude ratio `< gmm_merge_amplitude_ratio` (default 0.12)

On collapse: `fit_error = "post_merge_collapsed_{N}_to_{M}"`, attempt treated as non-converged in multi-start loop.

**No retry after collapse in production today.**

## B.4 After collapse

| Location | Behavior |
|----------|----------|
| `fit_two_component_multi_start()` | `continue` — attempt not counted as converged |
| All attempts fail | Re-runs `detector_based` fallback; returns fit with `n_starts_converged=0` |
| `select_balanced_model()` | If all collapsed → `no_successful_multi_component_fit` or single kept |
| `under_split_report._classify_failure()` | `gmm_tried_but_components_collapsed` if model_selection contains `collapsed_to_one` |
| `pipeline.py` gmm_init_diagnostics export | Only if `mixture.init_attempts` non-empty on **best** mixture; all-failed cases may export nothing |

## B.5 Residual peaks today

| Residual source | Used for |
|-----------------|----------|
| Single-fit `GaussianComponent.residual_patch` | Init strategy `residual_peak`; GMM trigger `structured_residual_two_lobes` |
| Mixture `MixtureFitResult.residual_patch` | Exported in diagnostics overlays; **not** used for init or N+1 expansion in production |
| `_has_structured_residual()` | Binary trigger only — not used to seed missing components after collapse |

## B.6 Proposed Phase 1 changes (pending approval)

### Change 1 — Peak-combination initialization

| | |
|-|-|
| **Current** | 2-comp inits use brightest peak + geometric offsets or single+residual pair |
| **Why it fails** | True second punctum may be P2/P3, not a symmetric offset from P1; optimizer collapses to one basin |
| **Proposal** | Add `peak_pair_{i}_{j}` (and capped triples for 3-comp later) from filtered peaks, ranked by intensity sum + separation, inserted early in `ordered_multi_start_strategies()` |
| **Validation** | `sep_benchmark_sep3_seed1010` multi_start_converged > 0; seed101 unchanged; oracle gap shrinks; false-split benchmark over_split_rate stable |
| **Over-split risk** | Low for 2-comp if merge + CandidateFilter unchanged; pairs from same object may over-init on noise peaks — cap count via `gmm_peak_combination_max` |

**Files:** `gmm_multi_start.py` (`generate_two_component_init_sets`, `ordered_multi_start_strategies`), `config.py` (new cap flag, default conservative)

### Change 2 — Improved residual-based initialization

| | |
|-|-|
| **Current** | `residual_peak` uses single-fit residual only; mixture residual unused |
| **Why it fails** | After partial single-fit, residual may not highlight second punctum; duplicate init at patch center observed in failed cases |
| **Proposal** | (Phase 1b) Add mixture-residual argmax init when single-fit residual_peak duplicates first center; require min separation from existing init centers |
| **Validation** | Unit test on synthetic doublet with assigned peaks; failed-case regression; no change on clean singles |
| **Over-split risk** | Medium if residual noise peaks seed extra components — gate on residual significance + min separation |

**Files:** `gmm_multi_start.py`, possibly helper in new `residual_analysis.py` (shared, read-only heuristics)

### Change 3 — Collapse-aware retry

| | |
|-|-|
| **Current** | Collapsed attempt discarded; search continues to next strategy; no targeted retry |
| **Why it fails** | Optimizer converged but merge removed component — correct basin may exist with different second seed |
| **Proposal** | After collapse (`pre_merge 2 → post_merge 1`), one bounded retry: alternate peak pair or residual seed ≥ `gmm_min_component_separation` from first center; record as `{strategy}_collapse_retry` |
| **Validation** | Synthetic close-doublet test; sep3_seed1010; attempt diagnostics show retry; seed101 no regression |
| **Over-split risk** | Low if retry only on collapse + under-split evidence; avoid retry on clean singles |

**Files:** `gmm_multi_start.py` (`fit_two_component_multi_start`, new helpers), `config.py` (`gmm_collapse_retry_enabled`, `gmm_collapse_retry_max`)

### Change 4 — Coarse search for hard cases (optional Phase 1d)

| | |
|-|-|
| **Current** | Only fixed strategy templates |
| **Why it fails** | Hard 2-comp objects need combinatorial center hypotheses |
| **Proposal** | For suspicious objects with ≥2 filtered peaks and 0 converged starts after staged pass, score limited center pairs (maxima + residual maxima) by cheap RSS proxy, optimize top-K only |
| **Validation** | Runtime budget test; separation failures recovered; false-split unchanged |
| **Over-split risk** | Medium — must be gated to hard objects only |

**Files:** `gmm_multi_start.py`, possibly `object_router.py` metadata (complexity hint)

## B.7 Oracle vs production mismatch

| Aspect | Production | Oracle (`validation/gmm_oracle.py`) |
|--------|------------|--------------------------------------|
| Init source | Detected peaks + geometric strategies | Ground-truth JSON centers |
| Fit path | `fit_mixture_from_init_peaks()` | `fit_oracle_mixture_from_init_peaks()` (validation-only copy) |
| Pre-merge capture | Not retained | Retained for reporting |
| Optimizer metadata | runtime, nfev | success, status, bounds_hit, cost, etc. |
| Wired to pipeline | Yes | No |

**Rule:** Do not copy oracle ground-truth init into production. Do reuse **ideas** (residual seeding, collapse retry logic) with detected peaks only.

## B.8 Existing tests protecting against false splitting

| Test | File |
|------|------|
| `test_single_gaussian_not_selected_as_two_components` | `test_gmm_multi_start.py` |
| `test_noisy_single_gaussian_mostly_not_oversplit` | `test_gmm_multi_start.py` |
| `test_fast_path_unchanged_with_multi_start_enabled` | `test_gmm_multi_start.py` |
| `test_balanced_mode_skips_gmm_for_clean_single` | `test_puncta_declump.py` |
| False-split benchmark (script-level) | `scripts/run_synthetic_benchmarks.py --benchmark false_split` |

## B.9 Proposed new tests (before production changes)

1. **`test_peak_pair_init_present_when_multiple_peaks`** — `generate_two_component_init_sets` includes ranked peak pairs; order prioritizes pairs after `detector_based`.
2. **`test_collapse_retry_records_second_attempt`** — forced collapse init → retry strategy recorded; recovery on close doublet.
3. **`test_sep3_seed1010_multi_start_converges`** — generated case, production path, `multi_start_converged > 0`.
4. **`test_sep3_seed101_control_no_regression`** — same harness, still exact-count or ≥1 accepted.
5. **`test_oracle_gap_metric`** — helper comparing converged count / BIC: production init vs oracle init on same patch (validation module).
6. **`test_false_split_benchmark_smoke`** — optional CI subset of false_split cases, over_split_rate ≤ baseline + ε.

## B.10 Runtime implications (Phase 1)

| Change | Cost |
|--------|------|
| Peak-pair inits | +O(k) optimizer runs, k capped by `gmm_peak_combination_max` and `gmm_max_multi_starts` |
| Collapse retry | +0–1 run per collapsed attempt, max `gmm_collapse_retry_max` |
| Coarse search (if added) | Significant — must be gated to hard objects only |
| Early stop (Phase 6) | Deferred until accuracy validated |

---

# Appendix C — Future phase file map (preview)

| Phase | Primary files |
|-------|---------------|
| 2 Dynamic N | `gaussian_fitter.py`, `object_router.py`, `object_processor.py` |
| 3 Residual N+1 | `gaussian_fitter.py`, `object_processor.py`, `residual_analysis.py` (new shared heuristics) |
| 4 Component diagnostics | `candidate_filter.py`, `types.py`, `export.py` |
| 5 Real validation | `real_validation_data/`, `scripts/evaluate_real_roi.py`, `scripts/annotate_real_roi.py` |
| 6 Runtime | `gmm_multi_start.py`, `config.py` |
