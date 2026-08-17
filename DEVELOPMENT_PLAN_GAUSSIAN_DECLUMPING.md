# Gaussian/GMM Puncta Declumping — Development Plan

This document summarizes the current state of the puncta declumping pipeline, confirmed limitations, and a phased roadmap for improving robustness on complex overlapping puncta and dense cloud-like objects.

**Status:** Phase A (peak-pair init) implemented. **Phase B is default production behavior.** Phase C (iterative dynamic K) is implemented but **disabled by default** as an optional dense-overlap fallback. Do not start Phase D until Phase B/C routing is stable in production.

See `PHASE_B_C_PRODUCTION_MODES.md` for config, validation conclusion, and routing details.

---

# Strategic pivot (2026-08-17)

## What changed

The pipeline already has sufficient initialization diversity:

| Strategy | Role |
|----------|------|
| `detector_based` | Top peaks by intensity |
| `peak_pair_{i}_{j}` | Phase 1A — ranked detected peak combinations |
| `residual_peak` | Single-fit residual argmax (profiling shows this is often strong) |
| `major_axis` | Object geometry |
| `symmetric_*`, `offset_*` | Geometric fallbacks at fixed separations |

**Stop adding init strategies.** Profiling shows residual-based init is already competitive; the remaining gaps are not "missing one more start template" but:

1. **Component count is fixed** (essentially 1 → try 2 → maybe 3) instead of evidence-driven N
2. **Residual map is underused** — only seeds one init, not used to decide split / N+1
3. **Merge is post-hoc cleanup**, not part of model refinement
4. **Acceptance relies too heavily on fixed separation** (1.5 px) instead of multi-criterion validity
5. **Runtime is unacceptable** on hard objects (7–14+ s) because all 15–20 starts run even when a good solution exists early

## New core thesis

> Upgrade from **"try many initializations for a mostly fixed 2/3-component GMM"** to **"fit → inspect residual → selectively split → dynamically change N → reject physically implausible components → early stop"**.

Inspired by:

- **Pan et al. 2010** — split/merge driven by fitting error; component count ≠ initial maxima count
- **G5M 2026** — BIC + σ constraints + local support + likelihood + resolvability for component validity

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

# Development phases (revised roadmap)

## Phase A — Stabilize current code ✅ (in progress)

**Goal:** Lock in Phase 1A peak-pair init; fix benchmark/control-flow issues; **do not add more init strategies.**

| Item | Status |
|------|--------|
| Peak-pair initialization (`gmm_peak_combination_max`, `_rank_peak_pairs`) | ✅ Implemented |
| Strategy ordering (detector → peak_pair → residual → geometry) | ✅ Implemented |
| GMM init diagnostics on failure | ✅ Implemented |
| Under-split categorization fix | ✅ Implemented |
| Phase 1A tests (`tests/test_phase1a_peak_combination_init.py`) | ⏳ Awaiting full run |
| Aggregate benchmark before/after | ⏳ Pending |
| Benchmark runtime / control-flow fixes | ⏳ Pending |

**Explicitly out of scope for Phase A:**

- Phase 1B/C/D (more init strategies, collapse retry, coarse search)
- Dynamic N
- Residual-guided splitting loop

---

## Phase B — Residual-guided splitting ✅ **DEFAULT PRODUCTION**

**Status:** ✅ Implemented and wired to production (`residual_refiner.py`, `gaussian_fitter.py`). **Default production mode** — one gated N→N+1 residual split after `select_balanced_model()`.

**Real-image validation (2026-08):** Phase B improved several difficult objects (example1: objects 260, 574, 642 recovered from 0→2 accepted) and produced overlays that looked more reliable overall than baseline OFF. This conclusion is based on real-image validation, not formal ground-truth accuracy.

**Production config (default):**

```python
residual_split_enabled = True
dynamic_model_order_enabled = False
residual_split_max_iterations = 1
```

**Goal:** Use residual evidence to decide **when** and **where** to add one component — not blind try N=1,2,3.

**Current gap:** `residual_peak` init uses single-fit residual only. Mixture `residual_patch` is exported in diagnostics but **not** used for init or N+1 expansion in production.

**Target control flow:**

```
1-Gaussian / N-GMM fit
    ↓
Compute residual map (observed − predicted)
    ↓
Structured residual peak present?
    ↓ yes
Identify worst-fit component (highest local error / lowest contribution)
    ↓
Seed new center near residual peak (≥ min separation from existing)
    ↓
Refit N+1
    ↓
Compare: BIC improvement? residual still structured? component physically valid?
    ↓
Accept N+1 or keep N
```

**Key design rules (Pan-inspired):**

- Increase N **only when residual evidence supports it** — not enumerate 1..5 on every object
- Local maxima count may be wrong (too few or too many); residual is the tie-breaker
- Gate on residual significance (prominence, integrated positive mass), not raw argmax noise

**Primary files:**

| File | Change |
|------|--------|
| `gaussian_fitter.py` | Residual map after N-comp fit; split decision |
| New `residual_analysis.py` (shared) | Structured peak detection, significance gating |
| `gmm_multi_start.py` | Residual-seeded init for N+1 refit (not just 2-comp multi-start) |
| `object_processor.py` | Wire iterative split loop |

**Tests before production:**

- Synthetic doublet where detector returns 1 peak → residual split recovers 2
- Synthetic triple where 2-GMM residual shows third lobe → N=3 only when justified
- Clean single → no residual-driven split (false-split gate)
- Mixture residual used (not only single-fit residual)

---

## Phase C — Dynamic model order (gated N) ✅ **OPTIONAL / OFF BY DEFAULT**

**Status:** ✅ Implemented (`residual_refiner.py`, iterative loop in `residual_split.py`). Kept behind `dynamic_model_order_enabled=False` by default.

**Real-image validation (2026-08):** Phase C successfully enabled additional growth (e.g. 3→4 on example2 object 307) and helped some dense objects, but in side-by-side overlay comparisons it did **not consistently** produce better final overlays than Phase B. Phase C remains useful as an experimental / selective dense-overlap fallback.

**Optional config (explicit opt-in):**

```python
dynamic_model_order_enabled = True
dynamic_model_order_max_iterations = 3
residual_split_max_components = 4
```

**Goal:** Evidence-driven iterative growth beyond Phase B's single step, for especially dense or under-split objects.

**Current:** `select_balanced_model()` tries 2, then 3 if peaks ≥ 3 or 2-comp still poor. Caps at `gmm_max_components` (3) or `gmm_max_components_large` (5).

**Target:**

```
Start N = max(detected maxima, current accepted model)
    ↓
Fit N (with bounded multi-start for N=2; direct init for N≥3)
    ↓
Phase B residual check → N+1 if evidence strong
    ↓
Stop when:
  - residual improvement below threshold
  - BIC no longer improves (margin)
  - new component fails validity (Phase D)
  - N reaches cap (object-size dependent)
```

**Runtime guard:** Do **not** run full multi-start for every N increment. Use:

- Multi-start only for N=2 (hardest basin problem)
- Residual-seeded direct init for N+1 splits
- BIC comparison between N and N+1 after each step

**Primary files:** `gaussian_fitter.py` (`GaussianModelSelector`), `config.py` (stop thresholds)

---

## Phase D — Stronger component validity (G5M-inspired)

**Goal:** Replace separation-only acceptance with multi-criterion component validation.

**Current CandidateFilter checks:**

- σ bounds, amplitude, R², residual relative, center shift, center inside mask
- Duplicate distance: `gmm_acceptance_min_separation` (1.5 px) — **too dominant**

**Target validity (all must pass or component rejected):**

| Criterion | Rationale |
|-----------|-----------|
| σ in plausible range | Reject hallucinated sharp/diffuse components |
| Amplitude / integrated intensity sufficient | Reject noise peaks |
| Local pixel support around center | Component explains real signal, not optimizer artifact |
| Center inside object / sensible neighborhood | Already partially implemented |
| Adding component improves BIC or residual enough | Model-level justification |
| Separation physically resolvable | Not fixed 1.5 px alone — normalized by σ or FWHM |

**Primary files:** `candidate_filter.py`, `types.py` (new diagnostic fields), `export.py`

**Rule:** No threshold tuning until Phase D diagnostics are exported and visible.

---

## Phase E — Merge/refit loop (model refinement)

**Goal:** Upgrade merge from post-fit cleanup to active model refinement.

**Current:** `_merge_close_components()` drops components < 1.5 px or weak amplitude ratio **after** fit. No refit. Collapsed attempts discarded.

**Target:**

```
Fit N components
    ↓
Two components collapsed / too close / one too weak?
    ↓ yes
Merge pair → refit N−1
    ↓
Compare BIC / residual vs pre-merge N
    ↓
Keep better model
```

Pan-style split/merge/fit/compare loop integrated with Phase B (split) and Phase C (dynamic N).

**Primary files:** `gaussian_fitter.py` (`_merge_close_components`, new `_merge_and_refit`)

---

## Phase F — Runtime optimization

**Goal:** Reduce per-object GMM time without sacrificing accuracy on hard cases.

**Current:** Up to 20 strategies × 3000 nfev each; hard objects 7–14+ s. `staged_early_stop` exists but accuracy not yet validated with it.

**Target:**

```
Cheap / historically strong strategies first
  (detector_based, peak_pair_*, residual_peak)
    ↓
Valid excellent solution? (BIC beats single + components valid)
    ↓ yes → early stop
    ↓ no
Try expensive backup strategies (symmetric, offset, ...)
```

Additional (later):

- Parallelize independent objects (embarrassingly parallel per image)
- Optional Numba / vectorized residual evaluation
- Per-object complexity routing (skip GMM entirely on clean singles — already partially done)

**Primary files:** `gmm_multi_start.py`, `config.py`, `pipeline.py`

**Rule:** Do not enable aggressive early stop until Phase B–D accuracy is benchmarked.

---

## Phase G — Validation expansion

**Goal:** Measure what matters for the new model, not just sep2–6 recovery.

| Benchmark | Purpose |
|-----------|---------|
| Separation (sep2–6, 20 seeds) | Recovery by distance — **aggregate primary metric** |
| False-split (clean singles) | Over-split regression gate |
| **Hidden-component synthetic** (new) | 1 detected peak, 2–3 true spots — tests Phase B |
| **Separation normalized by σ** (new) | Resolvability in σ units, not raw pixels |
| Exact count / under-split / over-split rates | Per-object count agreement |
| Manually labeled real ROIs (20–50) | Biological ground truth |
| Oracle gap | Production init vs ground-truth init (secondary) |
| Runtime per object | Phase F gate |

**Tools:** `scripts/run_synthetic_benchmarks.py`, `scripts/evaluate_synthetic_puncta.py`, new hidden-component generator

**Do not optimize for one case (e.g. `sep3_seed1010`) in isolation.**

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

1. **Finish Phase A validation** — run Phase 1A tests + aggregate benchmarks (before/after)
2. **Design Phase B spec** — residual-guided split loop with tests-first approach
3. **Do not implement** Phase 1B/C/D (more init strategies) or Phase F aggressive early stop until Phase B accuracy is validated

**Highest-value next implementation:** Residual-guided deterministic split + gated dynamic N (Phases B + C together).

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

# Appendix B — Phase A (peak-pair init) — COMPLETE

## B.1 Initialization strategies (current)

**File:** `gmm_multi_start.py`

| Strategy key | Status | Role |
|--------------|--------|------|
| `detector_based` | Existing | Top peaks by intensity |
| `peak_pair_{i}_{j}` | **Phase 1A ✅** | Ranked peak combinations (intensity primary, separation tie-breaker) |
| `residual_peak` | Existing | Single-fit residual argmax — often strong in profiling |
| `major_axis` | Existing | Object geometry endpoints |
| `symmetric_*`, `offset_*` | Existing | Geometric fallbacks |

**Execution order:** `detector_based` → `peak_pair_*` → `residual_peak` → `major_axis` → symmetric → offset

**Ranking (`_rank_peak_pairs`):**

1. Reject pairs below `gmm_acceptance_min_separation`
2. Rank by combined intensity (descending)
3. Separation as tie-breaker only
4. No fallback if all pairs too close — let residual/geometry handle it

**Cap:** `gmm_peak_combination_max = 6`, global `gmm_max_multi_starts = 20`

## B.2 Cancelled Phase 1 sub-items (do not implement)

| Original plan | Status | Reason |
|---------------|--------|--------|
| 1B — More residual init variants | ❌ Cancelled | Superseded by Phase B residual-guided split loop |
| 1C — Collapse-aware retry | ❌ Cancelled | Superseded by Phase E merge/refit loop |
| 1D — Coarse combination search | ❌ Cancelled | Init strategy saturation; runtime cost |
| More peak-pair / triple init | ❌ Cancelled | Experimental clarity; move to dynamic N |

## B.3 Phase 1A validation criteria

See `PHASE1A_IMPLEMENTATION_COMPLETE.md`. Success based on **aggregate** benchmarks:

- False-split rate ≤ baseline + 1%
- sep3 or sep4 recovery ≥ baseline + 5%
- Median `multi_start_converged` ≥ baseline
- Runtime ≤ baseline × 1.5
- All critical tests (1–11) pass

`sep3_seed1010` is a regression case only, not primary success criterion.

---

# Appendix C — Code reference (current gaps → Phase B/C targets)

## C.1 Local maxima → GMM init trace

| Step | File / function |
|------|-----------------|
| Image detection | `candidate_detector.py` → `pipeline.run()` |
| Assignment | `peak_assignment.assign_peaks_to_objects()` |
| Routing | `object_router.classify()` |
| Peak source | `object_processor.process_suspicious()`: assigned peaks **or** `MaximaDetector.detect()` |
| Single fit (feeds residual_peak init) | `EllipticalGaussianFitter.fit_peak(peaks[0])` |
| Model selection | `GaussianModelSelector.select_balanced_model()` |
| 2-comp init | `generate_two_component_init_sets()` + `fit_two_component_multi_start()` |
| 3-comp init | `GaussianMixtureFitter._initial_peaks()` → top-N peaks only, **no multi-start** |

## C.2 Merge / collapse (Phase E will upgrade this)

**File:** `gaussian_fitter.py` → `_merge_close_components()`

- Drop if center distance `< gmm_min_component_separation` (1.5 px)
- Drop weaker if amplitude ratio `< gmm_merge_amplitude_ratio` (0.12)
- On collapse: attempt discarded; **no refit**

## C.3 Residual usage gap (Phase B target)

| Residual source | Used today | Phase B target |
|-----------------|------------|----------------|
| Single-fit `GaussianComponent.residual_patch` | `residual_peak` init; GMM trigger | Keep; also feed split decision |
| Mixture `MixtureFitResult.residual_patch` | Diagnostics only | **Drive N+1 split + new center seed** |
| `_has_structured_residual()` | Binary GMM trigger | Extend: significance + integrated mass gating |

## C.4 Oracle vs production

| Aspect | Production | Oracle (`validation/gmm_oracle.py`) |
|--------|------------|--------------------------------------|
| Init source | Detected peaks + strategies | Ground-truth JSON centers |
| Wired to pipeline | Yes | No (validation only) |

**Rule:** Do not copy oracle ground-truth init into production. Reuse **ideas** (residual seeding, split/merge logic) with detected peaks only.

## C.5 False-split protection tests

| Test | File |
|------|------|
| `test_single_gaussian_not_selected_as_two_components` | `test_gmm_multi_start.py` |
| `test_noisy_single_gaussian_mostly_not_oversplit` | `test_gmm_multi_start.py` |
| `test_peak_pair_*` (Phase 1A) | `test_phase1a_peak_combination_init.py` |
| False-split benchmark | `scripts/run_synthetic_benchmarks.py --benchmark false_split` |

---

# Appendix D — Phase file map

| Phase | Primary files |
|-------|---------------|
| A Stabilize | `gmm_multi_start.py`, `config.py`, tests |
| B Residual split | `gaussian_fitter.py`, `residual_analysis.py` (new), `object_processor.py` |
| C Dynamic N | `gaussian_fitter.py` (`GaussianModelSelector`), `config.py` |
| D Component validity | `candidate_filter.py`, `types.py`, `export.py` |
| E Merge/refit | `gaussian_fitter.py` |
| F Runtime | `gmm_multi_start.py`, `pipeline.py` |
| G Validation | `scripts/generate_synthetic_puncta.py`, `scripts/evaluate_synthetic_puncta.py`, real ROI tooling |

