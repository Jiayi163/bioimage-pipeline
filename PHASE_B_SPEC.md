# Phase B Specification: Residual-Guided Splitting

**Status:** Specification + testable pure functions implemented. **Not wired to production fitting.**

**Module:** `bioimage_pipeline/puncta/residual_split.py`  
**Tests:** `tests/test_phase_b_residual_guided_split.py` (13 tests, all passing)

---

## Overview

Phase B upgrades the GMM workflow from fixed model order (1 → try 2 → maybe 3) to **evidence-driven N+1 growth**:

```
Fit N components
    ↓
Compute residual map (observed − predicted, masked)
    ↓
Structured positive residual present?
    ↓ yes
Propose one deterministic N+1 split seed
    ↓
Refit N+1 (integration step — not implemented yet)
    ↓
Accept only if model + physical criteria pass
    ↓
Repeat until stop conditions
```

Inspired by Pan et al. 2010 (split on fitting error) and G5M 2026 (multi-criterion component validity).

**Production behavior is unchanged.** This spec lives in an isolated module until integration is approved.

---

## 1. Residual Significance Criteria

A positive residual lobe is **structured and significant** when **all** of the following hold (within `object_mask`):

| Criterion | Parameter | Default | Rationale |
|-----------|-----------|---------|-----------|
| Peak height | `min_peak_fraction_of_max` | 0.35 | Aligns with existing `_has_structured_residual` heuristic |
| Lobe area | `min_lobe_area_px` | 4 px | Reject single-pixel noise spikes |
| Prominence | `min_prominence_fraction` | 0.15 | Peak must stand out within its lobe: `(peak − lobe_mean) / peak` |
| Mass share | `min_positive_mass_fraction` | 0.08 | Lobe integrated mass ≥ 8% of total positive residual mass |

**Algorithm:**

1. `positive = clip(residual, 0)` masked by `object_mask`
2. Threshold at `min_peak_fraction_of_max × max(positive)`
3. Connected-component labeling (`scipy.ndimage.label`)
4. Filter lobes by area, mass share, prominence
5. Local peak = argmax within each surviving lobe

**Function:** `find_structured_residual_peaks()`, `is_positive_residual_structured()`

---

## 2. Residual Peak Exclusion Near Existing Centers

Candidate residual peaks are **rejected** if within `exclusion_radius` of any existing component center `(col, row)`:

```
exclusion_radius = exclusion_radius_px          # default: gmm_acceptance_min_separation (1.5)
                 OR exclusion_sigma_multiplier × 2.0   # fallback if px unset
```

**Rationale:** Prevents proposing a split that duplicates an already-fitted component. No fallback to near-duplicate peaks — if all lobes are excluded, **no split is proposed**.

**Function:** `_too_close_to_existing()`, applied inside `find_structured_residual_peaks()`

---

## 3. Deterministic N → N+1 Split Proposal

**Input:** current N, residual patch, object mask, existing components  
**Output:** `SplitProposal | None` — at most **one** proposal per call

**Steps:**

1. If `current_n >= max_components` → return `None`
2. Find structured residual peaks (Section 1–2)
3. If none → return `None`
4. **Deterministic ranking:** sort peaks by  
   `(-integrated_mass, -peak_value, row, col, label_id)`
5. Take rank-0 peak as split seed
6. Attribute to worst-fit component (highest local positive residual mass in component neighborhood)
7. Return `SplitProposal(current_n, current_n+1, seed center, seed intensity, ...)`

**Determinism guarantee:** Same inputs → same `SplitProposal` (tested).

**Function:** `propose_n_plus_one_split()`

**Integration note:** N+1 refit uses this seed directly — **no full multi-start** for split-driven growth (runtime guardrail).

---

## 4. Acceptance Criteria (N+1 refit vs baseline N)

An N+1 candidate is **accepted** only if **all** groups pass:

### Model quality (at least one required)

| Check | Condition |
|-------|-----------|
| Residual improved | `candidate.rmse ≤ baseline.rmse × (1 − min_residual_improvement_fraction)` |
| BIC improved | `candidate.bic + bic_improvement_margin < baseline.bic` |

Default `min_residual_improvement_fraction = 0.05` (5% RMSE reduction).

### Physical validity (all required)

| Check | Condition |
|-------|-----------|
| Fit succeeded | `candidate.fit_succeeded` |
| N increased | `candidate.n_components == proposed_n` |
| σ valid | `min_sigma ≤ σ_row, σ_col ≤ max_sigma` |
| Amplitude | `amplitude ≥ min_amplitude` |
| Local support | Fraction of pixels above threshold within `local_support_radius_sigma × σ` ≥ `min_local_support_fraction` |
| Resolvable | Min distance to nearest other center ≥ `min_resolvability_sigma_units × mean(σ)` |

Default `min_resolvability_sigma_units = 0.75` — separation normalized by σ, not fixed pixels.

**Rejection reasons:** `candidate_fit_failed`, `insufficient_model_improvement`, `invalid_sigma`, `amplitude_too_low`, `insufficient_local_support`, `not_resolvable`

**Function:** `evaluate_split_acceptance()`

**Important:** Model improvement alone is **insufficient** — physical validity must also pass (G5M-inspired).

---

## 5. Stop Conditions

The split loop stops when **any** condition is true:

| Condition | Reason string |
|-----------|---------------|
| `current_n >= max_components` | `max_components_reached` |
| `iterations >= max_split_iterations` | `max_split_iterations_reached` |
| No structured residual | `no_structured_residual` |
| No valid proposal (all peaks excluded) | `no_valid_split_proposal` |
| N+1 refit rejected by acceptance | `last_rejection_reason` stored in state |

**Functions:** `should_stop_split_loop()`, `should_propose_split()`

---

## 6. Runtime Guardrails

| Guardrail | Default | Purpose |
|-----------|---------|---------|
| `max_split_iterations` | 2 | At most 2 N→N+1 steps per object |
| One proposal per iteration | — | No combinatorial peak-pair explosion |
| Direct init for N+1 split | — | No multi-start on split-driven refit |
| `max_components` | 3 (from config) | Hard cap on model order |

**Function:** `split_loop_budget_remaining()`

**Deferred to Phase F:** Early stop on multi-start for N=2 initial fit.

---

## Test Coverage

| Test | Requirement |
|------|-------------|
| `test_hidden_doublet_proposes_n_plus_one_split` | 1 peak detected, residual proposes N=2 |
| `test_hidden_doublet_should_propose_split` | `should_propose_split` returns true |
| `test_clean_single_gaussian_no_split_proposed` | Clean single → no proposal |
| `test_two_component_model_proposes_third_component` | 2-comp + third residual → N=3 |
| `test_noisy_unstructured_residual_no_split` | Noise only → no peaks |
| `test_residual_peak_near_existing_center_excluded` | Too close → excluded |
| `test_split_rejected_when_physically_invalid_despite_residual_improvement` | Better RMSE/BIC but bad σ → reject |
| `test_split_rejected_when_not_resolvable` | Too close components → reject |
| `test_max_components_stops_split_proposal` | N=3 → no N=4 |
| `test_max_split_iterations_guardrail` | Iteration cap enforced |
| `test_split_proposal_is_deterministic` | Repeatable proposal |
| `test_hidden_doublet_patch_residual_is_structured` | Synthetic patch integration |

Run: `pytest tests/test_phase_b_residual_guided_split.py -v`

---

## Expected Integration Points (not implemented)

When Phase B is approved for production wiring:

### Primary integration

| Location | Change |
|----------|--------|
| `gaussian_fitter.py` → `GaussianModelSelector` | Replace/supplement fixed 2→3 try with split loop orchestration |
| `gaussian_fitter.py` → `_build_residual_patch()` | Already exists — reuse for mixture residual |
| `object_processor.py` → `process_suspicious()` | Call split loop after initial N-comp fit |
| `gmm_multi_start.py` | Keep multi-start for N=2 only; N+1 split uses `SplitProposal` direct init via `fit_mixture_from_init_peaks()` |

### Proposed control flow (post-integration)

```
process_suspicious():
  single = fit_peak(peaks[0])
  if not gmm_triggered: return single path

  state = SplitLoopState(current_n=1)
  fit = single_as_mixture_or_single()

  while not should_stop_split_loop(state):
      if not should_propose_split(state, fit.residual_patch, ...):
          break
      proposal = propose_n_plus_one_split(...)
      candidate = refit_n_plus_one(proposal)   # NEW integration function
      acceptance = evaluate_split_acceptance(baseline=fit, candidate=..., proposal=...)
      if acceptance.accepted:
          fit = candidate
          state.current_n += 1
      else:
          state.last_rejection_reason = acceptance.reason
          break
      state.iterations += 1

  CandidateFilter.evaluate_mixture_components(fit)
```

### Config wiring

| New / reused | Source |
|--------------|--------|
| `ResidualSplitConfig.from_puncta_config()` | Maps existing `PunctaDeclumpConfig` fields |
| Phase B-specific thresholds | Stay in `ResidualSplitConfig` until tuned — **do not change production thresholds based on tests yet** |

### Export / diagnostics

| Field | Location |
|-------|----------|
| Split proposals | New fields in `gmm_init_diagnostics.json` |
| Acceptance checks | Per-iteration in diagnostics |
| `last_rejection_reason` | `measurements.csv` / under-split report |

### Files NOT modified in this phase

- `gaussian_fitter.py` (production fit path)
- `object_processor.py` (pipeline routing)
- `candidate_filter.py` (production acceptance — Phase D will extend)
- `config.py` (no new production parameters yet)

---

## Relationship to Other Phases

| Phase | Relationship |
|-------|--------------|
| A (peak-pair init) | Unchanged — multi-start for N=2 initial fit |
| B (this spec) | Residual-guided N+1 growth |
| C (dynamic N) | Split loop **is** gated dynamic N — Phase C integration merges with B |
| D (component validity) | Extends acceptance criteria in `evaluate_split_acceptance()` |
| E (merge/refit) | Adds merge → refit N−1 branch inside split loop |
| F (runtime) | Early stop on N=2 multi-start, parallel objects |
| G (validation) | Hidden-component synthetic benchmark |

---

## Open Questions (for integration review)

1. Should `min_peak_fraction_of_max = 0.35` remain aligned with `_has_structured_residual` or diverge?
2. Is `max_split_iterations = 2` sufficient for 4–5 component clouds on large objects?
3. Should worst-fit component attribution influence **which** existing center to perturb (Pan-style split) vs only seeding a new center?
4. When N+1 split fails acceptance, should the loop try the **second-ranked** residual peak or stop immediately? (Current spec: stop — one proposal per iteration, deterministic.)

---

**Prepared:** 2026-08-17  
**Next step:** Review spec → approve integration → wire `refit_n_plus_one()` in production path
