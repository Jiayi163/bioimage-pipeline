# Phase D: Component Validity / Residual Handling / Saturation-Aware Fitting

## Problem

For **joint GMM fits**, each mixture component currently inherits the **object-level** RMSE and R² from the full mixture fit (`gmm_multi_start.fit_mixture_from_init_peaks`). `CandidateFilter` then rejects components when:

```text
component.residual_rmse / component.amplitude > max_fit_residual_relative
```

This treats each component as if it must explain the **entire object ROI**. In multi-component fits on large, irregular, or saturated blobs, the joint RMSE is dominated by structure between peaks, mask irregularity, and clipped pixels — not by poor local fit near an individual component center. Good mixture components are therefore rejected with `residual_too_high` even when the joint model is the correct explanation.

Object 49 (example3) is a canonical case: K=2 selected, both components rejected at ~0.27 and ~0.42 relative joint residual despite correct routing and maxima.

## Design principles

1. **Do not relax global thresholds** (`max_fit_residual_relative`, `max K`, init strategies).
2. **Single-component fits unchanged** — global residual remains appropriate when one Gaussian must explain the object.
3. **Mixture components judged locally** — evidence gathered near the fitted center.
4. **Saturation-aware** — clipped/near-clipped pixels excluded from local residual scoring.
5. **Conservative mask tolerance** — subpixel centers slightly outside a binary mask edge may pass if local mask support and intensity evidence are strong.
6. **Ambiguous output preserved** when evidence remains insufficient after Phase D.

## Validity model (mixture components only)

When `component_validity_enabled=True` and `n_components_in_model > 1`:

### 1. Local component-support residual

Within a disk of radius `component_local_residual_radius_sigma × σ` centered on the fitted subpixel center (intersected with the object mask):

- Observed: background-corrected patch values.
- Predicted: full mixture prediction (`mixture.predicted_patch`).
- Residual: `observed − predicted` (same per-pixel joint residual, but **scoped to local support**).
- **Saturation exclusion**: pixels flagged as clipped/near-clipped in the ROI are excluded from RMSE/R² when `saturation_exclude_from_local_residual=True`.
- Metrics:
  - `local_rmse`
  - `local_residual_relative = local_rmse / amplitude`
  - `local_r_squared` within the local window

**Rejection:** `local_residual_relative > max_fit_residual_relative` → `residual_too_high`

This is mathematically appropriate because a mixture component is validated by how well the **joint model** explains data **where that component contributes**, not by requiring the component alone to explain the whole object.

### 2. Local support

Within the same local disk, render the component's own Gaussian contribution. Count the fraction of pixels where predicted contribution exceeds `max(0.2 × local_peak, 0.1 × min_amplitude)`.

**Rejection:** fraction < `component_local_min_support_fraction` → `insufficient_local_support`

### 3. Local R² (mixtures only)

**Rejection:** `local_r_squared < component_min_local_r_squared` → `r_squared_too_low`

Uses the same local window and saturation mask as local RMSE. Joint object-level R² is not applied per component.

### 4. Mask-boundary tolerance

Legacy rule: reject if rounded center pixel is outside the object mask and no pixel within `max_center_shift` neighborhood is inside.

Phase D rule for mixtures:
- If center pixel inside mask → pass.
- Else compute fraction of the disk `component_mask_support_radius_sigma × σ` that lies inside the mask.
- Pass if `mask_support_fraction ≥ component_min_mask_support_fraction` **and** local corrected peak ≥ `0.5 × min_amplitude`.
- Otherwise → `center_outside_object_mask`.

This accepts subpixel centers slightly outside jagged mask edges when nearby support is strong, without allowing clearly exterior centers.

### 5. Resolvability

Minimum center separation to any sibling component in the same mixture.

**Rejection:** separation < `0.85 × gmm_min_component_separation` → `not_resolvable`

### 6. Saturation detection

On ROI masked pixels (prefer `patch.raw`, else `corrected + background`):

- `clip_value = roi_max`
- Near-clip if `value ≥ clip_value × (1 − saturation_near_clip_margin)`
- `saturation_present` if near-clip fraction ≥ `saturation_near_clip_fraction`

Saturation does **not** auto-accept components; it only changes how residuals are scored. When saturation is present and local evidence after exclusion is still insufficient, rejection/ambiguity is preserved.

## Comparison: old vs new

| Aspect | Old (per-component) | Phase D (mixture) |
|--------|---------------------|-------------------|
| Residual scope | Full object joint RMSE | Local disk around center |
| Residual vs amplitude | Joint RMSE / component amp | Local RMSE / component amp |
| R² | Joint object R² | Local R² in support disk |
| Mask | Hard center + shift neighborhood | Tolerance disk + intensity evidence |
| Saturation | Ignored | Excluded from local scoring |
| Single-component | Global residual | Unchanged (global) |

## Configuration

New fields in `PunctaDeclumpConfig`:

| Field | Default | Purpose |
|-------|---------|---------|
| `component_validity_enabled` | `True` | Enable Phase D for mixture components |
| `component_local_residual_radius_sigma` | `1.25` | Local residual/support disk radius |
| `component_local_min_support_fraction` | `0.15` | Min predicted-support fraction |
| `component_min_local_r_squared` | `0.15` | Local R² floor (mixtures) |
| `component_mask_support_radius_sigma` | `1.0` | Mask tolerance disk |
| `component_min_mask_support_fraction` | `0.08` | Min in-mask fraction for edge centers |
| `saturation_near_clip_fraction` | `0.005` | ROI fraction to flag saturation |
| `saturation_near_clip_margin` | `0.02` | Relative margin below ROI max |
| `saturation_exclude_from_local_residual` | `True` | Exclude near-clip pixels from local RMSE |

## Integration point

`CandidateFilter.evaluate_mixture_components()` calls `evaluate_component_validity()` for each component when `n_components_in_model > 1`. Single-component and fast-path filtering use the existing global checks.
