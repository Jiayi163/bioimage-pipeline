# Phase B / Phase C Production Modes

## Summary

| Mode | Config | Default | Behavior |
|------|--------|---------|----------|
| **Phase B** | `dynamic_model_order_enabled=False` | **ON** (`residual_split_enabled=True`) | One gated N→N+1 residual split after initial model selection |
| **Phase C** | `dynamic_model_order_enabled=True` | **OFF** | Iterative N→N+1 growth up to K=4 for dense overlap fallback |
| **Phase C fallback** | `dynamic_model_order_fallback_enabled=True` | **ON** | After Phase B + filter, rerun iterative growth only for dense unresolved objects |

Phase B is the **default production path**. Phase C remains available as global opt-in (`dynamic_model_order_enabled=True`) or as **selective fallback** when Phase B leaves strong under-split evidence.

## Real-image validation conclusion (2026-08)

Based on current real-image validation (example1 EV, example2 Rabbit CD9), **not formal ground-truth accuracy**:

- **Phase B** improved several difficult objects and produced outputs that looked more reliable overall.
- **Phase C** successfully enabled additional growth (e.g. 3→4) and helped some dense objects, but in side-by-side overlay comparisons it did **not consistently** produce better final overlays than Phase B.
- **Therefore Phase B is currently preferred as the default production behavior.**
- Phase C remains useful as an experimental / selective dense-overlap fallback when explicitly enabled.

On example2, Phase B (`max_iterations=1`) and Phase C (`max_iterations=3`) produced **identical outputs** except where only one object triggered residual split; runtime differed only slightly.

## Configuration

```python
# Phase B (default production)
residual_split_enabled: bool = True
dynamic_model_order_enabled: bool = False
residual_split_max_iterations: int = 1

# Phase C (optional; off by default)
dynamic_model_order_enabled: bool = True   # enable explicitly
dynamic_model_order_max_iterations: int = 3
residual_split_max_components: int = 4     # only used when Phase C enabled
```

Effective limits are resolved via `PunctaDeclumpConfig` properties:

- `effective_residual_split_max_iterations` — Phase B uses `residual_split_max_iterations`; Phase C uses `dynamic_model_order_max_iterations`
- `effective_residual_split_max_components` — Phase B uses `gmm_max_components + 1`; Phase C uses `residual_split_max_components`

`ResidualSplitConfig.from_puncta_config()` maps these effective values into the split loop.

## Control flow

```
select_balanced_model()
    ↓
residual_split_enabled?
    ↓ yes
ResidualSplitRefiner.refine()
    ↓
Phase B: at most 1 N→N+1 step (default)
Phase C: up to 3 iterations, K≤4 (when enabled)
    ↓
final model
```

## Phase C selective fallback (default ON)

After Phase B refinement and CandidateFilter, unresolved dense objects may trigger **one additional** Phase C residual growth pass:

```python
dynamic_model_order_fallback_enabled: bool = True   # default
dynamic_model_order_enabled: bool = False           # global Phase C stays OFF
```

**Trigger requires all of:**

1. `dynamic_model_order_fallback_enabled=True` and global Phase C OFF
2. Unresolved multiplicity:
   - `n_filtered >= 3` and `n_accepted < n_filtered`, **or**
   - `n_filtered >= 2` and `n_accepted == 0`
3. `under_split_suspect=True` (post Phase B filter)
4. At least one evidence signal:
   - `large_diameter`, `large_area`, `weak_single_fit`, `high_single_residual`, `high_mixture_residual`, or `structured_residual`

Fallback runs `ResidualSplitRefiner` with Phase C limits (`max_iterations=3`, `max_components=4`), preserves physical checks, and may mark `ambiguous` at K=4 rather than forcing more components.

## Enabling global Phase C

```python
config = PunctaDeclumpConfig(
    residual_split_enabled=True,
    dynamic_model_order_enabled=True,
    dynamic_model_order_max_iterations=3,
    residual_split_max_components=4,
)
```

## Tests

- `tests/test_phase_c_fallback_trigger.py` — selective fallback triggers and limits
- `tests/test_phase_b_c_production_modes.py` — default routing and Phase B vs C limits
- `tests/test_phase_b_integration.py` — Phase B production integration
- `tests/test_phase_c_dynamic_model_order.py` — Phase C iterative growth (uses `dynamic_model_order_enabled=True`)

## Out of scope (not changed here)

- Phase D component validity expansion
- Threshold tuning, CandidateFilter rules, or GMM initialization strategies
