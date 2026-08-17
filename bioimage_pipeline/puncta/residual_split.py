"""Phase B/C: residual-guided split proposal and dynamic model-order logic.

Phase B (default production): one gated N->N+1 residual split after initial model
selection. Configured via ``residual_split_enabled=True`` and
``dynamic_model_order_enabled=False``.

Phase C (optional fallback): iterative N -> N+1 growth up to
``residual_split_max_components``, enabled only when
``dynamic_model_order_enabled=True``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import ndimage

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.types import GaussianComponent, MixtureFitResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualSplitConfig:
    """Phase B/C thresholds. Defaults align with ``PunctaDeclumpConfig`` where possible."""

    # --- Residual significance (Section 1) ---
    min_peak_fraction_of_max: float = 0.35
    min_lobe_area_px: int = 4
    min_prominence_fraction: float = 0.15
    min_positive_mass_fraction: float = 0.08

    # --- Exclusion near existing centers (Section 2) ---
    exclusion_radius_px: float | None = None
    exclusion_sigma_multiplier: float = 0.75

    # --- Resolvability (Section 4) ---
    min_resolvability_sigma_units: float = 0.75

    # --- Acceptance (Section 4) ---
    bic_improvement_margin: float = 2.0
    min_residual_improvement_fraction: float = 0.05
    min_sigma: float = 0.5
    max_sigma: float = 4.0
    min_amplitude: float = 10.0
    min_local_support_fraction: float = 0.15
    local_support_radius_sigma: float = 1.0

    # --- Stop conditions (Section 5) ---
    max_components: int = 3

    # --- Runtime guardrails (Section 6) ---
    max_split_iterations: int = 2

    @classmethod
    def from_puncta_config(cls, config: PunctaDeclumpConfig) -> ResidualSplitConfig:
        """Build split config from puncta settings.

        Phase B (default): one N->N+1 step via ``effective_residual_split_max_iterations``
        and ``gmm_max_components + 1`` component cap.

        Phase C (``dynamic_model_order_enabled=True``): iterative growth up to
        ``residual_split_max_components`` using ``dynamic_model_order_max_iterations``.
        """
        return cls(
            bic_improvement_margin=config.gmm_bic_improvement_margin,
            min_sigma=config.min_sigma,
            max_sigma=config.max_sigma,
            min_amplitude=config.min_amplitude,
            max_components=config.effective_residual_split_max_components,
            exclusion_radius_px=config.gmm_acceptance_min_separation,
            max_split_iterations=config.effective_residual_split_max_iterations,
        )

    @classmethod
    def for_phase_c_fallback(cls, config: PunctaDeclumpConfig) -> ResidualSplitConfig:
        """Phase C limits for selective post-Phase-B fallback refinement."""
        base = cls.from_puncta_config(config)
        return cls(
            bic_improvement_margin=base.bic_improvement_margin,
            min_sigma=base.min_sigma,
            max_sigma=base.max_sigma,
            min_amplitude=base.min_amplitude,
            max_components=config.residual_split_max_components,
            exclusion_radius_px=base.exclusion_radius_px,
            max_split_iterations=config.dynamic_model_order_max_iterations,
        )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualPeakCandidate:
    """One structured positive-residual lobe suitable for split seeding."""

    row: float
    col: float
    peak_value: float
    lobe_area_px: int
    integrated_mass: float
    prominence: float
    label_id: int


@dataclass(frozen=True)
class SplitProposal:
    """Deterministic proposal to grow model order by one component."""

    current_n: int
    proposed_n: int
    new_center_row: float
    new_center_col: float
    seed_intensity: float
    residual_peak: ResidualPeakCandidate
    attribution_component_id: int | None = None
    reason: str = "structured_residual_peak"


@dataclass(frozen=True)
class SplitAcceptanceResult:
    """Outcome of evaluating an N+1 refit against acceptance criteria."""

    accepted: bool
    reason: str
    checks: dict[str, bool] = field(default_factory=dict)


@dataclass
class SplitLoopState:
    """Mutable state for the gated dynamic-N split loop."""

    current_n: int
    iterations: int = 0
    last_rejection_reason: str | None = None
    proposals_made: int = 0
    ambiguous: bool = False
    stop_reason: str = ""


# ---------------------------------------------------------------------------
# Section 1 — Residual significance
# ---------------------------------------------------------------------------


def is_positive_residual_structured(
    residual_patch: np.ndarray,
    object_mask: np.ndarray,
    *,
    config: ResidualSplitConfig,
) -> bool:
    """Return True if the masked positive residual contains a significant lobe."""
    return len(find_structured_residual_peaks(residual_patch, object_mask, [], config=config)) > 0


def find_structured_residual_peaks(
    residual_patch: np.ndarray,
    object_mask: np.ndarray,
    existing_centers: Sequence[tuple[float, float]],
    *,
    config: ResidualSplitConfig,
) -> list[ResidualPeakCandidate]:
    """Find significant positive-residual lobes, excluding near existing centers."""
    positive = np.clip(np.asarray(residual_patch, dtype=np.float64), 0.0, None)
    mask = np.asarray(object_mask, dtype=bool)
    positive = np.where(mask, positive, 0.0)

    peak_max = float(positive.max())
    if peak_max <= 0.0:
        return []

    threshold = config.min_peak_fraction_of_max * peak_max
    lobe_mask = positive > threshold
    if int(lobe_mask.sum()) < config.min_lobe_area_px:
        return []

    labeled, n_labels = ndimage.label(lobe_mask)
    if n_labels < 1:
        return []

    total_positive_mass = float(positive.sum())
    if total_positive_mass <= 0.0:
        return []

    exclusion_radius = _exclusion_radius(existing_centers, config)
    candidates: list[ResidualPeakCandidate] = []

    for label_id in range(1, n_labels + 1):
        component_mask = labeled == label_id
        area = int(component_mask.sum())
        if area < config.min_lobe_area_px:
            continue

        integrated_mass = float(positive[component_mask].sum())
        if integrated_mass / total_positive_mass < config.min_positive_mass_fraction:
            continue

        lobe_values = positive[component_mask]
        local_peak_idx = int(np.argmax(lobe_values))
        lobe_indices = np.argwhere(component_mask)
        rr, cc = lobe_indices[local_peak_idx]
        peak_value = float(positive[rr, cc])

        local_mean = float(lobe_values.mean())
        prominence = (peak_value - local_mean) / max(peak_value, 1e-6)
        if prominence < config.min_prominence_fraction:
            continue

        row = float(rr)
        col = float(cc)

        if _too_close_to_existing(col, row, existing_centers, exclusion_radius):
            continue

        candidates.append(
            ResidualPeakCandidate(
                row=row,
                col=col,
                peak_value=peak_value,
                lobe_area_px=area,
                integrated_mass=integrated_mass,
                prominence=prominence,
                label_id=label_id,
            )
        )

    return _sort_peaks_deterministically(candidates)


# ---------------------------------------------------------------------------
# Section 2 — Exclusion near existing component centers
# ---------------------------------------------------------------------------


def _exclusion_radius(
    existing_centers: Sequence[tuple[float, float]],
    config: ResidualSplitConfig,
) -> float:
    if config.exclusion_radius_px is not None:
        return config.exclusion_radius_px
    return config.exclusion_sigma_multiplier * 2.0


def _too_close_to_existing(
    col: float,
    row: float,
    existing_centers: Sequence[tuple[float, float]],
    exclusion_radius: float,
) -> bool:
    for existing_col, existing_row in existing_centers:
        if math.hypot(col - existing_col, row - existing_row) < exclusion_radius:
            return True
    return False


# ---------------------------------------------------------------------------
# Section 3 — Deterministic N -> N+1 split proposal
# ---------------------------------------------------------------------------


def propose_n_plus_one_split(
    *,
    current_n: int,
    residual_patch: np.ndarray,
    object_mask: np.ndarray,
    existing_components: Sequence[GaussianComponent],
    background_level: float = 0.0,
    config: ResidualSplitConfig,
) -> SplitProposal | None:
    """Propose exactly one N+1 split seed from structured residual evidence."""
    if current_n >= config.max_components:
        return None

    existing_centers = [(c.fitted_col, c.fitted_row) for c in existing_components]
    peaks = find_structured_residual_peaks(
        residual_patch,
        object_mask,
        existing_centers,
        config=config,
    )
    if not peaks:
        return None

    best_peak = peaks[0]
    attribution_id = _worst_fit_component_id(existing_components, residual_patch, object_mask)

    return SplitProposal(
        current_n=current_n,
        proposed_n=current_n + 1,
        new_center_row=best_peak.row,
        new_center_col=best_peak.col,
        seed_intensity=best_peak.peak_value + background_level,
        residual_peak=best_peak,
        attribution_component_id=attribution_id,
        reason="structured_residual_peak",
    )


def _worst_fit_component_id(
    components: Sequence[GaussianComponent],
    residual_patch: np.ndarray,
    object_mask: np.ndarray,
) -> int | None:
    """Identify the component contributing most local positive residual mass."""
    if not components:
        return None

    positive = np.clip(np.asarray(residual_patch, dtype=np.float64), 0.0, None)
    mask = np.asarray(object_mask, dtype=bool)
    scores: list[tuple[float, int]] = []

    for component in components:
        row = int(round(component.fitted_row))
        col = int(round(component.fitted_col))
        radius = max(2, int(round(max(component.sigma_row, component.sigma_col) * 1.5)))
        min_row = max(0, row - radius)
        max_row = min(positive.shape[0], row + radius + 1)
        min_col = max(0, col - radius)
        max_col = min(positive.shape[1], col + radius + 1)
        region = positive[min_row:max_row, min_col:max_col]
        region_mask = mask[min_row:max_row, min_col:max_col]
        score = float(region[region_mask].sum()) if region_mask.any() else 0.0
        scores.append((score, component.component_id))

    scores.sort(key=lambda item: (-item[0], item[1]))
    return scores[0][1] if scores else None


def _sort_peaks_deterministically(
    peaks: list[ResidualPeakCandidate],
) -> list[ResidualPeakCandidate]:
    return sorted(
        peaks,
        key=lambda p: (-p.integrated_mass, -p.peak_value, p.row, p.col, p.label_id),
    )


# ---------------------------------------------------------------------------
# Section 4 — Split acceptance criteria
# ---------------------------------------------------------------------------


def evaluate_split_acceptance(
    *,
    baseline: MixtureFitResult,
    candidate: MixtureFitResult,
    proposal: SplitProposal,
    config: ResidualSplitConfig,
) -> SplitAcceptanceResult:
    """Evaluate whether an N+1 refit should be accepted over the baseline N-fit."""
    checks: dict[str, bool] = {}

    checks["candidate_fit_succeeded"] = candidate.fit_succeeded
    checks["candidate_n_increased"] = candidate.n_components == proposal.proposed_n

    residual_improved = (
        candidate.fit_succeeded
        and baseline.fit_succeeded
        and math.isfinite(candidate.residual_rmse)
        and math.isfinite(baseline.residual_rmse)
        and candidate.residual_rmse
        <= baseline.residual_rmse * (1.0 - config.min_residual_improvement_fraction)
    )
    checks["residual_improved"] = residual_improved

    bic_improved = (
        candidate.fit_succeeded
        and baseline.fit_succeeded
        and math.isfinite(candidate.bic)
        and math.isfinite(baseline.bic)
        and candidate.bic + config.bic_improvement_margin < baseline.bic
    )
    checks["bic_improved"] = bic_improved

    new_component = _find_new_component(candidate, proposal)
    checks["new_component_found"] = new_component is not None

    sigma_valid = False
    amplitude_valid = False
    support_valid = False
    resolvable = False

    if new_component is not None:
        sigma_valid = (
            config.min_sigma <= new_component.sigma_row <= config.max_sigma
            and config.min_sigma <= new_component.sigma_col <= config.max_sigma
        )
        amplitude_valid = new_component.amplitude >= config.min_amplitude
        support_valid = _local_support_ok(new_component, candidate, config)
        resolvable = _resolvability_ok(new_component, candidate.components, config)

    checks["sigma_valid"] = sigma_valid
    checks["amplitude_valid"] = amplitude_valid
    checks["local_support_valid"] = support_valid
    checks["resolvable"] = resolvable

    model_quality_ok = residual_improved or bic_improved
    physical_ok = (
        checks["new_component_found"]
        and sigma_valid
        and amplitude_valid
        and support_valid
        and resolvable
    )

    accepted = (
        checks["candidate_fit_succeeded"]
        and checks["candidate_n_increased"]
        and model_quality_ok
        and physical_ok
    )

    if not accepted:
        reason = _rejection_reason(checks, model_quality_ok, physical_ok)
    else:
        reason = "accepted"

    return SplitAcceptanceResult(accepted=accepted, reason=reason, checks=checks)


def _find_new_component(
    candidate: MixtureFitResult,
    proposal: SplitProposal,
) -> GaussianComponent | None:
    if not candidate.components:
        return None
    best: GaussianComponent | None = None
    best_dist = float("inf")
    for component in candidate.components:
        dist = math.hypot(
            component.fitted_col - proposal.new_center_col,
            component.fitted_row - proposal.new_center_row,
        )
        if dist < best_dist:
            best_dist = dist
            best = component
    return best


def _local_support_ok(
    component: GaussianComponent,
    fit: MixtureFitResult,
    config: ResidualSplitConfig,
) -> bool:
    if fit.residual_patch is None or fit.predicted_patch is None:
        return True

    residual = np.asarray(fit.residual_patch, dtype=np.float64)
    predicted = np.asarray(fit.predicted_patch, dtype=np.float64)
    row = int(round(component.fitted_row))
    col = int(round(component.fitted_col))
    radius = max(1, int(round(component.sigma * config.local_support_radius_sigma)))
    min_row = max(0, row - radius)
    max_row = min(residual.shape[0], row + radius + 1)
    min_col = max(0, col - radius)
    max_col = min(residual.shape[1], col + radius + 1)

    region_pred = predicted[min_row:max_row, min_col:max_col]
    if region_pred.size == 0:
        return False

    threshold = max(float(region_pred.max()) * 0.2, config.min_amplitude * 0.1)
    support_fraction = float((region_pred > threshold).sum()) / float(region_pred.size)
    return support_fraction >= config.min_local_support_fraction


def _resolvability_ok(
    new_component: GaussianComponent,
    all_components: Sequence[GaussianComponent],
    config: ResidualSplitConfig,
) -> bool:
    others = [c for c in all_components if c.component_id != new_component.component_id]
    if not others:
        return True

    mean_sigma = float(np.mean([c.sigma for c in all_components]))
    if mean_sigma <= 0.0:
        mean_sigma = 1.0

    min_distance = float("inf")
    for other in others:
        dist = math.hypot(
            new_component.fitted_col - other.fitted_col,
            new_component.fitted_row - other.fitted_row,
        )
        min_distance = min(min_distance, dist)

    required = config.min_resolvability_sigma_units * mean_sigma
    return min_distance >= required


def _rejection_reason(
    checks: dict[str, bool],
    model_quality_ok: bool,
    physical_ok: bool,
) -> str:
    if not checks.get("candidate_fit_succeeded"):
        return "candidate_fit_failed"
    if not checks.get("candidate_n_increased"):
        return "candidate_n_not_increased"
    if not model_quality_ok:
        return "insufficient_model_improvement"
    if not checks.get("sigma_valid"):
        return "invalid_sigma"
    if not checks.get("amplitude_valid"):
        return "amplitude_too_low"
    if not checks.get("local_support_valid"):
        return "insufficient_local_support"
    if not checks.get("resolvable"):
        return "not_resolvable"
    return "rejected"


# ---------------------------------------------------------------------------
# Section 5 — Stop conditions
# ---------------------------------------------------------------------------


def should_stop_split_loop(
    state: SplitLoopState,
    *,
    config: ResidualSplitConfig,
) -> tuple[bool, str]:
    if state.current_n >= config.max_components:
        return True, "max_components_reached"
    if state.iterations >= config.max_split_iterations:
        return True, "max_split_iterations_reached"
    return False, ""


def mark_ambiguous_if_needed(
    state: SplitLoopState,
    *,
    rejection_reason: str | None,
    residual_patch: np.ndarray | None,
    object_mask: np.ndarray,
    existing_components: Sequence[GaussianComponent],
    config: ResidualSplitConfig,
) -> None:
    """Mark loop ambiguous when residual evidence remains but growth is blocked."""
    if rejection_reason == "not_resolvable":
        state.ambiguous = True
        state.stop_reason = "ambiguous_unresolvable_component"
        return

    if state.current_n >= config.max_components and residual_patch is not None:
        if is_positive_residual_structured(residual_patch, object_mask, config=config):
            state.ambiguous = True
            state.stop_reason = "ambiguous_max_components_with_residual"
            return

    if rejection_reason == "insufficient_model_improvement" and residual_patch is not None:
        if is_positive_residual_structured(residual_patch, object_mask, config=config):
            state.ambiguous = True
            state.stop_reason = "ambiguous_insufficient_improvement_with_residual"
            return

    if rejection_reason and rejection_reason not in {"accepted", "candidate_fit_failed"}:
        if residual_patch is not None and is_positive_residual_structured(
            residual_patch,
            object_mask,
            config=config,
        ):
            existing_centers = [(c.fitted_col, c.fitted_row) for c in existing_components]
            peaks = find_structured_residual_peaks(
                residual_patch,
                object_mask,
                existing_centers,
                config=config,
            )
            if peaks and rejection_reason in {
                "invalid_sigma",
                "amplitude_too_low",
                "insufficient_local_support",
                "not_resolvable",
            }:
                state.ambiguous = True
                state.stop_reason = f"ambiguous_{rejection_reason}"


def remaining_component_budget(state: SplitLoopState, config: ResidualSplitConfig) -> int:
    """How many more components may still be added in this loop."""
    return max(0, config.max_components - state.current_n)


def remaining_iteration_budget(state: SplitLoopState, config: ResidualSplitConfig) -> int:
    """How many more split iterations are allowed."""
    return max(0, config.max_split_iterations - state.iterations)


def should_propose_split(
    *,
    state: SplitLoopState,
    residual_patch: np.ndarray,
    object_mask: np.ndarray,
    existing_components: Sequence[GaussianComponent],
    config: ResidualSplitConfig,
) -> tuple[bool, str]:
    stop, stop_reason = should_stop_split_loop(state, config=config)
    if stop:
        return False, stop_reason

    if not is_positive_residual_structured(residual_patch, object_mask, config=config):
        return False, "no_structured_residual"

    proposal = propose_n_plus_one_split(
        current_n=state.current_n,
        residual_patch=residual_patch,
        object_mask=object_mask,
        existing_components=existing_components,
        config=config,
    )
    if proposal is None:
        return False, "no_valid_split_proposal"

    return True, "structured_residual_present"


# ---------------------------------------------------------------------------
# Section 6 — Runtime guardrails (helpers)
# ---------------------------------------------------------------------------


def split_loop_budget_remaining(state: SplitLoopState, config: ResidualSplitConfig) -> int:
    """Return remaining split iterations allowed for this object."""
    return max(0, config.max_split_iterations - state.iterations)
