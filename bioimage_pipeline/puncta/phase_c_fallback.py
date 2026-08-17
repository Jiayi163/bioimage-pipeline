"""Conservative Phase C fallback after default Phase B refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.residual_split import ResidualSplitConfig, is_positive_residual_structured
from bioimage_pipeline.puncta.types import GaussianComponent, MixtureFitResult, ObjectInfo, ObjectPatch


@dataclass(frozen=True)
class PhaseCFallbackDecision:
    """Whether selective Phase C residual growth should run after Phase B."""

    trigger: bool
    reason: str


def _mixture_relative_residual(mixture: MixtureFitResult) -> float | None:
    if not mixture.fit_succeeded:
        return None
    amp = max((component.amplitude for component in mixture.components), default=0.0)
    return mixture.residual_rmse / max(amp, 1.0)


def _residual_patch_from_selected(
    selected: MixtureFitResult | GaussianComponent,
) -> np.ndarray | None:
    if isinstance(selected, MixtureFitResult):
        return selected.residual_patch
    return selected.residual_patch


def evaluate_phase_c_fallback(
    *,
    config: PunctaDeclumpConfig,
    obj: ObjectInfo,
    single: GaussianComponent,
    selected: MixtureFitResult | GaussianComponent,
    patch: ObjectPatch,
    n_filtered_peaks: int,
    n_accepted: int,
    under_split_suspect: bool,
) -> PhaseCFallbackDecision:
    """Decide if Phase C iterative growth should run after Phase B + CandidateFilter.

    Conservative gate: requires unresolved multiplicity plus supporting evidence.
    """
    if not config.residual_split_enabled:
        return PhaseCFallbackDecision(False, "residual_split_disabled")
    if not config.dynamic_model_order_fallback_enabled:
        return PhaseCFallbackDecision(False, "fallback_disabled")
    if config.dynamic_model_order_enabled:
        return PhaseCFallbackDecision(False, "phase_c_global_enabled")

    multiplicity_unresolved = (
        n_filtered_peaks >= 3 and n_accepted < n_filtered_peaks
    ) or (n_filtered_peaks >= 2 and n_accepted == 0)
    if not multiplicity_unresolved:
        return PhaseCFallbackDecision(False, "multiplicity_resolved")

    if not under_split_suspect:
        return PhaseCFallbackDecision(False, "not_under_split_suspect")

    evidence: list[str] = []

    if obj.equivalent_diameter > config.single_spot_max_diameter:
        evidence.append("large_diameter")
    if obj.area > config.expected_single_spot_area * config.gmm_trigger_area_factor:
        evidence.append("large_area")

    if single.fit_succeeded:
        if single.r_squared < config.gmm_weak_fit_r_squared:
            evidence.append("weak_single_fit")
        if single.residual_relative > config.diagnostic_high_residual_relative:
            evidence.append("high_single_residual")

    if isinstance(selected, MixtureFitResult):
        rel = _mixture_relative_residual(selected)
        if rel is not None and rel > config.diagnostic_high_residual_relative:
            evidence.append("high_mixture_residual")

    residual_patch = _residual_patch_from_selected(selected)
    split_config = ResidualSplitConfig.from_puncta_config(config)
    if residual_patch is not None and is_positive_residual_structured(
        residual_patch,
        patch.object_mask,
        config=split_config,
    ):
        evidence.append("structured_residual")

    if not evidence:
        return PhaseCFallbackDecision(False, "no_supporting_evidence")

    return PhaseCFallbackDecision(
        True,
        f"multiplicity_gap={n_accepted}/{n_filtered_peaks};" + ";".join(evidence),
    )
