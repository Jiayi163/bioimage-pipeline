"""Phase B/C: Residual-guided refinement for production pipeline.

Phase B (default): one gated N->N+1 residual split after initial model selection.
Phase C (optional, ``dynamic_model_order_enabled=True``): iterative N->N+1 growth
up to ``residual_split_max_components`` for dense overlap fallback.

Architecture:
    object_processor → GaussianModelSelector.select_balanced_model()
                    → ResidualSplitRefiner.refine()
                    → final model
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.residual_split import (
    ResidualSplitConfig,
    SplitLoopState,
    SplitProposal,
    evaluate_split_acceptance,
    mark_ambiguous_if_needed,
    propose_n_plus_one_split,
    should_propose_split,
    should_stop_split_loop,
)
from bioimage_pipeline.puncta.types import GaussianComponent, MixtureFitResult, ObjectPatch, PeakCandidate

if TYPE_CHECKING:
    from bioimage_pipeline.puncta.gaussian_fitter import GaussianMixtureFitter


@dataclass(frozen=True)
class SplitAttemptDiagnostic:
    """Diagnostics for one N+1 split attempt."""

    iteration: int
    old_n: int
    proposed_n: int
    residual_peak_row: float
    residual_peak_col: float
    residual_peak_mass: float
    old_bic: float
    old_rmse: float
    new_bic: float | None
    new_rmse: float | None
    accepted: bool
    rejection_reason: str | None
    runtime_s: float
    physical_checks: dict[str, bool] = field(default_factory=dict)


@dataclass
class RefinementResult:
    """Result of residual-guided dynamic model-order refinement."""

    final_model: MixtureFitResult | GaussianComponent
    split_triggered: bool
    split_attempts: list[SplitAttemptDiagnostic] = field(default_factory=list)
    total_split_runtime_s: float = 0.0
    final_n: int = 1
    initial_n: int = 1
    stop_reason: str = ""
    ambiguous: bool = False


class ResidualSplitRefiner:
    """Refine an initial model via gated iterative N -> N+1 residual splits."""

    def __init__(
        self,
        *,
        mixture_fitter: GaussianMixtureFitter,
        config: PunctaDeclumpConfig,
        split_config: ResidualSplitConfig | None = None,
    ) -> None:
        self.mixture_fitter = mixture_fitter
        self.config = config
        self.split_config = split_config or ResidualSplitConfig.from_puncta_config(config)

    def refine(
        self,
        *,
        initial_model: MixtureFitResult | GaussianComponent,
        patch: ObjectPatch,
        peaks: list[PeakCandidate],
    ) -> RefinementResult:
        """Refine initial model using gated dynamic model-order splitting."""
        if isinstance(initial_model, GaussianComponent):
            current_model = self._single_to_mixture(initial_model, patch)
            initial_n = 1
        else:
            current_model = initial_model
            initial_n = initial_model.n_components

        if not current_model.fit_succeeded or current_model.n_components < 1:
            return RefinementResult(
                final_model=initial_model,
                split_triggered=False,
                final_n=initial_n,
                initial_n=initial_n,
                stop_reason="initial_model_not_refineable",
            )

        state = SplitLoopState(current_n=current_model.n_components)
        attempts: list[SplitAttemptDiagnostic] = []
        total_runtime = 0.0

        while True:
            stop, stop_reason = should_stop_split_loop(state, config=self.split_config)
            if stop:
                state.stop_reason = stop_reason
                break

            current_model = self._refresh_residual_patch(current_model, patch)

            should_propose, propose_reason = should_propose_split(
                state=state,
                residual_patch=current_model.residual_patch,
                object_mask=patch.object_mask,
                existing_components=current_model.components,
                config=self.split_config,
            )

            if not should_propose:
                state.stop_reason = propose_reason
                break

            proposal = propose_n_plus_one_split(
                current_n=state.current_n,
                residual_patch=current_model.residual_patch,
                object_mask=patch.object_mask,
                existing_components=current_model.components,
                background_level=patch.background_level,
                config=self.split_config,
            )

            if proposal is None:
                state.stop_reason = "no_valid_split_proposal"
                break

            state.proposals_made += 1
            attempt_start = time.perf_counter()
            candidate = self._refit_n_plus_one(
                current_model=current_model,
                proposal=proposal,
                patch=patch,
            )
            attempt_runtime = time.perf_counter() - attempt_start
            total_runtime += attempt_runtime

            acceptance = evaluate_split_acceptance(
                baseline=current_model,
                candidate=candidate,
                proposal=proposal,
                config=self.split_config,
            )

            attempts.append(
                SplitAttemptDiagnostic(
                    iteration=state.iterations,
                    old_n=state.current_n,
                    proposed_n=proposal.proposed_n,
                    residual_peak_row=proposal.new_center_row,
                    residual_peak_col=proposal.new_center_col,
                    residual_peak_mass=proposal.residual_peak.integrated_mass,
                    old_bic=current_model.bic if current_model.fit_succeeded else float("inf"),
                    old_rmse=current_model.residual_rmse if current_model.fit_succeeded else float("inf"),
                    new_bic=candidate.bic if candidate.fit_succeeded else None,
                    new_rmse=candidate.residual_rmse if candidate.fit_succeeded else None,
                    accepted=acceptance.accepted,
                    rejection_reason=acceptance.reason if not acceptance.accepted else None,
                    runtime_s=attempt_runtime,
                    physical_checks=acceptance.checks,
                )
            )

            if acceptance.accepted:
                current_model = self._refresh_residual_patch(candidate, patch)
                state.current_n = current_model.n_components
                state.iterations += 1
                state.last_rejection_reason = None
                continue

            state.last_rejection_reason = acceptance.reason
            mark_ambiguous_if_needed(
                state,
                rejection_reason=acceptance.reason,
                residual_patch=current_model.residual_patch,
                object_mask=patch.object_mask,
                existing_components=current_model.components,
                config=self.split_config,
            )
            if not state.stop_reason:
                state.stop_reason = acceptance.reason
            break

        if not state.stop_reason and not attempts:
            state.stop_reason = "no_split_needed"

        return RefinementResult(
            final_model=current_model,
            split_triggered=len(attempts) > 0,
            split_attempts=attempts,
            total_split_runtime_s=total_runtime,
            final_n=current_model.n_components,
            initial_n=initial_n,
            stop_reason=state.stop_reason,
            ambiguous=state.ambiguous,
        )

    def _refit_n_plus_one(
        self,
        *,
        current_model: MixtureFitResult,
        proposal: SplitProposal,
        patch: ObjectPatch,
    ) -> MixtureFitResult:
        """Refit N+1 components using current centers + proposed residual center."""
        init_peaks: list[PeakCandidate] = []
        for component in current_model.components:
            init_peaks.append(
                PeakCandidate(
                    row=component.fitted_row,
                    col=component.fitted_col,
                    intensity=component.amplitude + patch.background_level,
                )
            )

        init_peaks.append(
            PeakCandidate(
                row=proposal.new_center_row,
                col=proposal.new_center_col,
                intensity=proposal.seed_intensity,
            )
        )

        from bioimage_pipeline.puncta.gmm_multi_start import fit_mixture_from_init_peaks

        return fit_mixture_from_init_peaks(
            self.mixture_fitter,
            patch,
            init_peaks,
            n_components=proposal.proposed_n,
            initialization_method=f"residual_split_iter{proposal.current_n}",
        )

    def _refresh_residual_patch(
        self,
        model: MixtureFitResult,
        patch: ObjectPatch,
    ) -> MixtureFitResult:
        """Always recompute mixture residual for the next split iteration."""
        component_params = [
            (
                component.amplitude,
                component.fitted_row,
                component.fitted_col,
                component.sigma_row,
                component.sigma_col,
            )
            for component in model.components
        ]

        residual_patch = self.mixture_fitter._build_residual_patch(patch, component_params)

        return MixtureFitResult(
            components=model.components,
            n_components=model.n_components,
            background=model.background,
            residual_rmse=model.residual_rmse,
            r_squared=model.r_squared,
            aic=model.aic,
            bic=model.bic,
            model_score=model.model_score,
            fit_succeeded=model.fit_succeeded,
            fit_error=model.fit_error,
            predicted_patch=model.predicted_patch,
            residual_patch=residual_patch,
            merge_notes=model.merge_notes,
            winning_init_strategy=model.winning_init_strategy,
            multi_start_attempts=model.multi_start_attempts,
        )

    def _single_to_mixture(
        self,
        single: GaussianComponent,
        patch: ObjectPatch,
    ) -> MixtureFitResult:
        """Convert single Gaussian to MixtureFitResult format for uniform handling."""
        n_points = max(int(patch.object_mask.sum()), 1)
        k_params = 6
        rss = single.residual_rmse**2 * n_points if single.fit_succeeded else float("inf")
        aic = n_points * np.log(max(rss / n_points, 1e-12)) + 2 * k_params
        bic = k_params * np.log(n_points) + n_points * np.log(max(rss / n_points, 1e-12))

        return MixtureFitResult(
            components=[single],
            n_components=1,
            background=single.background,
            residual_rmse=single.residual_rmse,
            r_squared=single.r_squared,
            aic=aic,
            bic=bic,
            model_score=bic,
            fit_succeeded=single.fit_succeeded,
            fit_error=single.fit_error,
            predicted_patch=single.predicted_patch,
            residual_patch=single.residual_patch,
        )
