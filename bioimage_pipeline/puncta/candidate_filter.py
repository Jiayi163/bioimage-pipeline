"""Quality filtering for punctum detection candidates."""

from __future__ import annotations

import math

import numpy as np

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.types import (
    DetectionPath,
    FitStatus,
    GaussianComponent,
    ObjectInfo,
    ObjectPatch,
    PeakCandidate,
    PunctumCandidate,
)


class CandidateFilter:
    """Apply rejection rules and deduplicate accepted puncta."""

    def __init__(self, config: PunctaDeclumpConfig) -> None:
        self.config = config
        self._accepted: list[PunctumCandidate] = []

    def reset(self) -> None:
        self._accepted.clear()

    @property
    def accepted(self) -> list[PunctumCandidate]:
        return list(self._accepted)

    def evaluate_component(
        self,
        obj: ObjectInfo,
        peak: PeakCandidate,
        component: GaussianComponent,
        *,
        candidate_id: int,
        component_id: int,
        path: DetectionPath,
        object_mask: np.ndarray,
        patch: ObjectPatch,
    ) -> PunctumCandidate:
        candidate = PunctumCandidate(
            object_id=obj.label,
            candidate_id=candidate_id,
            component_id=component_id,
            path=path,
            fit_status="fit_ok",
            initial_row=peak.row,
            initial_col=peak.col,
            fitted_row=component.fitted_row if component.fit_succeeded else None,
            fitted_col=component.fitted_col if component.fit_succeeded else None,
            center_shift=component.center_shift if component.fit_succeeded else None,
            sigma=component.sigma if component.fit_succeeded else None,
            sigma_row=component.sigma_row if component.fit_succeeded else None,
            sigma_col=component.sigma_col if component.fit_succeeded else None,
            width_fwhm=(component.width_fwhm_row + component.width_fwhm_col) / 2.0
            if component.fit_succeeded
            else None,
            amplitude=component.amplitude if component.fit_succeeded else None,
            background=component.background if component.fit_succeeded else None,
            residual_rmse=component.residual_rmse if component.fit_succeeded else None,
            residual_relative=component.residual_relative if component.fit_succeeded else None,
            r_squared=component.r_squared if component.fit_succeeded else None,
            model_score=component.model_score if component.fit_succeeded else None,
            n_components_in_model=component.n_components_in_model,
        )

        rejection = self._rejection_reason(candidate, component, object_mask, patch)
        if rejection is None:
            duplicate_reason = self._duplicate_reason(candidate)
            if duplicate_reason is None:
                candidate.accepted = True
                candidate.fit_status = "fit_ok"
                self._accepted.append(candidate)
            else:
                candidate.rejection_reason = duplicate_reason
                candidate.fit_status = "rejected_duplicate"
        else:
            candidate.rejection_reason = rejection
            candidate.fit_status = self._status_from_rejection(rejection)

        return candidate

    def accept_fast_peak(
        self,
        obj: ObjectInfo,
        peak: PeakCandidate,
        *,
        candidate_id: int,
        route_reason: str = "",
    ) -> PunctumCandidate:
        """Accept an image-level peak on the fast path without Gaussian fitting."""
        candidate = PunctumCandidate(
            object_id=obj.label,
            candidate_id=candidate_id,
            component_id=1,
            path="fast_single",
            fit_status="fit_ok",
            initial_row=peak.row,
            initial_col=peak.col,
            fitted_row=peak.row,
            fitted_col=peak.col,
            center_shift=0.0,
            amplitude=peak.intensity,
            accepted=True,
        )
        duplicate_reason = self._duplicate_reason(candidate)
        if duplicate_reason is not None:
            candidate.accepted = False
            candidate.fit_status = "rejected_duplicate"
            candidate.rejection_reason = duplicate_reason
        else:
            self._accepted.append(candidate)
        return candidate

    def accept_fallback(
        self,
        obj: ObjectInfo,
        peak: PeakCandidate,
        *,
        candidate_id: int,
        component_id: int,
        path: DetectionPath,
        prior_rejection: str | None = None,
    ) -> PunctumCandidate:
        candidate = PunctumCandidate(
            object_id=obj.label,
            candidate_id=candidate_id,
            component_id=component_id,
            path=path,
            fit_status="fit_failed_fallback",
            initial_row=peak.row,
            initial_col=peak.col,
            fitted_row=None,
            fitted_col=None,
            accepted=True,
            warning="fit_failed_used_brightest_pixel",
            rejection_reason=prior_rejection,
        )
        duplicate_reason = self._duplicate_reason(candidate)
        if duplicate_reason is not None:
            candidate.accepted = False
            candidate.fit_status = "rejected_duplicate"
            candidate.rejection_reason = duplicate_reason
        else:
            self._accepted.append(candidate)
        return candidate

    def _rejection_reason(
        self,
        candidate: PunctumCandidate,
        component: GaussianComponent,
        object_mask: np.ndarray,
        patch: ObjectPatch,
    ) -> str | None:
        if not component.fit_succeeded:
            return component.fit_error or "fit_failed"
        if (
            candidate.center_shift is not None
            and candidate.center_shift > self.config.max_center_shift + 0.15
        ):
            return "center_shift_too_large"
        if candidate.sigma_row is None or candidate.sigma_row < self.config.min_sigma:
            return "sigma_too_small"
        if candidate.sigma_col is None or candidate.sigma_col > self.config.max_sigma:
            return "sigma_too_large"
        if candidate.sigma_row is None or candidate.sigma_row > self.config.max_sigma:
            return "sigma_too_large"
        if candidate.sigma_col is None or candidate.sigma_col < self.config.min_sigma:
            return "sigma_too_small"
        if candidate.amplitude is None or candidate.amplitude < self.config.min_amplitude:
            return "amplitude_too_low"
        if candidate.residual_rmse is not None and candidate.amplitude is not None:
            relative_residual = candidate.residual_rmse / max(candidate.amplitude, 1.0)
            if relative_residual > self.config.max_fit_residual_relative:
                return "residual_too_high"
        if (
            self.config.max_fit_residual is not None
            and candidate.residual_rmse is not None
            and candidate.residual_rmse > self.config.max_fit_residual
        ):
            return "residual_too_high_absolute"
        if candidate.r_squared is not None and candidate.r_squared < self.config.min_r_squared:
            return "r_squared_too_low"
        if not self._center_inside_mask(candidate, object_mask, patch):
            return "center_outside_object_mask"
        return None

    @staticmethod
    def _status_from_rejection(rejection: str) -> FitStatus:
        if rejection == "amplitude_too_low":
            return "rejected_low_amplitude"
        if rejection == "center_outside_object_mask":
            return "rejected_outside_mask"
        if rejection == "duplicate_center_too_close":
            return "rejected_duplicate"
        return "rejected_bad_fit"

    def _center_inside_mask(
        self,
        candidate: PunctumCandidate,
        object_mask: np.ndarray,
        patch: ObjectPatch,
    ) -> bool:
        patch_row = candidate.fitted_row - patch.row_offset if candidate.fitted_row is not None else None
        patch_col = candidate.fitted_col - patch.col_offset if candidate.fitted_col is not None else None
        if patch_row is None or patch_col is None:
            return False
        row = int(round(patch_row))
        col = int(round(patch_col))
        if row < 0 or col < 0 or row >= object_mask.shape[0] or col >= object_mask.shape[1]:
            return False
        if object_mask[row, col]:
            return True
        radius = max(1, int(round(self.config.max_center_shift)))
        min_row = max(0, row - radius)
        max_row = min(object_mask.shape[0], row + radius + 1)
        min_col = max(0, col - radius)
        max_col = min(object_mask.shape[1], col + radius + 1)
        return bool(object_mask[min_row:max_row, min_col:max_col].any())

    def _duplicate_reason(self, candidate: PunctumCandidate) -> str | None:
        for accepted in self._accepted:
            distance = math.hypot(
                candidate.final_row - accepted.final_row,
                candidate.final_col - accepted.final_col,
            )
            if distance < self.config.min_center_separation:
                if self._should_replace_accepted(accepted, candidate):
                    self._accepted.remove(accepted)
                    return None
                return "duplicate_center_too_close"
        return None

    def _should_replace_accepted(
        self,
        existing: PunctumCandidate,
        challenger: PunctumCandidate,
    ) -> bool:
        if existing.fit_status == "fit_ok" and challenger.fit_status != "fit_ok":
            return False
        if challenger.fit_status == "fit_ok" and existing.fit_status != "fit_ok":
            return True
        existing_amp = existing.amplitude or 0.0
        challenger_amp = challenger.amplitude or 0.0
        if challenger_amp > existing_amp:
            return True
        if challenger_amp < existing_amp:
            return False
        existing_res = existing.residual_rmse if existing.residual_rmse is not None else float("inf")
        challenger_res = challenger.residual_rmse if challenger.residual_rmse is not None else float("inf")
        return challenger_res < existing_res
