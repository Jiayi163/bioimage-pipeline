"""Quality filtering for punctum detection candidates."""

from __future__ import annotations

import math

import numpy as np

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.types import (
    DetectionPath,
    GaussianFitResult,
    ObjectInfo,
    PeakCandidate,
    PunctumCandidate,
)


class CandidateFilter:
    """Apply rejection rules and deduplicate accepted puncta."""

    def __init__(self, config: PunctaDeclumpConfig) -> None:
        self.config = config
        self._accepted: list[PunctumCandidate] = []

    def reset(self) -> None:
        """Clear accepted candidates for a new run."""
        self._accepted.clear()

    @property
    def accepted(self) -> list[PunctumCandidate]:
        return list(self._accepted)

    def evaluate(
        self,
        obj: ObjectInfo,
        peak: PeakCandidate,
        fit: GaussianFitResult,
        *,
        candidate_id: int,
        path: DetectionPath,
        object_mask: np.ndarray | None = None,
    ) -> PunctumCandidate:
        """Evaluate one candidate and optionally accept it."""
        center_shift = self._center_shift(peak, fit)
        candidate = PunctumCandidate(
            object_id=obj.label,
            candidate_id=candidate_id,
            path=path,
            initial_row=peak.row,
            initial_col=peak.col,
            fitted_row=fit.fitted_row if fit.fit_succeeded else None,
            fitted_col=fit.fitted_col if fit.fit_succeeded else None,
            center_shift=center_shift,
            sigma=fit.sigma if fit.fit_succeeded else None,
            width_fwhm=fit.width_fwhm if fit.fit_succeeded else None,
            amplitude=fit.amplitude if fit.fit_succeeded else None,
            background=fit.background if fit.fit_succeeded else None,
            residual_rmse=fit.residual_rmse if fit.fit_succeeded else None,
        )

        rejection = self._rejection_reason(candidate, fit, object_mask)
        if rejection is None:
            duplicate_reason = self._duplicate_reason(candidate)
            if duplicate_reason is None:
                candidate.accepted = True
                self._accepted.append(candidate)
            else:
                candidate.rejection_reason = duplicate_reason
        else:
            candidate.rejection_reason = rejection

        return candidate

    def accept_without_fit(
        self,
        obj: ObjectInfo,
        peak: PeakCandidate,
        *,
        candidate_id: int,
        path: DetectionPath,
        warning: str | None = None,
    ) -> PunctumCandidate:
        """Accept a brightest-pixel fallback when fitting is unavailable."""
        candidate = PunctumCandidate(
            object_id=obj.label,
            candidate_id=candidate_id,
            path=path,
            initial_row=peak.row,
            initial_col=peak.col,
            accepted=True,
            warning=warning,
        )
        duplicate_reason = self._duplicate_reason(candidate)
        if duplicate_reason is not None:
            candidate.accepted = False
            candidate.rejection_reason = duplicate_reason
        else:
            self._accepted.append(candidate)
        return candidate

    def _rejection_reason(
        self,
        candidate: PunctumCandidate,
        fit: GaussianFitResult,
        object_mask: np.ndarray | None,
    ) -> str | None:
        if fit.roi_touches_edge:
            return "roi_touches_image_edge"
        if not fit.fit_succeeded:
            return fit.fit_error or "fit_failed"
        if (
            candidate.center_shift is not None
            and candidate.center_shift > self.config.max_center_shift + 0.15
        ):
            return "center_shift_too_large"
        if candidate.sigma is None or candidate.sigma < self.config.min_sigma:
            return "sigma_too_small"
        if candidate.sigma is None or candidate.sigma > self.config.max_sigma:
            return "sigma_too_large"
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
        if object_mask is not None and not self._center_inside_mask(candidate, object_mask):
            return "center_outside_object_mask"
        return None

    def _center_inside_mask(
        self,
        candidate: PunctumCandidate,
        object_mask: np.ndarray,
    ) -> bool:
        row = int(round(candidate.final_row))
        col = int(round(candidate.final_col))
        if row < 0 or col < 0 or row >= object_mask.shape[0] or col >= object_mask.shape[1]:
            return False
        if object_mask[row, col]:
            return True

        # Allow sub-pixel shifts to fall on pixels immediately adjacent to tiny mask blobs.
        radius = max(1, int(round(self.config.max_center_shift)))
        min_row = max(0, row - radius)
        max_row = min(object_mask.shape[0], row + radius + 1)
        min_col = max(0, col - radius)
        max_col = min(object_mask.shape[1], col + radius + 1)
        neighborhood = object_mask[min_row:max_row, min_col:max_col]
        return bool(neighborhood.any())

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
        """Prefer higher amplitude, then lower residual."""
        existing_amp = existing.amplitude or 0.0
        challenger_amp = challenger.amplitude or 0.0
        if challenger_amp > existing_amp:
            return True
        if challenger_amp < existing_amp:
            return False
        existing_res = existing.residual_rmse if existing.residual_rmse is not None else float("inf")
        challenger_res = challenger.residual_rmse if challenger.residual_rmse is not None else float("inf")
        return challenger_res < existing_res

    @staticmethod
    def _center_shift(peak: PeakCandidate, fit: GaussianFitResult) -> float | None:
        if not fit.fit_succeeded:
            return None
        return math.hypot(fit.fitted_row - peak.row, fit.fitted_col - peak.col)
