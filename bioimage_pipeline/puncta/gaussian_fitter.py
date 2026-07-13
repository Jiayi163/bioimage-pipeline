"""Elliptical and mixture Gaussian fitting for puncta declumping."""

from __future__ import annotations

import math
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.gmm_multi_start import (
    fit_mixture_from_init_peaks,
    fit_two_component_multi_start,
)
from bioimage_pipeline.puncta.fit_metrics import compute_aic_bic, compute_r_squared, compute_rmse
from bioimage_pipeline.puncta.types import (
    GaussianComponent,
    MixtureFitResult,
    ObjectInfo,
    ObjectPatch,
    PeakCandidate,
)

FWHM_FACTOR = 2.355


def _elliptical_component(
    rows: np.ndarray,
    cols: np.ndarray,
    amplitude: float,
    row_center: float,
    col_center: float,
    sigma_row: float,
    sigma_col: float,
) -> np.ndarray:
    sigma_row = max(sigma_row, 1e-6)
    sigma_col = max(sigma_col, 1e-6)
    return amplitude * np.exp(
        -((rows - row_center) ** 2) / (2.0 * sigma_row**2)
        - ((cols - col_center) ** 2) / (2.0 * sigma_col**2)
    )


def _predict_mixture(
    rows: np.ndarray,
    cols: np.ndarray,
    background: float,
    component_params: list[tuple[float, float, float, float, float]],
) -> np.ndarray:
    predicted = np.full(rows.shape, background, dtype=np.float64)
    for amplitude, row_center, col_center, sigma_row, sigma_col in component_params:
        predicted += _elliptical_component(
            rows,
            cols,
            amplitude,
            row_center,
            col_center,
            sigma_row,
            sigma_col,
        )
    return predicted


@dataclass
class _PatchArrays:
    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray
    weights: np.ndarray


class EllipticalGaussianFitter:
    """Fit one elliptical Gaussian on a background-corrected patch."""

    def __init__(self, config: PunctaDeclumpConfig) -> None:
        self.config = config

    def fit_peak(
        self,
        patch: ObjectPatch,
        peak: PeakCandidate,
        *,
        component_id: int = 1,
        n_components_in_model: int = 1,
    ) -> GaussianComponent:
        patch_row, patch_col = peak.row - patch.row_offset, peak.col - patch.col_offset
        radius = self.config.fit_roi_radius
        center_row = int(round(patch_row))
        center_col = int(round(patch_col))

        min_row = max(0, center_row - radius)
        max_row = min(patch.corrected.shape[0], center_row + radius + 1)
        min_col = max(0, center_col - radius)
        max_col = min(patch.corrected.shape[1], center_col + radius + 1)

        if min_row >= max_row or min_col >= max_col:
            return self._failed_component(peak, component_id, n_components_in_model, "empty_roi")

        roi = patch.corrected[min_row:max_row, min_col:max_col]
        rows, cols = np.mgrid[min_row:max_row, min_col:max_col]
        coords_rows = rows.ravel().astype(np.float64)
        coords_cols = cols.ravel().astype(np.float64)
        observed = roi.ravel().astype(np.float64)

        sigma_guess = max(
            self.config.expected_single_spot_diameter / FWHM_FACTOR,
            self.config.min_sigma,
        )
        amplitude_guess = max(float(peak.intensity - patch.background_level), self.config.min_amplitude)
        initial = np.array(
            [amplitude_guess, patch_row, patch_col, sigma_guess, sigma_guess],
            dtype=np.float64,
        )
        lower = np.array(
            [
                0.0,
                patch_row - self.config.max_center_shift,
                patch_col - self.config.max_center_shift,
                self.config.min_sigma,
                self.config.min_sigma,
            ],
            dtype=np.float64,
        )
        upper = np.array(
            [
                max(amplitude_guess * 3.0, self.config.min_amplitude),
                patch_row + self.config.max_center_shift,
                patch_col + self.config.max_center_shift,
                self.config.max_sigma,
                self.config.max_sigma,
            ],
            dtype=np.float64,
        )

        def residuals(params: np.ndarray) -> np.ndarray:
            amplitude, row_center, col_center, sigma_row, sigma_col = params
            predicted = _elliptical_component(
                coords_rows,
                coords_cols,
                amplitude,
                row_center,
                col_center,
                sigma_row,
                sigma_col,
            )
            return observed - predicted

        try:
            result = least_squares(
                residuals,
                initial,
                bounds=(lower, upper),
                max_nfev=2000,
            )
            amplitude, row_center, col_center, sigma_row, sigma_col = result.x
            predicted = _elliptical_component(
                coords_rows,
                coords_cols,
                amplitude,
                row_center,
                col_center,
                sigma_row,
                sigma_col,
            )
            rmse = compute_rmse(observed, predicted)
            rel = rmse / max(amplitude, 1.0)
            r2 = compute_r_squared(observed, predicted)

            # Full-patch prediction/residual for diagnostics.
            full_rows, full_cols = np.indices(patch.corrected.shape)
            full_pred = _elliptical_component(
                full_rows.astype(np.float64),
                full_cols.astype(np.float64),
                amplitude,
                row_center,
                col_center,
                sigma_row,
                sigma_col,
            )
            full_pred = np.where(patch.object_mask, full_pred, 0.0)
            full_resid = np.zeros_like(patch.corrected)
            full_resid[patch.object_mask] = (
                patch.corrected[patch.object_mask] - full_pred[patch.object_mask]
            )

            return GaussianComponent(
                component_id=component_id,
                initial_row=peak.row,
                initial_col=peak.col,
                fitted_row=row_center + patch.row_offset,
                fitted_col=col_center + patch.col_offset,
                sigma_row=float(sigma_row),
                sigma_col=float(sigma_col),
                amplitude=float(amplitude),
                background=float(patch.background_level),
                residual_rmse=rmse,
                residual_relative=rel,
                r_squared=r2,
                model_score=r2,
                n_components_in_model=n_components_in_model,
                fit_succeeded=True,
                predicted_patch=full_pred,
                residual_patch=full_resid,
            )
        except Exception as exc:
            return self._failed_component(
                peak,
                component_id,
                n_components_in_model,
                str(exc),
            )

    @staticmethod
    def _failed_component(
        peak: PeakCandidate,
        component_id: int,
        n_components_in_model: int,
        error: str,
    ) -> GaussianComponent:
        return GaussianComponent(
            component_id=component_id,
            initial_row=peak.row,
            initial_col=peak.col,
            fitted_row=peak.row,
            fitted_col=peak.col,
            sigma_row=float("nan"),
            sigma_col=float("nan"),
            amplitude=float("nan"),
            background=float("nan"),
            residual_rmse=float("inf"),
            residual_relative=float("inf"),
            r_squared=0.0,
            model_score=0.0,
            n_components_in_model=n_components_in_model,
            fit_succeeded=False,
            fit_error=error,
        )


class GaussianMixtureFitter:
    """Jointly fit multiple elliptical Gaussian components inside one object ROI."""

    def __init__(self, config: PunctaDeclumpConfig) -> None:
        self.config = config
        self.single_fitter = EllipticalGaussianFitter(config)

    def fit_patch(
        self,
        patch: ObjectPatch,
        peaks: list[PeakCandidate],
        *,
        n_components: int,
        obj: ObjectInfo | None = None,
        single_component: GaussianComponent | None = None,
    ) -> MixtureFitResult:
        if (
            n_components == 2
            and self.config.gmm_multi_start_enabled
        ):
            multi = fit_two_component_multi_start(
                self,
                patch,
                peaks,
                obj=obj,
                single_component=single_component,
            )
            return multi.fit

        init_peaks = self._initial_peaks(peaks, n_components, patch)
        return fit_mixture_from_init_peaks(
            self,
            patch,
            init_peaks,
            n_components=n_components,
            initialization_method="detector_based",
            max_nfev=self.config.gmm_multi_start_max_nfev,
        )

    def _extract_weighted_patch(self, patch: ObjectPatch) -> _PatchArrays:
        rows, cols = np.indices(patch.corrected.shape)
        mask = patch.object_mask
        return _PatchArrays(
            rows=rows[mask].astype(np.float64),
            cols=cols[mask].astype(np.float64),
            values=patch.corrected[mask].astype(np.float64),
            weights=np.ones(int(mask.sum()), dtype=np.float64),
        )

    def _initial_peaks(
        self,
        peaks: list[PeakCandidate],
        n_components: int,
        patch: ObjectPatch,
    ) -> list[PeakCandidate]:
        if not peaks:
            center_row = patch.row_offset + patch.corrected.shape[0] / 2.0
            center_col = patch.col_offset + patch.corrected.shape[1] / 2.0
            return [
                PeakCandidate(row=center_row, col=center_col, intensity=float(patch.corrected.max()))
                for _ in range(n_components)
            ]

        sorted_peaks = sorted(peaks, key=lambda p: p.intensity, reverse=True)
        if len(sorted_peaks) >= n_components:
            return sorted_peaks[:n_components]

        init = list(sorted_peaks)
        while len(init) < n_components:
            base = init[-1]
            offset = 0.75 * (len(init) + 1)
            init.append(
                PeakCandidate(
                    row=base.row,
                    col=base.col + offset,
                    intensity=base.intensity * 0.8,
                )
            )
        return init

    def _merge_close_components(
        self,
        components: list[GaussianComponent],
    ) -> tuple[list[GaussianComponent], list[str]]:
        if len(components) <= 1:
            return components, []

        kept: list[GaussianComponent] = []
        notes: list[str] = []
        for component in sorted(components, key=lambda c: c.amplitude, reverse=True):
            too_close = False
            for existing in kept:
                distance = math.hypot(
                    component.fitted_row - existing.fitted_row,
                    component.fitted_col - existing.fitted_col,
                )
                if distance < self.config.gmm_min_component_separation:
                    notes.append(
                        f"merged_comp_amp={component.amplitude:.1f}_too_close_"
                        f"dist={distance:.2f}<{self.config.gmm_min_component_separation}"
                    )
                    too_close = True
                    break
                max_amp = max(existing.amplitude, component.amplitude)
                amp_ratio = (
                    min(existing.amplitude, component.amplitude) / max_amp if max_amp > 0 else 0.0
                )
                if amp_ratio < self.config.gmm_merge_amplitude_ratio:
                    notes.append(
                        f"removed_weak_comp_amp={component.amplitude:.1f}_"
                        f"ratio={amp_ratio:.3f}<{self.config.gmm_merge_amplitude_ratio}"
                    )
                    too_close = True
                    break
            if not too_close:
                kept.append(component)

        for index, component in enumerate(kept, start=1):
            component.component_id = index
        return kept, notes

    def _build_predicted_patch(
        self,
        patch: ObjectPatch,
        component_params: list[tuple[float, float, float, float, float]],
    ) -> np.ndarray:
        rows, cols = np.indices(patch.corrected.shape)
        predicted = _predict_mixture(
            rows.astype(np.float64),
            cols.astype(np.float64),
            0.0,
            component_params,
        )
        return np.where(patch.object_mask, predicted, 0.0)

    def _build_residual_patch(
        self,
        patch: ObjectPatch,
        component_params: list[tuple[float, float, float, float, float]],
    ) -> np.ndarray:
        predicted = self._build_predicted_patch(patch, component_params)
        residual = np.zeros_like(patch.corrected)
        residual[patch.object_mask] = patch.corrected[patch.object_mask] - predicted[patch.object_mask]
        return residual


@dataclass
class ModelComparisonResult:
    """Result of comparing one-Gaussian vs multi-component models."""

    selected: MixtureFitResult | GaussianComponent
    single: GaussianComponent
    best_mixture: MixtureFitResult | None
    selection_reason: str
    rejected_component_reason: str | None = None
    candidate_component_counts: list[int] = field(default_factory=list)


class GaussianModelSelector:
    """Compare 1..M Gaussian mixture models using BIC."""

    def __init__(self, config: PunctaDeclumpConfig) -> None:
        self.config = config
        self.mixture_fitter = GaussianMixtureFitter(config)
        self.single_fitter = EllipticalGaussianFitter(config)

    def select_best_model(
        self,
        patch: ObjectPatch,
        peaks: list[PeakCandidate],
        *,
        force_try_two: bool = False,
        single_component: GaussianComponent | None = None,
    ) -> ModelComparisonResult:
        peak_count = max(1, len(peaks))
        candidate_counts = {1}
        candidate_counts.add(min(peak_count, self.config.gmm_max_components))
        if peak_count > 1 or force_try_two:
            candidate_counts.add(2)
            candidate_counts.add(
                min(
                    max(1, peak_count - self.config.gmm_try_component_delta),
                    self.config.gmm_max_components,
                )
            )
            candidate_counts.add(
                min(peak_count + self.config.gmm_try_component_delta, self.config.gmm_max_components)
            )

        primary_peak = peaks[0]
        single = single_component or self.single_fitter.fit_peak(
            patch,
            primary_peak,
            component_id=1,
            n_components_in_model=1,
        )
        single_bic = self._single_component_bic(patch, single)

        best_mixture: MixtureFitResult | None = None
        for n_components in sorted(c for c in candidate_counts if c >= 1):
            if n_components == 1:
                continue
            fit = self.mixture_fitter.fit_patch(patch, peaks, n_components=n_components)
            if not fit.fit_succeeded:
                continue
            if best_mixture is None or fit.bic < best_mixture.bic:
                best_mixture = fit

        rejected_reason: str | None = None
        if best_mixture is None:
            reason = "no_successful_multi_component_fit"
            return ModelComparisonResult(
                selected=single,
                single=single,
                best_mixture=None,
                selection_reason=reason,
                rejected_component_reason=reason,
                candidate_component_counts=sorted(candidate_counts),
            )

        if best_mixture.n_components <= 1:
            reason = "multi_component_collapsed_to_one_after_merge"
            return ModelComparisonResult(
                selected=single,
                single=single,
                best_mixture=best_mixture,
                selection_reason=reason,
                rejected_component_reason="; ".join(best_mixture.merge_notes) or reason,
                candidate_component_counts=sorted(candidate_counts),
            )

        bic_margin = self.config.gmm_bic_improvement_margin
        if best_mixture.bic + bic_margin < single_bic:
            reason = (
                f"selected_gmm_n={best_mixture.n_components}_bic={best_mixture.bic:.1f}"
                f"_vs_single_bic={single_bic:.1f}"
            )
            return ModelComparisonResult(
                selected=best_mixture,
                single=single,
                best_mixture=best_mixture,
                selection_reason=reason,
                rejected_component_reason=None,
                candidate_component_counts=sorted(candidate_counts),
            )

        # Prefer GMM when residual/R2 clearly better even if BIC is close.
        if (
            single.fit_succeeded
            and best_mixture.r_squared > single.r_squared + 0.05
            and best_mixture.residual_rmse < single.residual_rmse * 0.85
        ):
            reason = (
                f"selected_gmm_by_residual_r2_gain_"
                f"r2={best_mixture.r_squared:.3f}_vs_{single.r_squared:.3f}"
            )
            return ModelComparisonResult(
                selected=best_mixture,
                single=single,
                best_mixture=best_mixture,
                selection_reason=reason,
                rejected_component_reason=None,
                candidate_component_counts=sorted(candidate_counts),
            )

        rejected_reason = (
            f"gmm_rejected_bic_not_better_"
            f"gmm_bic={best_mixture.bic:.1f}_single_bic={single_bic:.1f}_"
            f"gmm_r2={best_mixture.r_squared:.3f}_single_r2={single.r_squared:.3f}"
        )
        if best_mixture.merge_notes:
            rejected_reason += "; " + "; ".join(best_mixture.merge_notes)

        return ModelComparisonResult(
            selected=single,
            single=single,
            best_mixture=best_mixture,
            selection_reason="kept_single_gaussian",
            rejected_component_reason=rejected_reason,
            candidate_component_counts=sorted(candidate_counts),
        )

    def select_balanced_model(
        self,
        patch: ObjectPatch,
        peaks: list[PeakCandidate],
        *,
        single_component: GaussianComponent,
        n_filtered_peaks: int,
        n_raw_peaks: int,
        obj: ObjectInfo | None = None,
    ) -> ModelComparisonResult:
        """Two-stage GMM: try 2 components first; 3 only when warranted."""
        single = single_component
        single_bic = self._single_component_bic(patch, single)
        single_aic = self._single_component_aic(patch, single)
        candidate_counts: list[int] = []

        max_components = self._max_components_for_object(obj)
        fit_two: MixtureFitResult | None = None
        fit_three: MixtureFitResult | None = None

        if max_components >= 2:
            fit_two = self.mixture_fitter.fit_patch(
                patch,
                peaks,
                n_components=2,
                obj=obj,
                single_component=single,
            )
            candidate_counts.append(2)

        try_three = max_components >= 3 and (n_filtered_peaks >= 3 or n_raw_peaks >= 3)
        if not try_three and fit_two is not None and fit_two.fit_succeeded:
            if self._mixture_still_poor(single, fit_two):
                try_three = True
            elif not self._score_improved(
                single_bic,
                single_aic,
                fit_two.bic,
                fit_two.aic,
            ):
                try_three = False

        if try_three and max_components >= 3:
            fit_three = self.mixture_fitter.fit_patch(
                patch,
                peaks,
                n_components=3,
                obj=obj,
                single_component=single,
            )
            candidate_counts.append(3)
            if fit_two is not None and fit_three is not None and fit_three.fit_succeeded:
                if not self._score_improved(fit_two.bic, fit_two.aic, fit_three.bic, fit_three.aic):
                    fit_three = None
                    candidate_counts = [count for count in candidate_counts if count != 3]

        best_mixture = self._pick_best_mixture(fit_two, fit_three)
        return self._compare_single_vs_mixture(
            patch,
            single,
            single_bic,
            best_mixture,
            candidate_counts,
            single_aic=single_aic,
        )

    def _max_components_for_object(self, obj: ObjectInfo | None) -> int:
        if obj is None:
            return self.config.gmm_max_components
        if obj.equivalent_diameter > self.config.large_object_diameter_threshold:
            return self.config.gmm_max_components_large
        return self.config.gmm_max_components

    def _score_improved(
        self,
        baseline_bic: float,
        baseline_aic: float,
        candidate_bic: float,
        candidate_aic: float,
    ) -> bool:
        bic_margin = self.config.gmm_bic_improvement_margin
        aic_margin = self.config.gmm_aic_improvement_margin
        bic_ok = candidate_bic + bic_margin < baseline_bic
        aic_ok = candidate_aic + aic_margin < baseline_aic
        return bic_ok or aic_ok

    def _single_component_aic(self, patch: ObjectPatch, single: GaussianComponent) -> float:
        if not single.fit_succeeded:
            return float("inf")
        n_points = max(int(patch.object_mask.sum()), 1)
        k = 6
        rss = single.residual_rmse**2 * n_points
        return n_points * np.log(max(rss / n_points, 1e-12)) + 2 * k

    def _mixture_still_poor(
        self,
        single: GaussianComponent,
        mixture: MixtureFitResult,
    ) -> bool:
        if not mixture.fit_succeeded or not single.fit_succeeded:
            return False
        rel = mixture.residual_rmse / max(
            max((c.amplitude for c in mixture.components), default=1.0),
            1.0,
        )
        return (
            mixture.r_squared < single.r_squared + 0.03
            or rel > single.residual_relative * 0.9
        )

    def _pick_best_mixture(
        self,
        fit_two: MixtureFitResult | None,
        fit_three: MixtureFitResult | None,
    ) -> MixtureFitResult | None:
        candidates = [
            fit
            for fit in (fit_two, fit_three)
            if fit is not None and fit.fit_succeeded and fit.n_components > 1
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda fit: fit.bic)

    def _mixture_spurious_split(
        self,
        single: GaussianComponent,
        mixture: MixtureFitResult,
    ) -> bool:
        """Reject tight multi-component splits when the single fit is already good."""
        if mixture.n_components < 2 or not single.fit_succeeded:
            return False
        if single.r_squared < self.config.gmm_weak_fit_r_squared:
            return False
        centers = [(component.fitted_row, component.fitted_col) for component in mixture.components]
        min_distance = float("inf")
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                min_distance = min(
                    min_distance,
                    math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1]),
                )
        return min_distance < self.config.min_center_separation

    def _compare_single_vs_mixture(
        self,
        patch: ObjectPatch,
        single: GaussianComponent,
        single_bic: float,
        best_mixture: MixtureFitResult | None,
        candidate_counts: list[int],
        *,
        single_aic: float | None = None,
    ) -> ModelComparisonResult:
        if best_mixture is None:
            reason = "no_successful_multi_component_fit"
            return ModelComparisonResult(
                selected=single,
                single=single,
                best_mixture=None,
                selection_reason=reason,
                rejected_component_reason=reason,
                candidate_component_counts=candidate_counts,
            )

        if best_mixture.n_components <= 1:
            reason = "multi_component_collapsed_to_one_after_merge"
            return ModelComparisonResult(
                selected=single,
                single=single,
                best_mixture=best_mixture,
                selection_reason=reason,
                rejected_component_reason="; ".join(best_mixture.merge_notes) or reason,
                candidate_component_counts=candidate_counts,
            )

        bic_margin = self.config.gmm_bic_improvement_margin
        aic_margin = self.config.gmm_aic_improvement_margin
        single_aic_val = single_aic if single_aic is not None else float("inf")
        bic_improved = best_mixture.bic + bic_margin < single_bic
        aic_improved = best_mixture.aic + aic_margin < single_aic_val
        if bic_improved or aic_improved:
            if self._mixture_spurious_split(single, best_mixture):
                reason = (
                    "kept_single_rejected_spurious_tight_split_"
                    f"r2={single.r_squared:.3f}_min_dist<{self.config.min_center_separation}"
                )
                return ModelComparisonResult(
                    selected=single,
                    single=single,
                    best_mixture=best_mixture,
                    selection_reason=reason,
                    rejected_component_reason=reason,
                    candidate_component_counts=candidate_counts,
                )
            reason = (
                f"selected_gmm_n={best_mixture.n_components}_bic={best_mixture.bic:.1f}"
                f"_vs_single_bic={single_bic:.1f}"
            )
            return ModelComparisonResult(
                selected=best_mixture,
                single=single,
                best_mixture=best_mixture,
                selection_reason=reason,
                rejected_component_reason=None,
                candidate_component_counts=candidate_counts,
            )

        if (
            single.fit_succeeded
            and best_mixture.r_squared > single.r_squared + 0.05
            and best_mixture.residual_rmse < single.residual_rmse * 0.85
        ):
            if self._mixture_spurious_split(single, best_mixture):
                reason = (
                    "kept_single_rejected_spurious_tight_split_"
                    f"r2={single.r_squared:.3f}_min_dist<{self.config.min_center_separation}"
                )
                return ModelComparisonResult(
                    selected=single,
                    single=single,
                    best_mixture=best_mixture,
                    selection_reason=reason,
                    rejected_component_reason=reason,
                    candidate_component_counts=candidate_counts,
                )
            reason = (
                f"selected_gmm_by_residual_r2_gain_"
                f"r2={best_mixture.r_squared:.3f}_vs_{single.r_squared:.3f}"
            )
            return ModelComparisonResult(
                selected=best_mixture,
                single=single,
                best_mixture=best_mixture,
                selection_reason=reason,
                rejected_component_reason=None,
                candidate_component_counts=candidate_counts,
            )

        rejected_reason = (
            f"gmm_rejected_bic_not_better_"
            f"gmm_bic={best_mixture.bic:.1f}_single_bic={single_bic:.1f}_"
            f"gmm_r2={best_mixture.r_squared:.3f}_single_r2={single.r_squared:.3f}"
        )
        if best_mixture.merge_notes:
            rejected_reason += "; " + "; ".join(best_mixture.merge_notes)

        return ModelComparisonResult(
            selected=single,
            single=single,
            best_mixture=best_mixture,
            selection_reason="kept_single_gaussian",
            rejected_component_reason=rejected_reason,
            candidate_component_counts=candidate_counts,
        )

    def _single_component_bic(self, patch: ObjectPatch, component: GaussianComponent) -> float:
        if not component.fit_succeeded:
            return float("inf")
        arrays = self.mixture_fitter._extract_weighted_patch(patch)
        rows = arrays.rows
        cols = arrays.cols
        predicted = _elliptical_component(
            rows,
            cols,
            component.amplitude,
            component.fitted_row - patch.row_offset,
            component.fitted_col - patch.col_offset,
            component.sigma_row,
            component.sigma_col,
        )
        rss = float(np.sum((arrays.values - predicted) ** 2))
        _, bic = compute_aic_bic(rss, arrays.values.size, 5)
        return bic


# Backward-compatible alias used by older tests/scripts.
GaussianFitter2D = EllipticalGaussianFitter
