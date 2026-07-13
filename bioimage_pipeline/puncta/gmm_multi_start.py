"""Multi-start initialization for two-component Gaussian mixture fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.fit_metrics import compute_aic_bic, compute_r_squared, compute_rmse
from bioimage_pipeline.puncta.types import (
    GaussianComponent,
    GmmInitAttemptDiagnostics,
    MixtureFitResult,
    ObjectInfo,
    ObjectPatch,
    PeakCandidate,
)

if TYPE_CHECKING:
    from bioimage_pipeline.puncta.gaussian_fitter import GaussianMixtureFitter

FWHM_FACTOR = 2.355


def _detector_based_init_peaks(
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


@dataclass(frozen=True)
class MultiStartFitResult:
    """Best converged two-component mixture from multi-start search."""

    fit: MixtureFitResult
    winning_strategy: str
    n_starts_attempted: int
    n_starts_converged: int


def generate_two_component_init_sets(
    peaks: list[PeakCandidate],
    patch: ObjectPatch,
    obj: ObjectInfo | None,
    *,
    config: PunctaDeclumpConfig,
    single_component: GaussianComponent | None = None,
) -> dict[str, list[PeakCandidate]]:
    """Return named initialization peak sets for 2-component multi-start."""
    strategies: dict[str, list[PeakCandidate]] = {
        "detector_based": _detector_based_init_peaks(peaks, 2, patch),
    }

    if not peaks:
        return strategies

    base = peaks[0]
    center_row = base.row
    center_col = base.col
    intensity = base.intensity

    separations = config.gmm_multi_start_separations
    for separation in separations:
        half = separation / 2.0
        tag = f"{separation:g}"
        strategies[f"symmetric_x_sep{tag}"] = [
            PeakCandidate(row=center_row, col=center_col - half, intensity=intensity),
            PeakCandidate(row=center_row, col=center_col + half, intensity=intensity),
        ]
        strategies[f"symmetric_y_sep{tag}"] = [
            PeakCandidate(row=center_row - half, col=center_col, intensity=intensity),
            PeakCandidate(row=center_row + half, col=center_col, intensity=intensity),
        ]
        strategies[f"offset_x_sep{tag}"] = [
            PeakCandidate(row=center_row, col=center_col, intensity=intensity),
            PeakCandidate(
                row=center_row,
                col=center_col + separation,
                intensity=intensity * 0.85,
            ),
        ]
        strategies[f"offset_y_sep{tag}"] = [
            PeakCandidate(row=center_row, col=center_col, intensity=intensity),
            PeakCandidate(
                row=center_row + separation,
                col=center_col,
                intensity=intensity * 0.85,
            ),
        ]

    if obj is not None and obj.major_axis_length > 0:
        axis_len = max(obj.major_axis_length, 1.0)
        row_delta = (obj.major_axis_length / axis_len) * 2.0
        col_delta = (obj.minor_axis_length / axis_len) * 2.0
        strategies["major_axis"] = [
            PeakCandidate(
                row=center_row - row_delta,
                col=center_col - col_delta,
                intensity=intensity,
            ),
            PeakCandidate(
                row=center_row + row_delta,
                col=center_col + col_delta,
                intensity=intensity,
            ),
        ]

    if single_component is not None and single_component.residual_patch is not None:
        residual = np.asarray(single_component.residual_patch)
        if residual.any():
            idx = int(np.argmax(residual))
            rr, cc = np.unravel_index(idx, residual.shape)
            strategies["residual_peak"] = [
                PeakCandidate(row=base.row, col=base.col, intensity=intensity),
                PeakCandidate(
                    row=float(rr + patch.row_offset),
                    col=float(cc + patch.col_offset),
                    intensity=float(residual[rr, cc] + patch.background_level),
                ),
            ]

    return strategies


def _attempt_diagnostics(
    strategy: str,
    fit: MixtureFitResult,
    *,
    pre_merge_count: int | None = None,
    selected: bool = False,
) -> GmmInitAttemptDiagnostics:
    centers = [(component.fitted_col, component.fitted_row) for component in fit.components]
    merge_collapsed = bool(
        fit.merge_notes
        or (pre_merge_count is not None and fit.n_components < pre_merge_count)
        or fit.n_components < 2
    )
    rss = fit.residual_rmse**2 if fit.fit_succeeded and math.isfinite(fit.residual_rmse) else None
    return GmmInitAttemptDiagnostics(
        strategy=strategy,
        converged=fit.fit_succeeded and fit.n_components >= 2,
        post_merge_component_count=fit.n_components,
        merge_collapsed=merge_collapsed,
        merge_notes=list(fit.merge_notes),
        fitted_centers=centers,
        fitted_amplitudes=[component.amplitude for component in fit.components],
        fitted_sigma_x=[component.sigma_col for component in fit.components],
        fitted_sigma_y=[component.sigma_row for component in fit.components],
        rss=rss,
        bic=fit.bic if fit.fit_succeeded else None,
        selected=selected,
    )


def fit_mixture_from_init_peaks(
    mixture_fitter: GaussianMixtureFitter,
    patch: ObjectPatch,
    init_peaks: list[PeakCandidate],
    *,
    n_components: int,
    initialization_method: str = "explicit",
    max_nfev: int | None = None,
) -> MixtureFitResult:
    """Fit a mixture from explicit initialization peaks."""
    from bioimage_pipeline.puncta.gaussian_fitter import _predict_mixture

    config = mixture_fitter.config
    arrays = mixture_fitter._extract_weighted_patch(patch)
    if arrays.values.size < 5:
        return MixtureFitResult(
            components=[],
            n_components=n_components,
            background=patch.background_level,
            residual_rmse=float("inf"),
            r_squared=0.0,
            aic=float("inf"),
            bic=float("inf"),
            model_score=float("inf"),
            fit_succeeded=False,
            fit_error="insufficient_pixels",
            winning_init_strategy=initialization_method,
        )

    sigma_guess = max(config.expected_single_spot_diameter / FWHM_FACTOR, config.min_sigma)
    params: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for peak in init_peaks[:n_components]:
        patch_row = peak.row - patch.row_offset
        patch_col = peak.col - patch.col_offset
        amp = max(float(peak.intensity - patch.background_level), config.min_amplitude)
        params.extend([amp, patch_row, patch_col, sigma_guess, sigma_guess])
        lower.extend(
            [
                0.0,
                patch_row - config.max_center_shift,
                patch_col - config.max_center_shift,
                config.min_sigma,
                config.min_sigma,
            ]
        )
        upper.extend(
            [
                amp * 3.0,
                patch_row + config.max_center_shift,
                patch_col + config.max_center_shift,
                config.max_sigma,
                config.max_sigma,
            ]
        )

    def residuals(param_vec: np.ndarray) -> np.ndarray:
        component_params = [
            (
                param_vec[i * 5 + 0],
                param_vec[i * 5 + 1],
                param_vec[i * 5 + 2],
                param_vec[i * 5 + 3],
                param_vec[i * 5 + 4],
            )
            for i in range(n_components)
        ]
        predicted = _predict_mixture(arrays.rows, arrays.cols, 0.0, component_params)
        return (arrays.values - predicted) * arrays.weights

    try:
        result: OptimizeResult = least_squares(
            residuals,
            np.array(params, dtype=np.float64),
            bounds=(np.array(lower), np.array(upper)),
            max_nfev=max_nfev or config.gmm_multi_start_max_nfev,
        )
        component_params = [
            (
                float(result.x[i * 5 + 0]),
                float(result.x[i * 5 + 1]),
                float(result.x[i * 5 + 2]),
                float(result.x[i * 5 + 3]),
                float(result.x[i * 5 + 4]),
            )
            for i in range(n_components)
        ]
        predicted = _predict_mixture(arrays.rows, arrays.cols, 0.0, component_params)
        rmse = compute_rmse(arrays.values, predicted)
        r2 = compute_r_squared(arrays.values, predicted)
        rss = float(np.sum((arrays.values - predicted) ** 2))
        aic, bic = compute_aic_bic(rss, arrays.values.size, 5 * n_components)

        components: list[GaussianComponent] = []
        for index, peak in enumerate(init_peaks[:n_components], start=1):
            amplitude, row_center, col_center, sigma_row, sigma_col = component_params[index - 1]
            components.append(
                GaussianComponent(
                    component_id=index,
                    initial_row=peak.row,
                    initial_col=peak.col,
                    fitted_row=row_center + patch.row_offset,
                    fitted_col=col_center + patch.col_offset,
                    sigma_row=sigma_row,
                    sigma_col=sigma_col,
                    amplitude=amplitude,
                    background=patch.background_level,
                    residual_rmse=rmse,
                    residual_relative=rmse / max(amplitude, 1.0),
                    r_squared=r2,
                    model_score=bic,
                    n_components_in_model=n_components,
                    fit_succeeded=True,
                )
            )

        pre_merge_count = len(components)
        components, merge_notes = mixture_fitter._merge_close_components(components)
        fit_error = None
        if len(components) < pre_merge_count:
            fit_error = f"post_merge_collapsed_{pre_merge_count}_to_{len(components)}"

        return MixtureFitResult(
            components=components,
            n_components=len(components),
            background=patch.background_level,
            residual_rmse=rmse,
            r_squared=r2,
            aic=aic,
            bic=bic,
            model_score=bic,
            fit_succeeded=True,
            predicted_patch=mixture_fitter._build_predicted_patch(patch, component_params),
            residual_patch=mixture_fitter._build_residual_patch(patch, component_params),
            merge_notes=merge_notes,
            fit_error=fit_error,
            winning_init_strategy=initialization_method,
            multi_start_attempts=1,
        )
    except Exception as exc:
        return MixtureFitResult(
            components=[],
            n_components=n_components,
            background=patch.background_level,
            residual_rmse=float("inf"),
            r_squared=0.0,
            aic=float("inf"),
            bic=float("inf"),
            model_score=float("inf"),
            fit_succeeded=False,
            fit_error=str(exc),
            winning_init_strategy=initialization_method,
        )


def fit_two_component_multi_start(
    mixture_fitter: GaussianMixtureFitter,
    patch: ObjectPatch,
    peaks: list[PeakCandidate],
    *,
    obj: ObjectInfo | None,
    single_component: GaussianComponent | None,
) -> MultiStartFitResult:
    """Run bounded multi-start search and return the best BIC mixture fit."""
    config = mixture_fitter.config
    init_sets = generate_two_component_init_sets(
        peaks,
        patch,
        obj,
        config=config,
        single_component=single_component,
    )

    ordered_names = sorted(init_sets.keys())
    if config.gmm_max_multi_starts > 0:
        ordered_names = ordered_names[: config.gmm_max_multi_starts]

    best_fit: MixtureFitResult | None = None
    winning_strategy = "none"
    n_converged = 0
    attempt_records: list[GmmInitAttemptDiagnostics] = []

    for name in ordered_names:
        fit = fit_mixture_from_init_peaks(
            mixture_fitter,
            patch,
            init_sets[name],
            n_components=2,
            initialization_method=name,
            max_nfev=config.gmm_multi_start_max_nfev,
        )
        attempt_records.append(_attempt_diagnostics(name, fit))
        if not fit.fit_succeeded or fit.n_components < 2:
            continue
        n_converged += 1
        if best_fit is None or fit.bic < best_fit.bic:
            best_fit = fit
            winning_strategy = name

    if best_fit is None:
        fallback = fit_mixture_from_init_peaks(
            mixture_fitter,
            patch,
            init_sets.get("detector_based", _detector_based_init_peaks(peaks, 2, patch)),
            n_components=2,
            initialization_method="detector_based",
            max_nfev=config.gmm_multi_start_max_nfev,
        )
        if not any(record.strategy == "detector_based" for record in attempt_records):
            attempt_records.append(_attempt_diagnostics("detector_based", fallback))
        fallback.init_attempts = attempt_records
        return MultiStartFitResult(
            fit=fallback,
            winning_strategy=fallback.winning_init_strategy or "detector_based",
            n_starts_attempted=len(ordered_names),
            n_starts_converged=0,
        )

    for record in attempt_records:
        record.selected = record.strategy == winning_strategy
    best_fit.winning_init_strategy = winning_strategy
    best_fit.multi_start_attempts = len(ordered_names)
    best_fit.multi_start_converged = n_converged
    best_fit.init_attempts = attempt_records
    return MultiStartFitResult(
        fit=best_fit,
        winning_strategy=winning_strategy,
        n_starts_attempted=len(ordered_names),
        n_starts_converged=n_converged,
    )
