"""Ground-truth oracle experiments for synthetic GMM diagnostics.

These utilities use known synthetic spot locations for initialization only.
They must never be wired into production pipeline behavior.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy.optimize import OptimizeResult, least_squares

from bioimage_pipeline.puncta.background import build_object_patch
from bioimage_pipeline.puncta.candidate_filter import CandidateFilter
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.connected_objects import ConnectedObjectAnalyzer
from bioimage_pipeline.puncta.fit_metrics import compute_aic_bic, compute_r_squared, compute_rmse
from bioimage_pipeline.puncta.gaussian_fitter import GaussianMixtureFitter
from bioimage_pipeline.puncta.types import (
    GaussianComponent,
    MixtureFitResult,
    ObjectPatch,
    PeakCandidate,
)

FWHM_FACTOR = 2.355


def _min_component_center_separation(components: list[GaussianComponent]) -> float | None:
    if len(components) < 2:
        return None
    min_distance = float("inf")
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            c0, c1 = components[i], components[j]
            min_distance = min(
                min_distance,
                math.hypot(c0.fitted_col - c1.fitted_col, c0.fitted_row - c1.fitted_row),
            )
    return None if not math.isfinite(min_distance) else float(min_distance)


def assess_mixture_validity(
    fit: MixtureFitResult,
    *,
    pre_merge_count: int | None = None,
) -> tuple[bool, str | None]:
    """Validation helper mirroring multi-start validity semantics."""
    merge_collapsed = bool(
        fit.merge_notes
        or (pre_merge_count is not None and fit.n_components < pre_merge_count)
        or fit.n_components < 2
    )
    if not fit.fit_succeeded:
        return False, fit.fit_error or "fit_failed"
    if merge_collapsed:
        return False, "components_merged_or_collapsed"
    if fit.n_components < 2:
        return False, f"insufficient_components_{fit.n_components}"
    return True, None


@dataclass
class OracleAttemptDiagnostics:
    """Rich attempt record for oracle reports (validation-only)."""

    strategy: str
    converged: bool
    post_merge_component_count: int
    merge_collapsed: bool
    merge_notes: list[str]
    fitted_centers: list[tuple[float, float]]
    fitted_amplitudes: list[float]
    fitted_sigma_x: list[float]
    fitted_sigma_y: list[float]
    rss: float | None
    bic: float | None
    aic: float | None = None
    selected: bool = False
    optimizer_runtime_s: float | None = None
    n_optimizer_evaluations: int | None = None
    initial_centers: list[tuple[float, float]] = field(default_factory=list)
    optimizer_success: bool = False
    optimizer_status: int | None = None
    optimizer_message: str | None = None
    optimizer_termination_reason: str | None = None
    cost: float | None = None
    pre_merge_component_count: int | None = None
    validity_passed: bool = False
    rejection_reason: str | None = None
    fit_error: str | None = None
    bounds_hit: dict[str, bool] = field(default_factory=dict)
    exception_caught: str | None = None
    pre_merge_fitted_centers: list[tuple[float, float]] = field(default_factory=list)
    pre_merge_fitted_amplitudes: list[float] = field(default_factory=list)
    pre_merge_fitted_sigma_x: list[float] = field(default_factory=list)
    pre_merge_fitted_sigma_y: list[float] = field(default_factory=list)
    pre_merge_center_separation_px: float | None = None


@dataclass
class OracleFitDetails:
    fit: MixtureFitResult
    pre_merge_components: list[GaussianComponent]
    pre_merge_count: int
    optimizer_success: bool = False
    optimizer_status: int | None = None
    optimizer_message: str | None = None
    optimizer_termination_reason: str | None = None
    cost: float | None = None
    bounds_hit: dict[str, bool] = field(default_factory=dict)
    exception_caught: str | None = None


def _optimizer_termination_reason(status: int | None) -> str | None:
    if status is None:
        return None
    mapping = {
        1: "ftol",
        2: "xtol",
        3: "gtol",
        4: "max_nfev",
        5: "max_nfev",
        -1: "interrupted",
        -2: "error",
    }
    return mapping.get(status, f"status_{status}")


def _bounds_hit_flags(
    result_x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    tol: float = 1e-6,
) -> dict[str, bool]:
    return {
        "lower": bool(np.any(result_x <= lower + tol)),
        "upper": bool(np.any(result_x >= upper - tol)),
    }


def _oracle_attempt_diagnostics(
    strategy: str,
    details: OracleFitDetails,
    *,
    initial_centers: list[tuple[float, float]],
    selected: bool = False,
) -> OracleAttemptDiagnostics:
    fit = details.fit
    pre_merge = details.pre_merge_components
    centers = [(component.fitted_col, component.fitted_row) for component in fit.components]
    pre_merge_centers = [(component.fitted_col, component.fitted_row) for component in pre_merge]
    merge_collapsed = bool(
        fit.merge_notes
        or fit.n_components < details.pre_merge_count
        or fit.n_components < 2
    )
    rss = fit.residual_rmse**2 if fit.fit_succeeded and math.isfinite(fit.residual_rmse) else None
    validity_passed, rejection_reason = assess_mixture_validity(
        fit,
        pre_merge_count=details.pre_merge_count,
    )
    if validity_passed:
        rejection_reason = None
    return OracleAttemptDiagnostics(
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
        aic=fit.aic if fit.fit_succeeded else None,
        selected=selected,
        optimizer_runtime_s=fit.optimizer_runtime_s,
        n_optimizer_evaluations=fit.optimizer_nfev,
        initial_centers=initial_centers,
        optimizer_success=details.optimizer_success,
        optimizer_status=details.optimizer_status,
        optimizer_message=details.optimizer_message,
        optimizer_termination_reason=details.optimizer_termination_reason,
        cost=details.cost,
        pre_merge_component_count=details.pre_merge_count,
        validity_passed=validity_passed,
        rejection_reason=rejection_reason,
        fit_error=fit.fit_error,
        bounds_hit=dict(details.bounds_hit),
        exception_caught=details.exception_caught,
        pre_merge_fitted_centers=pre_merge_centers,
        pre_merge_fitted_amplitudes=[component.amplitude for component in pre_merge],
        pre_merge_fitted_sigma_x=[component.sigma_col for component in pre_merge],
        pre_merge_fitted_sigma_y=[component.sigma_row for component in pre_merge],
        pre_merge_center_separation_px=_min_component_center_separation(pre_merge),
    )


def fit_oracle_mixture_from_init_peaks(
    mixture_fitter: GaussianMixtureFitter,
    patch: ObjectPatch,
    init_peaks: list[PeakCandidate],
    *,
    n_components: int,
    initialization_method: str = "oracle_ground_truth",
    max_nfev: int | None = None,
) -> OracleFitDetails:
    """Validation-only fit that captures pre-merge components and optimizer metadata."""
    from bioimage_pipeline.puncta.gaussian_fitter import _predict_mixture

    config = mixture_fitter.config
    arrays = mixture_fitter._extract_weighted_patch(patch)
    empty_fit = MixtureFitResult(
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
    if arrays.values.size < 5:
        return OracleFitDetails(
            fit=empty_fit,
            pre_merge_components=[],
            pre_merge_count=0,
            exception_caught="insufficient_pixels",
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

    lower_arr = np.array(lower, dtype=np.float64)
    upper_arr = np.array(upper, dtype=np.float64)

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

    optimizer_success = False
    optimizer_status: int | None = None
    optimizer_message: str | None = None
    cost: float | None = None
    bounds_hit: dict[str, bool] = {}
    exception_caught: str | None = None
    optimizer_runtime_s: float | None = None
    optimizer_nfev: int | None = None

    try:
        optimizer_start = time.perf_counter()
        result: OptimizeResult = least_squares(
            residuals,
            np.array(params, dtype=np.float64),
            bounds=(lower_arr, upper_arr),
            max_nfev=max_nfev or config.gmm_multi_start_max_nfev,
        )
        optimizer_runtime_s = time.perf_counter() - optimizer_start
        optimizer_nfev = int(getattr(result, "nfev", 0) or 0)
        optimizer_success = bool(getattr(result, "success", False))
        optimizer_status = int(result.status) if hasattr(result, "status") else None
        optimizer_message = str(result.message) if getattr(result, "message", None) is not None else None
        cost = float(getattr(result, "cost", 0.0) or 0.0)
        bounds_hit = _bounds_hit_flags(result.x, lower_arr, upper_arr)

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

        pre_merge_components: list[GaussianComponent] = []
        for index, peak in enumerate(init_peaks[:n_components], start=1):
            amplitude, row_center, col_center, sigma_row, sigma_col = component_params[index - 1]
            pre_merge_components.append(
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

        pre_merge_count = len(pre_merge_components)
        components, merge_notes = mixture_fitter._merge_close_components(list(pre_merge_components))
        fit_error = None
        if len(components) < pre_merge_count:
            fit_error = f"post_merge_collapsed_{pre_merge_count}_to_{len(components)}"

        fit = MixtureFitResult(
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
            optimizer_runtime_s=optimizer_runtime_s,
            optimizer_nfev=optimizer_nfev,
        )
        return OracleFitDetails(
            fit=fit,
            pre_merge_components=pre_merge_components,
            pre_merge_count=pre_merge_count,
            optimizer_success=optimizer_success,
            optimizer_status=optimizer_status,
            optimizer_message=optimizer_message,
            optimizer_termination_reason=_optimizer_termination_reason(optimizer_status),
            cost=cost,
            bounds_hit=bounds_hit,
        )
    except Exception as exc:
        failed = MixtureFitResult(
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
            optimizer_runtime_s=optimizer_runtime_s,
            optimizer_nfev=optimizer_nfev,
        )
        return OracleFitDetails(
            fit=failed,
            pre_merge_components=[],
            pre_merge_count=0,
            optimizer_success=False,
            optimizer_status=optimizer_status,
            optimizer_message=optimizer_message,
            optimizer_termination_reason=_optimizer_termination_reason(optimizer_status),
            cost=cost,
            bounds_hit=bounds_hit,
            exception_caught=f"{type(exc).__name__}: {exc}",
        )


@dataclass
class GmmOracleExperimentReport:
    """Report from a single ground-truth-initialized 2-component fit."""

    case_name: str
    object_id: int
    true_centers: list[tuple[float, float]]
    initial_centers: list[tuple[float, float]]
    optimizer_success: bool
    optimizer_status: int | None
    optimizer_message: str | None
    optimizer_termination_reason: str | None
    n_optimizer_evaluations: int | None
    optimizer_runtime_s: float | None
    bounds_hit: dict[str, bool] = field(default_factory=dict)
    exception_caught: str | None = None
    pre_merge_fitted_centers: list[tuple[float, float]] = field(default_factory=list)
    post_merge_fitted_centers: list[tuple[float, float]] = field(default_factory=list)
    pre_merge_center_separation_px: float | None = None
    post_merge_center_separation_px: float | None = None
    fitted_amplitudes_pre_merge: list[float] = field(default_factory=list)
    fitted_amplitudes_post_merge: list[float] = field(default_factory=list)
    fitted_sigma_x_pre_merge: list[float] = field(default_factory=list)
    fitted_sigma_y_pre_merge: list[float] = field(default_factory=list)
    fitted_sigma_x_post_merge: list[float] = field(default_factory=list)
    fitted_sigma_y_post_merge: list[float] = field(default_factory=list)
    rss: float | None = None
    bic: float | None = None
    aic: float | None = None
    cost: float | None = None
    merge_notes: list[str] = field(default_factory=list)
    pre_merge_component_count: int | None = None
    post_merge_component_count: int | None = None
    pre_merge_validity_passed: bool = False
    post_merge_validity_passed: bool = False
    post_merge_validity_rejection_reason: str | None = None
    candidate_filter_would_accept_post_merge: list[bool] = field(default_factory=list)
    candidate_filter_rejection_reasons_post_merge: list[str | None] = field(default_factory=list)
    attempt_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_ground_truth_spots(path: Path) -> list[dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["spots"])


def _spot_centers_col_row(spots: list[dict[str, float]]) -> list[tuple[float, float]]:
    return [(float(spot["x"]), float(spot["y"])) for spot in spots]


def _oracle_init_peaks(
    spots: list[dict[str, float]],
    *,
    background_level: float,
    image: np.ndarray,
) -> list[PeakCandidate]:
    peaks: list[PeakCandidate] = []
    image_arr = np.asarray(image, dtype=np.float64)
    for spot in spots:
        row = int(round(float(spot["y"])))
        col = int(round(float(spot["x"])))
        row = min(max(row, 0), image_arr.shape[0] - 1)
        col = min(max(col, 0), image_arr.shape[1] - 1)
        intensity = float(image_arr[row, col])
        peaks.append(
            PeakCandidate(
                row=float(spot["y"]),
                col=float(spot["x"]),
                intensity=max(intensity, float(spot.get("amplitude", 0.0)) + background_level),
            )
        )
    return peaks


def _clone_mixture_with_components(
    fit: MixtureFitResult,
    components: list[GaussianComponent],
) -> MixtureFitResult:
    return MixtureFitResult(
        components=list(components),
        n_components=len(components),
        background=fit.background,
        residual_rmse=fit.residual_rmse,
        r_squared=fit.r_squared,
        aic=fit.aic,
        bic=fit.bic,
        model_score=fit.model_score,
        fit_succeeded=fit.fit_succeeded,
        fit_error=fit.fit_error,
        predicted_patch=fit.predicted_patch,
        residual_patch=fit.residual_patch,
        merge_notes=[],
        winning_init_strategy=fit.winning_init_strategy,
    )


def run_ground_truth_oracle_experiment(
    *,
    data_root: Path,
    case_name: str,
    config: PunctaDeclumpConfig,
    object_index: int = 0,
) -> GmmOracleExperimentReport:
    """Fit a 2-component mixture initialized at synthetic ground-truth centers."""
    image = tifffile.imread(data_root / "images" / case_name / "synthetic_noisy.tif")
    mask = tifffile.imread(data_root / "masks" / case_name / "synthetic_mask.tif") > 0
    gt_path = data_root / "ground_truth" / case_name / "synthetic_ground_truth.json"
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth not found for case {case_name!r}: {gt_path}")

    spots = _load_ground_truth_spots(gt_path)
    true_centers = _spot_centers_col_row(spots)

    _, objects = ConnectedObjectAnalyzer().analyze(mask, image)
    if not objects:
        raise ValueError(f"No mask objects found for case {case_name!r}")
    if object_index < 0 or object_index >= len(objects):
        raise IndexError(
            f"object_index {object_index} out of range for {len(objects)} object(s) in {case_name!r}"
        )
    obj = objects[object_index]
    patch = build_object_patch(image, mask, obj, config)
    init_peaks = _oracle_init_peaks(spots, background_level=patch.background_level, image=image)
    initial_centers = [(peak.col, peak.row) for peak in init_peaks]

    mixture_fitter = GaussianMixtureFitter(config)
    details = fit_oracle_mixture_from_init_peaks(
        mixture_fitter,
        patch,
        init_peaks,
        n_components=2,
        initialization_method="oracle_ground_truth",
        max_nfev=config.gmm_multi_start_max_nfev,
    )
    fit = details.fit
    attempt = _oracle_attempt_diagnostics(
        "oracle_ground_truth",
        details,
        initial_centers=initial_centers,
    )

    pre_merge_fit = _clone_mixture_with_components(fit, details.pre_merge_components)
    pre_merge_validity_passed, _ = assess_mixture_validity(
        pre_merge_fit,
        pre_merge_count=details.pre_merge_count,
    )

    candidate_filter = CandidateFilter(config)
    all_candidates = candidate_filter.evaluate_mixture_components(
        obj,
        init_peaks,
        fit,
        candidate_id_start=1,
        object_mask=patch.object_mask,
        patch=patch,
    )
    candidate_filter_would_accept = [candidate.accepted for candidate in all_candidates]
    candidate_filter_rejection_reasons = [candidate.rejection_reason for candidate in all_candidates]

    rss = fit.residual_rmse**2 if fit.fit_succeeded and math.isfinite(fit.residual_rmse) else None

    return GmmOracleExperimentReport(
        case_name=case_name,
        object_id=obj.label,
        true_centers=true_centers,
        initial_centers=initial_centers,
        optimizer_success=details.optimizer_success,
        optimizer_status=details.optimizer_status,
        optimizer_message=details.optimizer_message,
        optimizer_termination_reason=details.optimizer_termination_reason,
        n_optimizer_evaluations=fit.optimizer_nfev,
        optimizer_runtime_s=fit.optimizer_runtime_s,
        bounds_hit=dict(details.bounds_hit),
        exception_caught=details.exception_caught,
        pre_merge_fitted_centers=list(attempt.pre_merge_fitted_centers),
        post_merge_fitted_centers=list(attempt.fitted_centers),
        pre_merge_center_separation_px=attempt.pre_merge_center_separation_px,
        post_merge_center_separation_px=_min_component_center_separation(fit.components),
        fitted_amplitudes_pre_merge=list(attempt.pre_merge_fitted_amplitudes),
        fitted_amplitudes_post_merge=list(attempt.fitted_amplitudes),
        fitted_sigma_x_pre_merge=list(attempt.pre_merge_fitted_sigma_x),
        fitted_sigma_y_pre_merge=list(attempt.pre_merge_fitted_sigma_y),
        fitted_sigma_x_post_merge=list(attempt.fitted_sigma_x),
        fitted_sigma_y_post_merge=list(attempt.fitted_sigma_y),
        rss=rss,
        bic=fit.bic if fit.fit_succeeded else None,
        aic=fit.aic if fit.fit_succeeded else None,
        cost=details.cost,
        merge_notes=list(fit.merge_notes),
        pre_merge_component_count=details.pre_merge_count,
        post_merge_component_count=fit.n_components,
        pre_merge_validity_passed=pre_merge_validity_passed,
        post_merge_validity_passed=attempt.validity_passed,
        post_merge_validity_rejection_reason=attempt.rejection_reason,
        candidate_filter_would_accept_post_merge=candidate_filter_would_accept,
        candidate_filter_rejection_reasons_post_merge=candidate_filter_rejection_reasons,
        attempt_diagnostics=asdict(attempt),
    )


def write_oracle_report(report: GmmOracleExperimentReport, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return output_path


def format_oracle_report(report: GmmOracleExperimentReport) -> str:
    lines = [
        f"=== GMM oracle experiment: {report.case_name} ===",
        f"object_id: {report.object_id}",
        f"true centers (col,row): {report.true_centers}",
        f"initial centers (col,row): {report.initial_centers}",
        "",
        "Optimizer:",
        f"  success: {report.optimizer_success}",
        f"  status: {report.optimizer_status}",
        f"  message: {report.optimizer_message}",
        f"  termination: {report.optimizer_termination_reason}",
        f"  nfev: {report.n_optimizer_evaluations}",
        f"  runtime_s: {report.optimizer_runtime_s}",
        f"  bounds_hit: {report.bounds_hit}",
        f"  exception: {report.exception_caught}",
        "",
        "Pre-merge fit:",
        f"  centers (col,row): {report.pre_merge_fitted_centers}",
        f"  center separation px: {report.pre_merge_center_separation_px}",
        f"  amplitudes: {report.fitted_amplitudes_pre_merge}",
        f"  sigma_x: {report.fitted_sigma_x_pre_merge}",
        f"  sigma_y: {report.fitted_sigma_y_pre_merge}",
        f"  component count: {report.pre_merge_component_count}",
        f"  validity_passed: {report.pre_merge_validity_passed}",
        "",
        "Post-merge fit:",
        f"  centers (col,row): {report.post_merge_fitted_centers}",
        f"  center separation px: {report.post_merge_center_separation_px}",
        f"  amplitudes: {report.fitted_amplitudes_post_merge}",
        f"  sigma_x: {report.fitted_sigma_x_post_merge}",
        f"  sigma_y: {report.fitted_sigma_y_post_merge}",
        f"  component count: {report.post_merge_component_count}",
        f"  merge_notes: {report.merge_notes}",
        f"  validity_passed: {report.post_merge_validity_passed}",
        f"  validity_rejection_reason: {report.post_merge_validity_rejection_reason}",
        "",
        "Scores:",
        f"  RSS: {report.rss}",
        f"  BIC: {report.bic}",
        f"  AIC: {report.aic}",
        f"  cost: {report.cost}",
        "",
        "CandidateFilter (post-merge):",
        f"  would_accept: {report.candidate_filter_would_accept_post_merge}",
        f"  rejection_reasons: {report.candidate_filter_rejection_reasons_post_merge}",
    ]
    return "\n".join(lines)
