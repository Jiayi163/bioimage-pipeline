"""Instrumented GMM fitting for synthetic validation and ablation studies.

This module is intentionally separate from the production puncta pipeline.
It reuses production types and math but adds detailed diagnostics, multi-start
initialization, and filter ablation modes without changing runtime behavior of
``run_puncta_declump``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import tifffile
from scipy.optimize import OptimizeResult, least_squares

from bioimage_pipeline.puncta.background import build_object_patch
from bioimage_pipeline.puncta.candidate_filter import CandidateFilter
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.connected_objects import ConnectedObjectAnalyzer
from bioimage_pipeline.puncta.fit_metrics import compute_aic_bic, compute_r_squared, compute_rmse
from bioimage_pipeline.puncta.gmm_multi_start import generate_two_component_init_sets
from bioimage_pipeline.puncta.gaussian_fitter import (
    EllipticalGaussianFitter,
    GaussianMixtureFitter,
    GaussianModelSelector,
    _predict_mixture,
)
from bioimage_pipeline.puncta.peak_assignment import assign_peaks_to_objects
from bioimage_pipeline.puncta.types import (
    GaussianComponent,
    MixtureFitResult,
    ObjectInfo,
    ObjectPatch,
    PeakCandidate,
)

FilterMode = Literal["none", "duplicate_only", "full"]


class InitStrategy(str, Enum):
    DETECTOR_BASED = "detector_based"
    SYMMETRIC_X = "symmetric_x"
    MAJOR_AXIS = "major_axis"
    OFFSET_X = "offset_x"
    OFFSET_Y = "offset_y"
    RESIDUAL_PEAK = "residual_peak"
    MULTI_START_BEST = "multi_start_best"


@dataclass
class OptimizerDiagnostics:
    converged: bool
    status: int
    message: str
    n_iterations: int
    bounds_hit: list[str]
    initial_params: list[float]
    final_params: list[float]


@dataclass
class ComponentRecord:
    component_id: int
    initial_row: float
    initial_col: float
    fitted_row: float
    fitted_col: float
    amplitude: float
    sigma_x: float
    sigma_y: float
    background: float
    acceptance_filter: str | None = None
    accepted: bool | None = None


@dataclass
class ModelAttemptRecord:
    n_components: int
    attempted: bool
    initialization_method: str
    initialization_details: str
    initial_centers: list[tuple[float, float]]
    converged: bool
    optimizer_status: int
    optimizer_message: str
    n_iterations: int
    bounds_hit: list[str]
    fitted_centers: list[tuple[float, float]]
    fitted_amplitudes: list[float]
    fitted_sigma_x: list[float]
    fitted_sigma_y: list[float]
    background: float
    rss: float | None
    r_squared: float | None
    aic: float | None
    bic: float | None
    pairwise_component_distances_px: list[float]
    post_merge_component_count: int
    merge_notes: list[str]
    model_level_rejection_reason: str | None = None
    component_records: list[ComponentRecord] = field(default_factory=list)
    component_rejection_reasons: list[str] = field(default_factory=list)
    raw_fit_succeeded: bool = False
    selected_by_model_selection: bool = False


@dataclass
class ModelSelectionDiagnostics:
    peak_count_used_for_init: int
    n_filtered_peaks: int
    n_raw_peaks: int
    candidate_component_counts_legacy_select_best: list[int]
    balanced_model_attempted_n2: bool
    balanced_model_attempted_n3: bool
    balanced_model_n3_gate_reason: str
    single_bic: float | None
    single_aic: float | None
    best_mixture_n_components: int | None
    best_mixture_bic: float | None
    best_mixture_aic: float | None
    bic_delta_2_vs_1: float | None
    aic_delta_2_vs_1: float | None
    selection_reason: str | None
    rejected_component_reason: str | None
    exact_second_component_rejection: str | None


@dataclass
class AblationResult:
    mode: str
    init_strategy: str
    filter_mode: FilterMode
    predicted_accepted_count: int
    true_positives: int
    model_selection_reason: str | None
    second_component_rejection: str | None
    best_two_component_bic: float | None
    notes: str = ""


@dataclass
class GmmProbeReport:
    run_name: str
    ground_truth_case: str
    object_id: int | None
    n_detected_peaks: int
    detected_peak_coords: list[tuple[float, float]]
    model_attempts: list[ModelAttemptRecord]
    model_selection: ModelSelectionDiagnostics
    ablation_results: list[AblationResult]
    production_path_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return None
    return value


def _pairwise_distances(centers: list[tuple[float, float]]) -> list[float]:
    distances: list[float] = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            distances.append(
                float(math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1]))
            )
    return distances


def _bounds_hit_flags(
    final: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    n_components: int,
) -> list[str]:
    hits: list[str] = []
    labels = ["amp", "row", "col", "sigma_row", "sigma_col"]
    for comp in range(n_components):
        for param_index, label in enumerate(labels):
            idx = comp * 5 + param_index
            if abs(final[idx] - lower[idx]) < 1e-4:
                hits.append(f"comp{comp + 1}_{label}_at_lower")
            if abs(final[idx] - upper[idx]) < 1e-4:
                hits.append(f"comp{comp + 1}_{label}_at_upper")
    return hits


def generate_init_peak_sets(
    peaks: list[PeakCandidate],
    patch: ObjectPatch,
    obj: ObjectInfo | None,
    *,
    n_components: int,
    single_component: GaussianComponent | None = None,
    config: PunctaDeclumpConfig | None = None,
) -> dict[str, list[PeakCandidate]]:
    """Return named initialization peak sets for multi-start probing."""
    cfg = config or PunctaDeclumpConfig()
    if n_components != 2:
        from bioimage_pipeline.puncta.gmm_multi_start import _detector_based_init_peaks

        return {
            "detector_based": _detector_based_init_peaks(peaks, n_components, patch),
        }
    return generate_two_component_init_sets(
        peaks,
        patch,
        obj,
        config=cfg,
        single_component=single_component,
    )


def fit_mixture_with_init(
    patch: ObjectPatch,
    init_peaks: list[PeakCandidate],
    *,
    n_components: int,
    config: PunctaDeclumpConfig,
    initialization_method: str,
) -> tuple[MixtureFitResult | None, OptimizerDiagnostics | None, list[PeakCandidate]]:
    """Fit a mixture model from explicit initialization peaks with optimizer metadata."""
    mixture = GaussianMixtureFitter(config)
    arrays = mixture._extract_weighted_patch(patch)
    if arrays.values.size < 5:
        return None, None, init_peaks

    sigma_guess = max(config.expected_single_spot_diameter / 2.355, config.min_sigma)
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

    initial = np.array(params, dtype=np.float64)
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

    try:
        result: OptimizeResult = least_squares(
            residuals,
            initial,
            bounds=(lower_arr, upper_arr),
            max_nfev=3000,
        )
    except Exception as exc:
        diag = OptimizerDiagnostics(
            converged=False,
            status=-1,
            message=str(exc),
            n_iterations=0,
            bounds_hit=[],
            initial_params=initial.tolist(),
            final_params=[],
        )
        return None, diag, init_peaks

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
    components, merge_notes = mixture._merge_close_components(components)
    diag = OptimizerDiagnostics(
        converged=result.status > 0,
        status=int(result.status),
        message=str(result.message),
        n_iterations=int(result.nfev),
        bounds_hit=_bounds_hit_flags(result.x, lower_arr, upper_arr, n_components=n_components),
        initial_params=initial.tolist(),
        final_params=result.x.tolist(),
    )

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
        merge_notes=merge_notes,
    )
    fit.fit_error = (
        None
        if len(components) == n_components
        else f"post_merge_collapsed_{pre_merge_count}_to_{len(components)}"
    )
    return fit, diag, init_peaks[:n_components]


def apply_filter_ablation(
    obj: ObjectInfo,
    patch: ObjectPatch,
    mixture: MixtureFitResult,
    peaks: list[PeakCandidate],
    *,
    config: PunctaDeclumpConfig,
    filter_mode: FilterMode,
) -> tuple[list[ComponentRecord], list[str]]:
    """Evaluate component acceptance under ablation filter modes."""
    if not mixture.fit_succeeded or not mixture.components:
        return [], ["mixture_fit_failed"]

    records: list[ComponentRecord] = []
    rejections: list[str] = []

    if filter_mode == "none":
        for index, component in enumerate(mixture.components):
            records.append(
                ComponentRecord(
                    component_id=component.component_id,
                    initial_row=component.initial_row,
                    initial_col=component.initial_col,
                    fitted_row=component.fitted_row,
                    fitted_col=component.fitted_col,
                    amplitude=component.amplitude,
                    sigma_x=component.sigma_col,
                    sigma_y=component.sigma_row,
                    background=component.background,
                    acceptance_filter=None,
                    accepted=True,
                )
            )
        return records, rejections

    filt = CandidateFilter(config)
    filt.reset()
    for index, component in enumerate(mixture.components):
        peak = peaks[min(index, len(peaks) - 1)]
        if filter_mode == "duplicate_only":
            accepted = True
            reason = None
            for prior in records:
                distance = math.hypot(
                    component.fitted_col - prior.fitted_col,
                    component.fitted_row - prior.fitted_row,
                )
                if distance < config.min_center_separation:
                    accepted = False
                    reason = "duplicate_center_too_close"
                    break
        else:
            evaluated = filt.evaluate_mixture_components(
                obj,
                peaks,
                mixture,
                candidate_id_start=1,
                object_mask=patch.object_mask,
                patch=patch,
            )
            candidate = evaluated[index]
            accepted = candidate.accepted
            reason = candidate.rejection_reason

        if reason:
            rejections.append(f"component_{component.component_id}:{reason}")
        records.append(
            ComponentRecord(
                component_id=component.component_id,
                initial_row=component.initial_row,
                initial_col=component.initial_col,
                fitted_row=component.fitted_row,
                fitted_col=component.fitted_col,
                amplitude=component.amplitude,
                sigma_x=component.sigma_col,
                sigma_y=component.sigma_row,
                background=component.background,
                acceptance_filter=reason,
                accepted=accepted,
            )
        )
    return records, rejections


def _legacy_candidate_counts(n_peaks: int, config: PunctaDeclumpConfig) -> list[int]:
    peak_count = max(1, n_peaks)
    candidate_counts = {1}
    candidate_counts.add(min(peak_count, config.gmm_max_components))
    if peak_count > 1:
        candidate_counts.add(2)
        candidate_counts.add(
            min(max(1, peak_count - config.gmm_try_component_delta), config.gmm_max_components)
        )
        candidate_counts.add(
            min(peak_count + config.gmm_try_component_delta, config.gmm_max_components)
        )
    return sorted(candidate_counts)


class GmmDiagnosticProbe:
    """Run instrumented GMM diagnostics on synthetic or direct-fit patches."""

    def __init__(self, config: PunctaDeclumpConfig | None = None) -> None:
        self.config = config or PunctaDeclumpConfig()
        self.single_fitter = EllipticalGaussianFitter(self.config)
        self.selector = GaussianModelSelector(self.config)

    def probe_patch(
        self,
        patch: ObjectPatch,
        obj: ObjectInfo,
        peaks: list[PeakCandidate],
        *,
        run_name: str = "direct",
        ground_truth_case: str = "direct",
        n_raw_peaks: int | None = None,
        n_filtered_peaks: int | None = None,
    ) -> GmmProbeReport:
        n_filtered = n_filtered_peaks if n_filtered_peaks is not None else len(peaks)
        n_raw = n_raw_peaks if n_raw_peaks is not None else len(peaks)

        primary = peaks[0] if peaks else PeakCandidate(
            row=obj.brightest_row,
            col=obj.brightest_col,
            intensity=obj.brightest_intensity,
        )
        single = self.single_fitter.fit_peak(patch, primary, component_id=1, n_components_in_model=1)
        single_bic = self.selector._single_component_bic(patch, single)
        single_aic = self.selector._single_component_aic(patch, single)

        init_sets = generate_init_peak_sets(
            peaks,
            patch,
            obj,
            n_components=2,
            single_component=single,
        )

        # Multi-start: pick best BIC among all 2-component strategies
        best_two: MixtureFitResult | None = None
        best_two_diag: OptimizerDiagnostics | None = None
        best_two_init_name = ""
        best_two_init_peaks: list[PeakCandidate] = []
        for name, init_peaks in init_sets.items():
            fit, diag, used_peaks = fit_mixture_with_init(
                patch,
                init_peaks,
                n_components=2,
                config=self.config,
                initialization_method=name,
            )
            if fit is None or not fit.fit_succeeded:
                continue
            if best_two is None or fit.bic < best_two.bic:
                best_two = fit
                best_two_diag = diag
                best_two_init_name = name
                best_two_init_peaks = used_peaks
        if best_two is not None:
            init_sets[InitStrategy.MULTI_START_BEST.value] = best_two_init_peaks

        model_attempts: list[ModelAttemptRecord] = []

        # 1-component (production path)
        model_attempts.append(self._record_single(single, single_bic, single_aic))

        # 2-component attempts for each init strategy
        for init_name, init_peaks in init_sets.items():
            fit, diag, used_peaks = fit_mixture_with_init(
                patch,
                init_peaks,
                n_components=2,
                config=self.config,
                initialization_method=init_name,
            )
            model_attempts.append(
                self._record_mixture_attempt(
                    n_components=2,
                    fit=fit,
                    diag=diag,
                    init_peaks=used_peaks,
                    initialization_method=init_name,
                    obj=obj,
                    patch=patch,
                    peaks=peaks,
                    filter_mode="full",
                )
            )

        # 3-component if gated open
        n3_gate = "blocked: need n_filtered>=3 or n_raw>=3 unless 2-comp poor"
        try_three = n_filtered >= 3 or n_raw >= 3
        fit_three: MixtureFitResult | None = None
        fit_three_diag: OptimizerDiagnostics | None = None
        if try_three:
            fit_three, fit_three_diag, used = fit_mixture_with_init(
                patch,
                init_sets[InitStrategy.DETECTOR_BASED.value],
                n_components=3,
                config=self.config,
                initialization_method=InitStrategy.DETECTOR_BASED.value,
            )
            model_attempts.append(
                self._record_mixture_attempt(
                    n_components=3,
                    fit=fit_three,
                    diag=fit_three_diag,
                    init_peaks=used,
                    initialization_method=InitStrategy.DETECTOR_BASED.value,
                    obj=obj,
                    patch=patch,
                    peaks=peaks,
                    filter_mode="full",
                )
            )
        else:
            model_attempts.append(
                ModelAttemptRecord(
                    n_components=3,
                    attempted=False,
                    initialization_method=InitStrategy.DETECTOR_BASED.value,
                    initialization_details=n3_gate,
                    initial_centers=[],
                    converged=False,
                    optimizer_status=-1,
                    optimizer_message=n3_gate,
                    n_iterations=0,
                    bounds_hit=[],
                    fitted_centers=[],
                    fitted_amplitudes=[],
                    fitted_sigma_x=[],
                    fitted_sigma_y=[],
                    background=patch.background_level,
                    rss=None,
                    r_squared=None,
                    aic=None,
                    bic=None,
                    pairwise_component_distances_px=[],
                    post_merge_component_count=0,
                    merge_notes=[],
                    model_level_rejection_reason=n3_gate,
                )
            )

        comparison = self.selector.select_balanced_model(
            patch,
            peaks,
            single_component=single,
            n_filtered_peaks=n_filtered,
            n_raw_peaks=n_raw,
            obj=obj,
        )

        best_mix = comparison.best_mixture
        bic_delta = None
        aic_delta = None
        if best_mix is not None and best_mix.n_components >= 2:
            bic_delta = best_mix.bic - single_bic
            aic_delta = best_mix.aic - self.selector._single_component_aic(patch, single)

        second_rejection = self._exact_second_component_rejection(
            comparison,
            best_two,
            obj,
            patch,
            peaks,
        )

        selection_diag = ModelSelectionDiagnostics(
            peak_count_used_for_init=len(peaks),
            n_filtered_peaks=n_filtered,
            n_raw_peaks=n_raw,
            candidate_component_counts_legacy_select_best=_legacy_candidate_counts(
                len(peaks),
                self.config,
            ),
            balanced_model_attempted_n2=True,
            balanced_model_attempted_n3=try_three,
            balanced_model_n3_gate_reason=n3_gate if not try_three else "attempted",
            single_bic=single_bic,
            single_aic=single_aic,
            best_mixture_n_components=best_mix.n_components if best_mix else None,
            best_mixture_bic=best_mix.bic if best_mix else None,
            best_mixture_aic=best_mix.aic if best_mix else None,
            bic_delta_2_vs_1=bic_delta,
            aic_delta_2_vs_1=aic_delta,
            selection_reason=comparison.selection_reason,
            rejected_component_reason=comparison.rejected_component_reason,
            exact_second_component_rejection=second_rejection,
        )

        ablation_results = self._run_ablations(
            patch,
            obj,
            peaks,
            single,
            init_sets,
            best_two,
            best_two_init_name,
        )

        return GmmProbeReport(
            run_name=run_name,
            ground_truth_case=ground_truth_case,
            object_id=obj.label,
            n_detected_peaks=len(peaks),
            detected_peak_coords=[(p.col, p.row) for p in peaks],
            model_attempts=model_attempts,
            model_selection=selection_diag,
            ablation_results=ablation_results,
            production_path_notes=(
                "select_balanced_model always attempts n=2 when max_components>=2; "
                "n=3 gated by peak count unless 2-comp fit still poor."
            ),
        )

    def probe_synthetic_run(
        self,
        *,
        data_root: Path,
        run_name: str,
        ground_truth_case: str,
        image_case: str | None = None,
    ) -> GmmProbeReport:
        image_case = image_case or ground_truth_case
        image = tifffile.imread(data_root / "images" / image_case / "synthetic_noisy.tif")
        mask = tifffile.imread(data_root / "masks" / image_case / "synthetic_mask.tif") > 0

        labels, objects = ConnectedObjectAnalyzer().analyze(mask.astype(bool), np.asarray(image))
        if not objects:
            raise ValueError(f"No mask objects found for {run_name}")

        from bioimage_pipeline.puncta.candidate_detectors.python_log import PythonLoGDetector

        peak_table = PythonLoGDetector().detect(np.asarray(image), config=self.config)
        assigned = assign_peaks_to_objects(labels, objects, peak_table, self.config)

        obj = max(objects, key=lambda item: item.area)
        peaks = assigned.get(obj.label, [])
        patch = build_object_patch(np.asarray(image), mask, obj, self.config)

        return self.probe_patch(
            patch,
            obj,
            peaks,
            run_name=run_name,
            ground_truth_case=ground_truth_case,
            n_raw_peaks=len(peaks),
            n_filtered_peaks=len(peaks),
        )

    def _record_single(
        self,
        single: GaussianComponent,
        single_bic: float,
        single_aic: float,
    ) -> ModelAttemptRecord:
        n_obs = 1
        rss = single.residual_rmse**2 * n_obs if single.fit_succeeded else None
        return ModelAttemptRecord(
            n_components=1,
            attempted=True,
            initialization_method="single_peak",
            initialization_details="EllipticalGaussianFitter.fit_peak",
            initial_centers=[(single.initial_col, single.initial_row)],
            converged=single.fit_succeeded,
            optimizer_status=1 if single.fit_succeeded else 0,
            optimizer_message=single.fit_error or "ok",
            n_iterations=0,
            bounds_hit=[],
            fitted_centers=[(single.fitted_col, single.fitted_row)]
            if single.fit_succeeded
            else [],
            fitted_amplitudes=[single.amplitude] if single.fit_succeeded else [],
            fitted_sigma_x=[single.sigma_col] if single.fit_succeeded else [],
            fitted_sigma_y=[single.sigma_row] if single.fit_succeeded else [],
            background=single.background,
            rss=rss,
            r_squared=single.r_squared if single.fit_succeeded else None,
            aic=single_aic,
            bic=single_bic,
            pairwise_component_distances_px=[],
            post_merge_component_count=1 if single.fit_succeeded else 0,
            merge_notes=[],
            raw_fit_succeeded=single.fit_succeeded,
        )

    def _record_mixture_attempt(
        self,
        *,
        n_components: int,
        fit: MixtureFitResult | None,
        diag: OptimizerDiagnostics | None,
        init_peaks: list[PeakCandidate],
        initialization_method: str,
        obj: ObjectInfo,
        patch: ObjectPatch,
        peaks: list[PeakCandidate],
        filter_mode: FilterMode,
    ) -> ModelAttemptRecord:
        if fit is None or diag is None:
            return ModelAttemptRecord(
                n_components=n_components,
                attempted=True,
                initialization_method=initialization_method,
                initialization_details="optimizer_failed",
                initial_centers=[(p.col, p.row) for p in init_peaks],
                converged=False,
                optimizer_status=diag.status if diag else -1,
                optimizer_message=diag.message if diag else "fit returned None",
                n_iterations=diag.n_iterations if diag else 0,
                bounds_hit=diag.bounds_hit if diag else [],
                fitted_centers=[],
                fitted_amplitudes=[],
                fitted_sigma_x=[],
                fitted_sigma_y=[],
                background=patch.background_level,
                rss=None,
                r_squared=None,
                aic=None,
                bic=None,
                pairwise_component_distances_px=[],
                post_merge_component_count=0,
                merge_notes=[],
                model_level_rejection_reason=diag.message if diag else "fit_failed",
            )

        fitted_centers = [(c.fitted_col, c.fitted_row) for c in fit.components]
        comp_records, comp_rejections = apply_filter_ablation(
            obj,
            patch,
            fit,
            peaks,
            config=self.config,
            filter_mode=filter_mode,
        )
        rss = fit.residual_rmse**2 * max(int(patch.object_mask.sum()), 1)
        return ModelAttemptRecord(
            n_components=n_components,
            attempted=True,
            initialization_method=initialization_method,
            initialization_details=f"init_peaks={[(p.col, p.row) for p in init_peaks]}",
            initial_centers=[(p.col, p.row) for p in init_peaks],
            converged=diag.converged,
            optimizer_status=diag.status,
            optimizer_message=diag.message,
            n_iterations=diag.n_iterations,
            bounds_hit=diag.bounds_hit,
            fitted_centers=fitted_centers,
            fitted_amplitudes=[c.amplitude for c in fit.components],
            fitted_sigma_x=[c.sigma_col for c in fit.components],
            fitted_sigma_y=[c.sigma_row for c in fit.components],
            background=fit.background,
            rss=rss,
            r_squared=fit.r_squared,
            aic=fit.aic,
            bic=fit.bic,
            pairwise_component_distances_px=_pairwise_distances(fitted_centers),
            post_merge_component_count=len(fit.components),
            merge_notes=list(fit.merge_notes),
            model_level_rejection_reason=fit.fit_error,
            component_records=comp_records,
            component_rejection_reasons=comp_rejections,
            raw_fit_succeeded=fit.fit_succeeded,
        )

    def _exact_second_component_rejection(
        self,
        comparison,
        best_two: MixtureFitResult | None,
        obj: ObjectInfo,
        patch: ObjectPatch,
        peaks: list[PeakCandidate],
    ) -> str:
        if comparison.best_mixture is None:
            return comparison.rejected_component_reason or "no_successful_multi_component_fit"

        if comparison.best_mixture.n_components <= 1:
            if comparison.best_mixture.merge_notes:
                return "; ".join(comparison.best_mixture.merge_notes)
            return comparison.rejected_component_reason or "multi_component_collapsed_to_one"

        if isinstance(comparison.selected, MixtureFitResult):
            filt = CandidateFilter(self.config)
            filt.reset()
            rejections: list[str] = []
            accepted = 0
            for index, component in enumerate(comparison.best_mixture.components):
                peak = peaks[min(index, len(peaks) - 1)]
                candidate = filt.evaluate_component(
                    obj,
                    peak,
                    component,
                    candidate_id=index + 1,
                    component_id=component.component_id,
                    path="gmm",
                    object_mask=patch.object_mask,
                    patch=patch,
                )
                if candidate.accepted:
                    accepted += 1
                elif candidate.rejection_reason:
                    rejections.append(
                        f"component_{component.component_id}:{candidate.rejection_reason}"
                    )
            if accepted >= 2:
                return "none_second_component_accepted"
            if rejections:
                return "; ".join(rejections)

        if comparison.rejected_component_reason:
            return comparison.rejected_component_reason
        return "unknown"

    def _run_ablations(
        self,
        patch: ObjectPatch,
        obj: ObjectInfo,
        peaks: list[PeakCandidate],
        single: GaussianComponent,
        init_sets: dict[str, list[PeakCandidate]],
        best_two: MixtureFitResult | None,
        best_two_init_name: str,
    ) -> list[AblationResult]:
        configs = [
            ("detector_based", InitStrategy.DETECTOR_BASED.value, "full"),
            ("symmetric_two_component", f"{InitStrategy.SYMMETRIC_X.value}_sep2", "full"),
            ("multi_start_best", InitStrategy.MULTI_START_BEST.value, "full"),
            ("multi_start_no_filters", InitStrategy.MULTI_START_BEST.value, "none"),
            ("multi_start_duplicate_only", InitStrategy.MULTI_START_BEST.value, "duplicate_only"),
        ]
        results: list[AblationResult] = []
        for mode, init_key, filter_mode in configs:
            init_peaks = init_sets.get(init_key)
            if init_peaks is None and mode.startswith("multi_start"):
                if best_two is None:
                    results.append(
                        AblationResult(
                            mode=mode,
                            init_strategy=init_key,
                            filter_mode=filter_mode,  # type: ignore[arg-type]
                            predicted_accepted_count=0,
                            true_positives=0,
                            model_selection_reason=None,
                            second_component_rejection="no_converged_two_component_fit",
                            best_two_component_bic=None,
                            notes=f"best init was {best_two_init_name or 'none'}",
                        )
                    )
                    continue
                init_peaks = init_sets.get(best_two_init_name, init_sets.get(InitStrategy.DETECTOR_BASED.value, peaks))
            if init_peaks is None:
                init_peaks = init_sets.get(InitStrategy.DETECTOR_BASED.value, peaks)

            fit, _, _ = fit_mixture_with_init(
                patch,
                init_peaks,
                n_components=2,
                config=self.config,
                initialization_method=init_key,
            )
            if fit is None or not fit.fit_succeeded:
                results.append(
                    AblationResult(
                        mode=mode,
                        init_strategy=init_key,
                        filter_mode=filter_mode,  # type: ignore[arg-type]
                        predicted_accepted_count=0,
                        true_positives=0,
                        model_selection_reason="fit_failed",
                        second_component_rejection="fit_failed",
                        best_two_component_bic=None,
                    )
                )
                continue

            comp_records, rejections = apply_filter_ablation(
                obj,
                patch,
                fit,
                peaks,
                config=self.config,
                filter_mode=filter_mode,  # type: ignore[arg-type]
            )
            accepted = sum(1 for record in comp_records if record.accepted)
            second_rejection = rejections[1] if len(rejections) > 1 else (
                rejections[0] if len(rejections) == 1 and accepted < 2 else None
            )
            comparison = self.selector._compare_single_vs_mixture(
                patch,
                single,
                self.selector._single_component_bic(patch, single),
                fit if fit.n_components > 1 else None,
                [2],
                single_aic=self.selector._single_component_aic(patch, single),
            )
            results.append(
                AblationResult(
                    mode=mode,
                    init_strategy=init_key,
                    filter_mode=filter_mode,  # type: ignore[arg-type]
                    predicted_accepted_count=accepted,
                    true_positives=accepted,
                    model_selection_reason=comparison.selection_reason,
                    second_component_rejection=second_rejection,
                    best_two_component_bic=fit.bic,
                    notes=f"merge_notes={';'.join(fit.merge_notes)}",
                )
            )
        return results


def write_probe_outputs(report: GmmProbeReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "gmm_model_diagnostics.json"
    csv_path = output_dir / "gmm_model_diagnostics.csv"

    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for attempt in report.model_attempts:
        rows.append(
            {
                "n_components": attempt.n_components,
                "attempted": attempt.attempted,
                "initialization_method": attempt.initialization_method,
                "converged": attempt.converged,
                "optimizer_status": attempt.optimizer_status,
                "optimizer_message": attempt.optimizer_message,
                "n_iterations": attempt.n_iterations,
                "bounds_hit": ";".join(attempt.bounds_hit),
                "rss": attempt.rss,
                "r_squared": attempt.r_squared,
                "aic": attempt.aic,
                "bic": attempt.bic,
                "post_merge_component_count": attempt.post_merge_component_count,
                "pairwise_distances_px": ";".join(
                    f"{d:.3f}" for d in attempt.pairwise_component_distances_px
                ),
                "merge_notes": ";".join(attempt.merge_notes),
                "model_level_rejection_reason": attempt.model_level_rejection_reason,
                "component_rejection_reasons": ";".join(attempt.component_rejection_reasons),
                "fitted_centers": str(attempt.fitted_centers),
                "initial_centers": str(attempt.initial_centers),
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return json_path, csv_path


def make_clean_doublet_patch(
    *,
    separation_px: float = 4.0,
    sigma: float = 2.2,
    amplitude: float = 1800.0,
    background: float = 40.0,
    shape: tuple[int, int] = (48, 48),
    center: tuple[float, float] = (24.0, 24.0),
) -> tuple[ObjectPatch, ObjectInfo, list[tuple[float, float]]]:
    """Build a noise-free two-Gaussian patch for direct-fit unit tests."""
    height, width = shape
    center_row, center_col = center
    half = separation_px / 2.0
    true_centers = [
        (center_col - half, center_row),
        (center_col + half, center_row),
    ]

    rows, cols = np.indices((height, width))
    image = np.full((height, width), background, dtype=np.float64)
    for x_true, y_true in true_centers:
        image += amplitude * np.exp(
            -((cols - x_true) ** 2 + (rows - y_true) ** 2) / (2.0 * sigma**2)
        )

    mask = np.ones((height, width), dtype=bool)
    obj = ObjectInfo(
        label=1,
        area=float(mask.sum()),
        equivalent_diameter=float(min(height, width)),
        bbox=(0, 0, height, width),
        centroid=(center_row, center_col),
        brightest_row=center_row,
        brightest_col=center_col,
        brightest_intensity=float(image.max()),
        major_axis_length=float(width),
        minor_axis_length=float(height / 4),
        elongation=4.0,
    )
    patch = ObjectPatch(
        object_id=1,
        row_offset=0,
        col_offset=0,
        corrected=np.clip(image - background, 0.0, None),
        object_mask=mask,
        background_level=background,
        global_bbox=(0, 0, height, width),
        raw=image,
    )
    return patch, obj, true_centers
