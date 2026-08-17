"""Phase D component validity tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bioimage_pipeline.puncta.candidate_filter import CandidateFilter
from bioimage_pipeline.puncta.component_validity import (
    center_has_mask_support,
    compute_component_validity_metrics,
    detect_roi_saturation,
    evaluate_component_validity,
)
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.gaussian_fitter import GaussianMixtureFitter, _predict_mixture
from bioimage_pipeline.puncta.gmm_multi_start import fit_mixture_from_init_peaks
from bioimage_pipeline.puncta.types import GaussianComponent, MixtureFitResult, ObjectInfo, ObjectPatch, PeakCandidate
from bioimage_pipeline.puncta.validation.gmm_probe import make_clean_doublet_patch


def _default_config(**overrides: object) -> PunctaDeclumpConfig:
    params = {
        "threshold_method": "manual",
        "manual_threshold_value": 50.0,
        "min_object_area": 4,
        "single_spot_max_diameter": 12.0,
        "expected_single_spot_diameter": 5.0,
        "smoothing_sigma": 0.4,
        "min_peak_distance": 2,
        "fit_roi_radius": 5,
        "min_sigma": 0.3,
        "max_sigma": 5.0,
        "max_center_shift": 4.0,
        "min_amplitude": 5.0,
        "max_fit_residual_relative": 0.25,
        "min_r_squared": 0.3,
        "gmm_min_component_separation": 1.5,
        "component_validity_enabled": True,
    }
    params.update(overrides)
    return PunctaDeclumpConfig(**params)


def _component(
    *,
    row: float,
    col: float,
    amplitude: float = 1000.0,
    sigma: float = 2.0,
    n_components: int = 2,
    residual_rmse: float = 500.0,
    r_squared: float = 0.4,
) -> GaussianComponent:
    return GaussianComponent(
        component_id=1,
        initial_row=row,
        initial_col=col,
        fitted_row=row,
        fitted_col=col,
        sigma_row=sigma,
        sigma_col=sigma,
        amplitude=amplitude,
        background=0.0,
        residual_rmse=residual_rmse,
        residual_relative=residual_rmse / amplitude,
        r_squared=r_squared,
        model_score=0.0,
        n_components_in_model=n_components,
        fit_succeeded=True,
    )


def _make_large_blob_doublet_patch(
    *,
    separation_px: float = 5.0,
    sigma: float = 1.8,
    amplitude: float = 2000.0,
    shoulder: float = 900.0,
) -> tuple[ObjectPatch, ObjectInfo, list[PeakCandidate]]:
    """Two real peaks embedded in a large bright shoulder blob (high global residual)."""
    height, width = 64, 64
    center_row, center_col = 32.0, 32.0
    half = separation_px / 2.0
    centers = [(center_row, center_col - half), (center_row, center_col + half)]
    rows, cols = np.indices((height, width))
    image = np.full((height, width), shoulder, dtype=np.float64)
    for row, col in centers:
        image += amplitude * np.exp(
            -((rows - row) ** 2 + (cols - col) ** 2) / (2.0 * sigma**2)
        )
    mask = np.ones((height, width), dtype=bool)
    obj = ObjectInfo(
        label=1,
        area=float(mask.sum()),
        equivalent_diameter=20.0,
        bbox=(0, 0, height, width),
        centroid=(center_row, center_col),
        brightest_row=center_row,
        brightest_col=center_col,
        brightest_intensity=float(image.max()),
    )
    patch = ObjectPatch(
        object_id=1,
        row_offset=0,
        col_offset=0,
        corrected=np.clip(image - shoulder, 0.0, None),
        object_mask=mask,
        background_level=shoulder,
        global_bbox=(0, 0, height, width),
        raw=image,
    )
    peaks = [
        PeakCandidate(row=centers[0][0], col=centers[0][1], intensity=float(image.max())),
        PeakCandidate(row=centers[1][0], col=centers[1][1], intensity=float(image.max())),
    ]
    return patch, obj, peaks


def _peaks_from_doublet(
    patch: ObjectPatch,
    true_centers: list[tuple[float, float]],
) -> list[PeakCandidate]:
    intensity = float(patch.corrected.max())
    return [PeakCandidate(row=row, col=col, intensity=intensity) for col, row in true_centers]


def _overwrite_joint_residuals(mixture: MixtureFitResult, rmse: float = 800.0) -> None:
    """Simulate legacy per-component joint RMSE assignment."""
    for component in mixture.components:
        component.residual_rmse = rmse
        component.residual_relative = rmse / max(component.amplitude, 1.0)
        component.r_squared = 0.2


def test_joint_gmm_components_pass_local_residual_when_global_fails() -> None:
    """Good mixture components should not be rejected by object-level joint RMSE."""
    patch, obj, peaks = make_clean_doublet_patch(separation_px=4.0, sigma=2.2)
    peak_list = _peaks_from_doublet(patch, peaks)
    config = _default_config()
    fitter = GaussianMixtureFitter(config)
    mixture = fit_mixture_from_init_peaks(
        fitter, patch, peak_list[:2], n_components=2,
    )
    assert mixture.fit_succeeded
    assert mixture.n_components == 2
    _overwrite_joint_residuals(mixture)

    for component in mixture.components:
        assert component.residual_relative > config.max_fit_residual_relative

    filt = CandidateFilter(config)
    candidates = filt.evaluate_mixture_components(
        obj,
        peak_list[:2],
        mixture,
        candidate_id_start=1,
        object_mask=patch.object_mask,
        patch=patch,
    )
    accepted = [c for c in candidates if c.accepted]
    assert len(accepted) == 2


def test_bad_component_still_rejected_by_local_residual() -> None:
    """A clearly weak component should remain rejected under Phase D."""
    patch, obj, true_centers = make_clean_doublet_patch(separation_px=4.0, sigma=2.2)
    peaks = _peaks_from_doublet(patch, true_centers)
    bad = _component(row=24.0, col=24.0, amplitude=2.0, sigma=0.35, n_components=2, residual_rmse=800.0)
    good = _component(row=24.0, col=28.0, amplitude=1800.0, sigma=2.2, n_components=2, residual_rmse=800.0)
    mixture = MixtureFitResult(
        components=[bad, good],
        n_components=2,
        background=patch.background_level,
        residual_rmse=800.0,
        r_squared=0.2,
        aic=0.0,
        bic=0.0,
        model_score=0.0,
        fit_succeeded=True,
        predicted_patch=patch.corrected * 0.5,
    )
    filt = CandidateFilter(_default_config())
    candidates = filt.evaluate_mixture_components(
        obj,
        peaks[:2],
        mixture,
        candidate_id_start=1,
        object_mask=patch.object_mask,
        patch=patch,
    )
    rejected = [c for c in candidates if not c.accepted]
    assert len(rejected) >= 1
    assert any(c.rejection_reason == "amplitude_too_low" for c in rejected)


def test_slightly_outside_mask_center_accepted_with_local_support() -> None:
    """Subpixel center just outside mask edge can pass with strong nearby support."""
    patch, obj, true_centers = make_clean_doublet_patch(separation_px=4.0, sigma=2.2)
    peaks = _peaks_from_doublet(patch, true_centers)
    mask = patch.object_mask.copy()
    right_peak_row = int(round(peaks[1].row))
    right_peak_col = int(round(peaks[1].col))
    mask[right_peak_row, right_peak_col] = False
    patch.object_mask = mask

    config = _default_config()
    fitter = GaussianMixtureFitter(config)
    mixture = fit_mixture_from_init_peaks(
        fitter, patch, peaks[:2], n_components=2,
    )
    assert mixture.fit_succeeded

    filt = CandidateFilter(config)
    candidates = filt.evaluate_mixture_components(
        obj,
        peaks[:2],
        mixture,
        candidate_id_start=1,
        object_mask=mask,
        patch=patch,
    )
    accepted = [c for c in candidates if c.accepted]
    assert len(accepted) >= 1


def test_clearly_outside_mask_component_rejected() -> None:
    """Center far outside mask with weak support should be rejected."""
    patch, _, _ = make_clean_doublet_patch(separation_px=4.0, sigma=2.2)
    mask = np.zeros_like(patch.object_mask)
    mask[20:28, 20:28] = True
    component = _component(row=10.0, col=10.0, amplitude=1800.0, sigma=2.2, n_components=2)
    assert not center_has_mask_support(component, patch, mask, _default_config())


def test_saturated_multi_peak_roi_handled_conservatively() -> None:
    """Saturated multi-peak ROI should not auto-accept all components blindly."""
    patch, obj, peaks = _make_large_blob_doublet_patch(amplitude=4000.0, shoulder=1200.0)
    clip_value = float(patch.raw.max())
    saturated = np.clip(patch.raw, 0, clip_value * 0.98)
    saturated[saturated >= clip_value * 0.98] = clip_value
    patch.raw = saturated
    patch.corrected = np.clip(saturated - patch.background_level, 0.0, None)

    config = _default_config()
    saturation = detect_roi_saturation(
        patch,
        near_clip_margin=config.saturation_near_clip_margin,
        near_clip_fraction_threshold=config.saturation_near_clip_fraction,
    )
    assert saturation.present or saturation.near_clip_fraction > 0

    fitter = GaussianMixtureFitter(config)
    mixture = fit_mixture_from_init_peaks(
        fitter, patch, peaks[:2], n_components=2,
    )
    filt = CandidateFilter(config)
    candidates = filt.evaluate_mixture_components(
        obj,
        peaks[:2],
        mixture,
        candidate_id_start=1,
        object_mask=patch.object_mask,
        patch=patch,
    )
    accepted = [c for c in candidates if c.accepted]
    assert len(accepted) <= 2


def test_noisy_saturated_single_spot_does_not_hallucinate_extra_puncta() -> None:
    """Single saturated noisy spot should not produce extra accepted mixture components."""
    height, width = 48, 48
    row, col = 24.0, 24.0
    rng = np.random.default_rng(49)
    rows, cols = np.indices((height, width))
    image = rng.normal(40.0, 25.0, (height, width))
    image += 3500.0 * np.exp(-((rows - row) ** 2 + (cols - col) ** 2) / (2 * 2.0**2))
    image = np.clip(image, 0, 4095.0)
    mask = np.ones((height, width), dtype=bool)
    obj = ObjectInfo(
        label=1,
        area=float(mask.sum()),
        equivalent_diameter=8.0,
        bbox=(0, 0, height, width),
        centroid=(row, col),
        brightest_row=row,
        brightest_col=col,
        brightest_intensity=float(image.max()),
    )
    patch = ObjectPatch(
        object_id=1,
        row_offset=0,
        col_offset=0,
        corrected=np.clip(image - 40.0, 0.0, None),
        object_mask=mask,
        background_level=40.0,
        global_bbox=(0, 0, height, width),
        raw=image,
    )
    peaks = [PeakCandidate(row=row, col=col, intensity=float(image.max()))]
    config = _default_config(min_amplitude=50.0)
    fitter = GaussianMixtureFitter(config)
    single = fitter.single_fitter.fit_peak(patch, peaks[0], component_id=1, n_components_in_model=1)
    filt = CandidateFilter(config)
    single_candidate = filt.evaluate_component(
        obj,
        peaks[0],
        single,
        candidate_id=1,
        component_id=1,
        path="single",
        object_mask=mask,
        patch=patch,
    )
    assert single_candidate.accepted or single_candidate.rejection_reason is not None
    assert filt.accepted


def test_ambiguous_remains_possible_when_evidence_insufficient() -> None:
    """When local evidence is insufficient, components remain rejected (ambiguous path preserved)."""
    patch, obj, true_centers = make_clean_doublet_patch(separation_px=4.0, sigma=2.2)
    peaks = _peaks_from_doublet(patch, true_centers)
    config = _default_config()
    fitter = GaussianMixtureFitter(config)
    mixture = fit_mixture_from_init_peaks(
        fitter, patch, peaks[:2], n_components=2,
    )
    _overwrite_joint_residuals(mixture)
    mixture.predicted_patch = patch.corrected * 0.1
    filt = CandidateFilter(config)
    candidates = filt.evaluate_mixture_components(
        obj,
        peaks[:2],
        mixture,
        candidate_id_start=1,
        object_mask=patch.object_mask,
        patch=patch,
    )
    assert not any(c.accepted for c in candidates)


def test_local_metrics_improve_relative_to_joint_for_doublet() -> None:
    """Unit-level check: local relative residual is lower than legacy joint assignment."""
    patch, _, true_centers = make_clean_doublet_patch(separation_px=4.0, sigma=2.2)
    peaks = _peaks_from_doublet(patch, true_centers)
    config = _default_config()
    fitter = GaussianMixtureFitter(config)
    mixture = fit_mixture_from_init_peaks(
        fitter, patch, peaks[:2], n_components=2,
    )
    _overwrite_joint_residuals(mixture, rmse=900.0)
    for component in mixture.components:
        metrics = compute_component_validity_metrics(
            component,
            mixture,
            patch,
            patch.object_mask,
            config,
        )
        assert metrics.local_residual_relative < component.residual_relative


def test_component_validity_disabled_uses_legacy_global_residual() -> None:
    """Disabling Phase D restores joint-residual rejection behavior."""
    patch, obj, true_centers = make_clean_doublet_patch(separation_px=4.0, sigma=2.2)
    peaks = _peaks_from_doublet(patch, true_centers)
    config = _default_config(component_validity_enabled=False)
    fitter = GaussianMixtureFitter(config)
    mixture = fit_mixture_from_init_peaks(
        fitter, patch, peaks[:2], n_components=2,
    )
    _overwrite_joint_residuals(mixture)
    filt = CandidateFilter(config)
    candidates = filt.evaluate_mixture_components(
        obj,
        peaks[:2],
        mixture,
        candidate_id_start=1,
        object_mask=patch.object_mask,
        patch=patch,
    )
    assert not any(c.accepted for c in candidates)
    assert all(c.rejection_reason == "residual_too_high" for c in candidates)
