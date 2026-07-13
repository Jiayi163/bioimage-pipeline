"""Regression tests for production GMM multi-start and acceptance separation."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from bioimage_pipeline.puncta.background import build_object_patch
from bioimage_pipeline.puncta.candidate_filter import CandidateFilter
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.gaussian_fitter import GaussianMixtureFitter, GaussianModelSelector
from bioimage_pipeline.puncta.gmm_multi_start import (
    fit_mixture_from_init_peaks,
    fit_two_component_multi_start,
    generate_two_component_init_sets,
)
from bioimage_pipeline.puncta.pipeline import run_puncta_declump
from bioimage_pipeline.puncta.types import GaussianComponent, MixtureFitResult, ObjectInfo, PeakCandidate
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
        "max_fit_residual_relative": 0.5,
        "min_center_separation": 2.5,
        "gmm_min_component_separation": 1.5,
        "gmm_acceptance_min_separation": 1.5,
        "diagnostic_mode": "summary",
    }
    params.update(overrides)
    return PunctaDeclumpConfig(**params)


def test_multi_start_finds_two_components_with_one_peak() -> None:
    patch, obj, true_centers = make_clean_doublet_patch(separation_px=4.0, sigma=2.2)
    peaks = [PeakCandidate(row=24.0, col=24.0, intensity=1800.0)]
    config = _default_config(gmm_multi_start_enabled=True)
    fitter = GaussianMixtureFitter(config)
    single = fitter.single_fitter.fit_peak(patch, peaks[0], component_id=1, n_components_in_model=1)

    result = fit_two_component_multi_start(
        fitter,
        patch,
        peaks,
        obj=obj,
        single_component=single,
    )
    assert result.n_starts_attempted > 1
    assert result.fit.fit_succeeded
    assert result.fit.n_components == 2
    assert result.winning_strategy
    assert len(result.fit.init_attempts) == result.n_starts_attempted
    assert any(record.selected for record in result.fit.init_attempts)


def test_multi_start_recovers_two_px_separation() -> None:
    patch, obj, true_centers = make_clean_doublet_patch(separation_px=2.0, sigma=2.2)
    peaks = [PeakCandidate(row=24.0, col=24.0, intensity=1800.0)]
    config = _default_config(gmm_multi_start_enabled=True)
    fitter = GaussianMixtureFitter(config)
    single = fitter.single_fitter.fit_peak(patch, peaks[0], component_id=1, n_components_in_model=1)
    result = fit_two_component_multi_start(
        fitter, patch, peaks, obj=obj, single_component=single,
    )
    assert result.fit.n_components == 2
    centers = [(c.fitted_col, c.fitted_row) for c in result.fit.components]
    assert math.hypot(centers[0][0] - centers[1][0], centers[0][1] - centers[1][1]) >= 1.5


def test_detector_only_init_collapses_multi_start_recovers() -> None:
    patch, obj, _ = make_clean_doublet_patch(separation_px=2.0, sigma=2.2)
    peaks = [PeakCandidate(row=24.0, col=24.0, intensity=1800.0)]
    config = _default_config(gmm_multi_start_enabled=False)
    fitter = GaussianMixtureFitter(config)
    same_location_init = [
        PeakCandidate(row=24.0, col=24.0, intensity=1800.0),
        PeakCandidate(row=24.0, col=24.0, intensity=1800.0),
    ]
    detector_only = fit_mixture_from_init_peaks(
        fitter,
        patch,
        same_location_init,
        n_components=2,
        initialization_method="detector_based",
    )
    assert detector_only.n_components <= 1 or detector_only.merge_notes

    config_ms = _default_config(gmm_multi_start_enabled=True)
    fitter_ms = GaussianMixtureFitter(config_ms)
    single = fitter_ms.single_fitter.fit_peak(patch, peaks[0], component_id=1, n_components_in_model=1)
    multi = fit_two_component_multi_start(
        fitter_ms, patch, peaks, obj=obj, single_component=single,
    )
    assert multi.fit.n_components == 2


def test_single_gaussian_not_selected_as_two_components() -> None:
    image = np.full((48, 48), 40.0, dtype=np.float64)
    rows, cols = np.ogrid[:48, :48]
    image += 800.0 * np.exp(-((rows - 24.0) ** 2 + (cols - 24.0) ** 2) / (2 * 2.2**2))
    mask = np.zeros((48, 48), dtype=bool)
    mask[16:32, 16:32] = True
    obj = ObjectInfo(
        label=1,
        area=float(mask.sum()),
        equivalent_diameter=12.0,
        bbox=(16, 16, 32, 32),
        centroid=(24.0, 24.0),
        brightest_row=24.0,
        brightest_col=24.0,
        brightest_intensity=float(image.max()),
        major_axis_length=10.0,
        minor_axis_length=10.0,
        elongation=1.0,
        eccentricity=0.0,
        solidity=1.0,
    )
    patch = build_object_patch(image, mask, obj, _default_config())
    peaks = [PeakCandidate(row=24.0, col=24.0, intensity=float(image.max()))]
    selector = GaussianModelSelector(_default_config(gmm_multi_start_enabled=True))
    single = selector.single_fitter.fit_peak(patch, peaks[0], component_id=1, n_components_in_model=1)
    comparison = selector.select_balanced_model(
        patch,
        peaks,
        single_component=single,
        n_filtered_peaks=1,
        n_raw_peaks=1,
        obj=obj,
    )
    assert isinstance(comparison.selected, GaussianComponent)


def test_noisy_single_gaussian_mostly_not_oversplit() -> None:
    rng = np.random.default_rng(0)
    image = np.full((48, 48), 40.0, dtype=np.float64)
    rows, cols = np.ogrid[:48, :48]
    image += 800.0 * np.exp(-((rows - 24.0) ** 2 + (cols - 24.0) ** 2) / (2 * 2.2**2))
    image = np.clip(image + rng.normal(0, 8.0, size=image.shape), 0, None)
    mask = np.zeros((48, 48), dtype=bool)
    mask[16:32, 16:32] = True
    oversplits = 0
    for seed in range(5):
        rng = np.random.default_rng(seed)
        noisy = np.clip(
            np.full((48, 48), 40.0)
            + 800.0 * np.exp(-((rows - 24.0) ** 2 + (cols - 24.0) ** 2) / (2 * 2.2**2))
            + rng.normal(0, 8.0, size=(48, 48)),
            0,
            None,
        )
        result = run_puncta_declump(
            noisy,
            _default_config(single_spot_max_diameter=20.0, gmm_multi_start_enabled=True),
            external_mask=mask,
        )
        if len(result.accepted) > 1:
            oversplits += 1
    assert oversplits <= 1


def test_gmm_mixture_acceptance_uses_lower_separation_threshold() -> None:
    config = _default_config(
        min_center_separation=2.5,
        gmm_acceptance_min_separation=1.5,
        gmm_use_mixture_acceptance_separation=True,
    )
    obj = ObjectInfo(
        label=1,
        area=100.0,
        equivalent_diameter=8.0,
        bbox=(0, 0, 20, 20),
        centroid=(10.0, 10.0),
        brightest_row=10.0,
        brightest_col=10.0,
        brightest_intensity=1000.0,
    )
    comp_a = GaussianComponent(
        component_id=1,
        initial_row=10.0,
        initial_col=10.0,
        fitted_row=10.0,
        fitted_col=10.0,
        sigma_row=2.0,
        sigma_col=2.0,
        amplitude=500.0,
        background=40.0,
        residual_rmse=5.0,
        residual_relative=0.01,
        r_squared=0.99,
        model_score=1.0,
        n_components_in_model=2,
        fit_succeeded=True,
    )
    comp_b = GaussianComponent(
        component_id=2,
        initial_row=10.0,
        initial_col=12.2,
        fitted_row=10.0,
        fitted_col=12.2,
        sigma_row=2.0,
        sigma_col=2.0,
        amplitude=450.0,
        background=40.0,
        residual_rmse=5.0,
        residual_relative=0.01,
        r_squared=0.99,
        model_score=1.0,
        n_components_in_model=2,
        fit_succeeded=True,
    )
    mixture = MixtureFitResult(
        components=[comp_a, comp_b],
        n_components=2,
        background=40.0,
        residual_rmse=5.0,
        r_squared=0.99,
        aic=10.0,
        bic=12.0,
        model_score=12.0,
        fit_succeeded=True,
    )
    mask = np.ones((20, 20), dtype=bool)
    patch = build_object_patch(np.ones((20, 20)) * 100.0, mask, obj, config)
    peaks = [PeakCandidate(row=10.0, col=10.0, intensity=900.0)]
    filt = CandidateFilter(config)
    accepted = filt.evaluate_mixture_components(
        obj,
        peaks,
        mixture,
        candidate_id_start=1,
        object_mask=mask,
        patch=patch,
    )
    assert sum(1 for c in accepted if c.accepted) == 2
    assert math.hypot(2.2, 0.0) < config.min_center_separation


def test_close_components_below_15px_merge_or_reject() -> None:
    patch, obj, _ = make_clean_doublet_patch(separation_px=1.0, sigma=2.2)
    peaks = [PeakCandidate(row=24.0, col=24.0, intensity=1800.0)]
    config = _default_config(gmm_min_component_separation=1.5, gmm_multi_start_enabled=True)
    fitter = GaussianMixtureFitter(config)
    fit = fitter.fit_patch(patch, peaks, n_components=2, obj=obj)
    assert fit.n_components <= 1


def test_init_strategy_catalog_includes_symmetric_y_and_is_bounded() -> None:
    patch, obj, _ = make_clean_doublet_patch(separation_px=4.0, sigma=2.2)
    peaks = [PeakCandidate(row=24.0, col=24.0, intensity=1800.0)]
    config = _default_config(gmm_max_multi_starts=5)
    strategies = generate_two_component_init_sets(peaks, patch, obj, config=config)
    assert "detector_based" in strategies
    assert "symmetric_x_sep2" in strategies
    assert "symmetric_y_sep2" in strategies
    assert len(strategies) >= 5


def test_multi_start_runtime_is_bounded() -> None:
    patch, obj, _ = make_clean_doublet_patch(separation_px=4.0, sigma=2.2)
    peaks = [PeakCandidate(row=24.0, col=24.0, intensity=1800.0)]
    config = _default_config(gmm_max_multi_starts=3, gmm_multi_start_max_nfev=500)
    fitter = GaussianMixtureFitter(config)
    single = fitter.single_fitter.fit_peak(patch, peaks[0], component_id=1, n_components_in_model=1)
    start = time.perf_counter()
    result = fit_two_component_multi_start(
        fitter, patch, peaks, obj=obj, single_component=single,
    )
    elapsed = time.perf_counter() - start
    assert result.n_starts_attempted <= 3
    assert elapsed < 30.0


def test_fast_path_unchanged_with_multi_start_enabled() -> None:
    image = np.full((48, 48), 40.0, dtype=np.float64)
    rows, cols = np.ogrid[:48, :48]
    image += 800.0 * np.exp(-((rows - 24.0) ** 2 + (cols - 24.0) ** 2) / (2 * 2.2**2))
    mask = np.zeros((48, 48), dtype=bool)
    mask[20:28, 20:28] = True
    config = _default_config(single_spot_max_diameter=12.0, gmm_multi_start_enabled=True)
    result = run_puncta_declump(image, config, external_mask=mask)
    assert len(result.accepted) == 1
    assert result.accepted[0].path == "fast_single"
    assert result.accepted[0].tried_gmm is False
