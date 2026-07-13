"""Direct GMM fitting tests independent of the full puncta pipeline."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.gaussian_fitter import GaussianModelSelector
from bioimage_pipeline.puncta.types import PeakCandidate
from bioimage_pipeline.puncta.validation.gmm_probe import (
    GmmDiagnosticProbe,
    InitStrategy,
    apply_filter_ablation,
    fit_mixture_with_init,
    generate_init_peak_sets,
    make_clean_doublet_patch,
)


def test_direct_clean_doublet_recovers_two_centers() -> None:
    patch, obj, true_centers = make_clean_doublet_patch(separation_px=4.0, sigma=2.2)
    peaks = [
        PeakCandidate(row=true_centers[0][1], col=true_centers[0][0], intensity=1800.0),
    ]
    init_sets = generate_init_peak_sets(peaks, patch, obj, n_components=2, single_component=None)
    init_peaks = init_sets[f"{InitStrategy.SYMMETRIC_X.value}_sep4"]

    config = PunctaDeclumpConfig()
    config.gmm_min_component_separation = 1.0
    fit, diag, _ = fit_mixture_with_init(
        patch,
        init_peaks,
        n_components=2,
        config=config,
        initialization_method="symmetric_x_sep4",
    )
    assert fit is not None
    assert diag is not None
    assert diag.converged
    assert fit.n_components == 2

    for true_x, true_y in true_centers:
        distances = [
            math.hypot(comp.fitted_col - true_x, comp.fitted_row - true_y)
            for comp in fit.components
        ]
        assert min(distances) <= 1.5


def test_direct_fit_progressive_failure_layers() -> None:
    patch, obj, true_centers = make_clean_doublet_patch(separation_px=4.0, sigma=2.2)
    base_peak = PeakCandidate(row=patch.corrected.shape[0] / 2, col=patch.corrected.shape[1] / 2, intensity=1800.0)
    peaks = [base_peak]
    config = PunctaDeclumpConfig()

    # Layer 1: clean, symmetric init, no filters
    init_sets = generate_init_peak_sets(peaks, patch, obj, n_components=2)
    init_peaks = init_sets[f"{InitStrategy.SYMMETRIC_X.value}_sep4"]
    clean_fit, _, _ = fit_mixture_with_init(
        patch,
        init_peaks,
        n_components=2,
        config=config,
        initialization_method="symmetric_x_sep4",
    )
    assert clean_fit is not None and clean_fit.n_components == 2

    # Layer 2: add noise — fit may degrade but should still attempt 2 components
    rng = np.random.default_rng(0)
    noisy = patch.corrected + rng.normal(0, 5.0, size=patch.corrected.shape)
    patch_noisy = patch
    patch_noisy.corrected = np.clip(noisy, 0.0, None)
    noisy_fit, _, _ = fit_mixture_with_init(
        patch_noisy,
        init_peaks,
        n_components=2,
        config=config,
        initialization_method="symmetric_x_sep4",
    )
    assert noisy_fit is not None

    # Layer 3: detector-based init with one peak (production-like)
    detector_init = init_sets[InitStrategy.DETECTOR_BASED.value]
    assert len(detector_init) == 2
    detector_fit, _, _ = fit_mixture_with_init(
        patch,
        detector_init,
        n_components=2,
        config=config,
        initialization_method=InitStrategy.DETECTOR_BASED.value,
    )
    assert detector_fit is not None

    # Layer 4: model selection may keep single despite good 2-comp fit
    selector = GaussianModelSelector(config)
    single = selector.single_fitter.fit_peak(patch, peaks[0], component_id=1, n_components_in_model=1)
    comparison = selector.select_balanced_model(
        patch,
        peaks,
        single_component=single,
        n_filtered_peaks=1,
        n_raw_peaks=1,
        obj=obj,
    )
    assert 2 in comparison.candidate_component_counts

    # Layer 5: full filters can reject second component even when fit succeeds
    records, rejections = apply_filter_ablation(
        obj,
        patch,
        clean_fit,
        peaks,
        config=config,
        filter_mode="full",
    )
    assert len(records) == 2


def test_probe_reports_single_peak_still_attempts_two_component_model() -> None:
    patch, obj, _ = make_clean_doublet_patch(separation_px=4.0, sigma=2.2)
    peaks = [PeakCandidate(row=24.0, col=24.0, intensity=1800.0)]
    probe = GmmDiagnosticProbe()
    report = probe.probe_patch(
        patch,
        obj,
        peaks,
        n_raw_peaks=1,
        n_filtered_peaks=1,
    )
    assert report.model_selection.balanced_model_attempted_n2 is True
    two_attempts = [a for a in report.model_attempts if a.n_components == 2 and a.attempted]
    assert two_attempts
    assert InitStrategy.DETECTOR_BASED.value in {a.initialization_method for a in two_attempts}
