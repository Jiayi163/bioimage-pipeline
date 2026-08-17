"""Production routing tests: Phase B default vs optional Phase C."""

from __future__ import annotations

import numpy as np

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.gaussian_fitter import GaussianMixtureFitter
from bioimage_pipeline.puncta.residual_refiner import ResidualSplitRefiner
from bioimage_pipeline.puncta.residual_split import ResidualSplitConfig
from bioimage_pipeline.puncta.types import PeakCandidate
from tests.test_phase_b_integration import fit_single_component, make_patch
from tests.test_phase_c_dynamic_model_order import _make_multi_peak_patch


def test_default_config_is_phase_b() -> None:
    config = PunctaDeclumpConfig()
    split = ResidualSplitConfig.from_puncta_config(config)

    assert config.residual_split_enabled is True
    assert config.dynamic_model_order_enabled is False
    assert config.residual_split_max_iterations == 1
    assert config.effective_residual_split_max_iterations == 1
    assert config.effective_residual_split_max_components == config.gmm_max_components + 1
    assert split.max_split_iterations == 1
    assert split.max_components == config.gmm_max_components + 1


def test_phase_c_disabled_by_default() -> None:
    config = PunctaDeclumpConfig()
    assert config.dynamic_model_order_enabled is False


def test_enabling_phase_c_allows_iterative_growth_to_k4() -> None:
    centers = [(12.0, 6.0), (12.0, 12.0), (12.0, 18.0), (12.0, 24.0)]
    patch, peaks = _make_multi_peak_patch(centers, peak_indices=[0, 2], seed=404)
    config = PunctaDeclumpConfig(
        residual_split_enabled=True,
        dynamic_model_order_enabled=True,
        dynamic_model_order_max_iterations=3,
        residual_split_max_components=4,
        gmm_max_components=3,
        gmm_multi_start_enabled=True,
    )
    split = ResidualSplitConfig.from_puncta_config(config)
    assert split.max_split_iterations == 3
    assert split.max_components == 4

    fitter = GaussianMixtureFitter(config)
    single = fit_single_component(fitter.single_fitter, patch, peaks)
    init = fitter.fit_patch(patch, peaks, n_components=2, single_component=single)
    refiner = ResidualSplitRefiner(mixture_fitter=fitter, config=config)
    result = refiner.refine(initial_model=init, patch=patch, peaks=peaks)

    assert result.final_n <= 4
    assert len(result.split_attempts) <= config.effective_residual_split_max_iterations


def test_disabling_phase_c_limits_residual_growth_to_one_step() -> None:
    centers = [(12.0, 8.0), (12.0, 16.0), (12.0, 24.0)]
    patch, peaks = _make_multi_peak_patch(centers, peak_indices=[0, 1], seed=303)
    config = PunctaDeclumpConfig(
        residual_split_enabled=True,
        dynamic_model_order_enabled=False,
        residual_split_max_iterations=1,
        gmm_max_components=3,
        gmm_multi_start_enabled=True,
    )
    split = ResidualSplitConfig.from_puncta_config(config)
    assert split.max_split_iterations == 1

    fitter = GaussianMixtureFitter(config)
    single = fit_single_component(fitter.single_fitter, patch, peaks)
    refiner = ResidualSplitRefiner(mixture_fitter=fitter, config=config)
    result = refiner.refine(initial_model=single, patch=patch, peaks=peaks)

    accepted = [attempt for attempt in result.split_attempts if attempt.accepted]
    assert len(accepted) <= 1
    assert len(result.split_attempts) <= 1


def test_phase_b_respects_custom_single_step_override() -> None:
    """Phase B honors ``residual_split_max_iterations`` when Phase C is off."""
    config = PunctaDeclumpConfig(
        residual_split_enabled=True,
        dynamic_model_order_enabled=False,
        residual_split_max_iterations=2,
    )
    split = ResidualSplitConfig.from_puncta_config(config)
    assert split.max_split_iterations == 2
    assert split.max_components == config.gmm_max_components + 1
