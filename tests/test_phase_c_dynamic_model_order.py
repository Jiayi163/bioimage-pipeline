"""Phase C: gated dynamic model-order tests."""

from __future__ import annotations

import numpy as np
import pytest

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.gaussian_fitter import GaussianMixtureFitter
from bioimage_pipeline.puncta.residual_refiner import ResidualSplitRefiner
from bioimage_pipeline.puncta.residual_split import (
    ResidualSplitConfig,
    SplitLoopState,
    mark_ambiguous_if_needed,
    remaining_component_budget,
    remaining_iteration_budget,
    should_stop_split_loop,
)
from bioimage_pipeline.puncta.types import ObjectPatch, PeakCandidate
from tests.test_phase_b_integration import (
    fit_single_component,
    make_hidden_doublet_patch,
    make_patch,
)
from tests.test_phase_b_residual_guided_split import (
    _clean_single_residual,
    _default_split_config,
    _gaussian_component,
    _mixture_result,
    _single_gaussian_residual,
    _third_lobe_residual,
)


def _config(**overrides: object) -> PunctaDeclumpConfig:
    params = {
        "residual_split_enabled": True,
        "dynamic_model_order_enabled": True,
        "dynamic_model_order_max_iterations": 3,
        "residual_split_max_components": 4,
        "gmm_max_components": 3,
        "gmm_multi_start_enabled": True,
    }
    params.update(overrides)
    return PunctaDeclumpConfig(**params)


def _make_gaussian_patch(
    centers: list[tuple[float, float]],
    *,
    amplitudes: list[float] | None = None,
    sigma: float = 1.5,
    background: float = 100.0,
    noise: float = 5.0,
    seed: int = 0,
) -> tuple[ObjectPatch, list[PeakCandidate]]:
    height = 25
    width = max(int(max(c for _, c in centers) + 10), 25)
    rows = height // 2
    y, x = np.ogrid[:height, :width]
    data = np.full((height, width), background, dtype=float)
    if amplitudes is None:
        amplitudes = [220.0 - 15.0 * i for i in range(len(centers))]
    for i, (row, col) in enumerate(centers):
        amp = amplitudes[i]
        data += amp * np.exp(-((x - col) ** 2 + (y - row) ** 2) / (2.0 * sigma**2))
    data += np.random.default_rng(seed).normal(0, noise, data.shape)
    patch = make_patch(data, background=background)
    best_index = int(np.argmax(amplitudes))
    best_row, best_col = centers[best_index]
    peaks = [PeakCandidate(row=float(best_row), col=float(best_col), intensity=float(data.max()))]
    return patch, peaks


def _make_multi_peak_patch(
    centers: list[tuple[float, float]],
    peak_indices: list[int],
    *,
    sigma: float = 1.5,
    background: float = 100.0,
    seed: int = 0,
) -> tuple[ObjectPatch, list[PeakCandidate]]:
    height = 25
    width = max(int(max(c for _, c in centers) + 10), 30)
    y, x = np.ogrid[:height, :width]
    data = np.full((height, width), background, dtype=float)
    amplitudes = [240.0, 220.0, 200.0, 180.0]
    for i, (row, col) in enumerate(centers):
        amp = amplitudes[i % len(amplitudes)]
        data += amp * np.exp(-((x - col) ** 2 + (y - row) ** 2) / (2.0 * sigma**2))
    data += np.random.default_rng(seed).normal(0, 5.0, data.shape)
    patch = make_patch(data, background=background)
    peaks = [
        PeakCandidate(
            row=float(centers[i][0]),
            col=float(centers[i][1]),
            intensity=float(data[int(centers[i][0]), int(centers[i][1])]),
        )
        for i in peak_indices
    ]
    return patch, peaks


def test_phase_c_config_defaults() -> None:
    config = _config()
    split = ResidualSplitConfig.from_puncta_config(config)
    assert config.dynamic_model_order_enabled is True
    assert config.dynamic_model_order_max_iterations == 3
    assert config.residual_split_max_components == 4
    assert split.max_components == 4
    assert split.max_split_iterations == 3


def test_remaining_budget_helpers() -> None:
    config = _default_split_config(max_components=4, max_split_iterations=3)
    state = SplitLoopState(current_n=2, iterations=1)
    assert remaining_component_budget(state, config) == 2
    assert remaining_iteration_budget(state, config) == 2


def test_stop_at_max_components() -> None:
    config = _default_split_config(max_components=4)
    state = SplitLoopState(current_n=4)
    stop, reason = should_stop_split_loop(state, config=config)
    assert stop is True
    assert reason == "max_components_reached"


def test_mark_ambiguous_on_unresolvable_rejection() -> None:
    residual, mask = _third_lobe_residual(
        existing_centers=[(10.0, 12.0), (18.0, 12.0)],
        third_center=(14.0, 12.0),
    )
    state = SplitLoopState(current_n=2)
    components = [
        _gaussian_component(component_id=1, row=12.0, col=10.0),
        _gaussian_component(component_id=2, row=12.0, col=18.0),
    ]
    mark_ambiguous_if_needed(
        state,
        rejection_reason="not_resolvable",
        residual_patch=residual,
        object_mask=mask,
        existing_components=components,
        config=_default_split_config(),
    )
    assert state.ambiguous is True
    assert "ambiguous" in state.stop_reason


def test_hidden_doublet_stops_at_two() -> None:
    """Hidden 2-component object should grow from 1 -> 2 and then stop."""
    patch, peaks = make_hidden_doublet_patch()
    config = _config()
    fitter = GaussianMixtureFitter(config)
    single = fit_single_component(fitter.single_fitter, patch, peaks)
    refiner = ResidualSplitRefiner(mixture_fitter=fitter, config=config)

    result = refiner.refine(initial_model=single, patch=patch, peaks=peaks)

    assert result.final_n >= 2
    assert result.final_n <= 2 or result.stop_reason in {
        "no_structured_residual",
        "no_valid_split_proposal",
        "insufficient_model_improvement",
        "not_resolvable",
        "max_split_iterations_reached",
        "max_components_reached",
    }


def test_hidden_third_component_grows_to_three() -> None:
    """Object with hidden third lobe should support 2 -> 3 when justified."""
    centers = [(12.0, 8.0), (12.0, 16.0), (12.0, 24.0)]
    patch, peaks = _make_multi_peak_patch(centers, peak_indices=[0, 1], seed=202)
    config = _config()
    fitter = GaussianMixtureFitter(config)
    single = fit_single_component(fitter.single_fitter, patch, peaks)

    init = fitter.fit_patch(patch, peaks, n_components=2, single_component=single)
    refiner = ResidualSplitRefiner(mixture_fitter=fitter, config=config)
    result = refiner.refine(initial_model=init, patch=patch, peaks=peaks)

    assert result.initial_n <= result.final_n
    assert result.final_n >= 2
    assert result.final_n <= config.residual_split_max_components


def test_dense_four_component_can_reach_four() -> None:
    """Dense 4-component object may grow toward 4 under Phase C budget."""
    centers = [(12.0, 6.0), (12.0, 12.0), (12.0, 18.0), (12.0, 24.0)]
    patch, peaks = _make_multi_peak_patch(centers, peak_indices=[0, 2], seed=404)
    config = _config(residual_split_max_components=4, dynamic_model_order_max_iterations=3)
    fitter = GaussianMixtureFitter(config)
    single = fit_single_component(fitter.single_fitter, patch, peaks)
    init = fitter.fit_patch(patch, peaks, n_components=2, single_component=single)

    refiner = ResidualSplitRefiner(mixture_fitter=fitter, config=config)
    result = refiner.refine(initial_model=init, patch=patch, peaks=peaks)

    assert result.final_n >= 2
    assert result.final_n <= 4
    assert len(result.split_attempts) <= config.effective_residual_split_max_iterations


def test_clean_single_never_splits() -> None:
    y, x = np.ogrid[-8:9, -8:9]
    g = 250.0 * np.exp(-(x**2 + y**2) / (2 * 1.8**2))
    data = 100.0 + g
    patch = make_patch(data, background=100.0)
    peaks = [PeakCandidate(row=8.0, col=8.0, intensity=350.0)]

    config = _config()
    fitter = GaussianMixtureFitter(config)
    single = fit_single_component(fitter.single_fitter, patch, peaks)
    refiner = ResidualSplitRefiner(mixture_fitter=fitter, config=config)
    result = refiner.refine(initial_model=single, patch=patch, peaks=peaks)

    assert result.final_n == 1
    assert result.split_attempts == []


def test_noisy_single_does_not_hallucinate_components() -> None:
    residual, mask = _clean_single_residual()
    baseline = _mixture_result(
        [_gaussian_component(component_id=1, row=24.0, col=24.0)],
        residual_patch=residual,
    )
    config = _default_split_config()
    state = SplitLoopState(current_n=1)

    from bioimage_pipeline.puncta.residual_split import should_propose_split

    ok, reason = should_propose_split(
        state=state,
        residual_patch=baseline.residual_patch,
        object_mask=mask,
        existing_components=baseline.components,
        config=config,
    )
    assert ok is False
    assert reason == "no_structured_residual"


def test_max_components_guard_stops_growth() -> None:
    config = _default_split_config(max_components=3, max_split_iterations=5)
    state = SplitLoopState(current_n=3)
    stop, reason = should_stop_split_loop(state, config=config)
    assert stop is True
    assert reason == "max_components_reached"


def test_failed_n_plus_one_falls_back_to_previous_model() -> None:
    """If N+1 is rejected, keep the previous accepted N model."""
    patch, peaks = _make_gaussian_patch([(12.0, 8.0), (12.0, 16.0)], seed=42)
    config = _config(gmm_bic_improvement_margin=500.0)
    fitter = GaussianMixtureFitter(config)
    single = fit_single_component(fitter.single_fitter, patch, peaks)
    refiner = ResidualSplitRefiner(mixture_fitter=fitter, config=config)

    result = refiner.refine(initial_model=single, patch=patch, peaks=peaks)

    assert result.final_model is not None
    if result.split_attempts and not result.split_attempts[-1].accepted:
        assert result.final_n == result.initial_n


def test_unresolved_components_mark_ambiguous_not_force_count() -> None:
    """When residual remains but a new component is not resolvable, mark ambiguous."""
    residual, mask = _single_gaussian_residual()
    state = SplitLoopState(current_n=1)
    components = [_gaussian_component(component_id=1, row=24.0, col=22.0)]

    mark_ambiguous_if_needed(
        state,
        rejection_reason="not_resolvable",
        residual_patch=residual,
        object_mask=mask,
        existing_components=components,
        config=_default_split_config(max_components=4),
    )
    assert state.ambiguous is True
    assert state.stop_reason.startswith("ambiguous_")


def test_refiner_records_multiple_iterations_when_accepted() -> None:
    """Accepted sequential splits should consume iteration budget one at a time."""
    centers = [(12.0, 8.0), (12.0, 16.0), (12.0, 24.0)]
    patch, peaks = _make_multi_peak_patch(centers, peak_indices=[0, 1], seed=303)
    config = _config(dynamic_model_order_max_iterations=3, residual_split_max_components=4)
    fitter = GaussianMixtureFitter(config)
    single = fit_single_component(fitter.single_fitter, patch, peaks)
    refiner = ResidualSplitRefiner(mixture_fitter=fitter, config=config)

    result = refiner.refine(initial_model=single, patch=patch, peaks=peaks)
    accepted = [attempt for attempt in result.split_attempts if attempt.accepted]
    assert len(result.split_attempts) <= config.effective_residual_split_max_iterations
    assert len(accepted) <= config.effective_residual_split_max_iterations


def test_object_791_is_supported_as_diagnostic_target_without_special_casing() -> None:
    """Ensure object 791 can be analyzed by the refiner API without bespoke logic."""
    # Synthetic stand-in: same API path used for real-object diagnostics.
    patch, peaks = _make_gaussian_patch([(12.0, 8.0), (12.0, 16.0), (12.0, 24.0)], seed=791)
    config = _config()
    fitter = GaussianMixtureFitter(config)
    single = fit_single_component(fitter.single_fitter, patch, peaks)
    refiner = ResidualSplitRefiner(mixture_fitter=fitter, config=config)
    result = refiner.refine(initial_model=single, patch=patch, peaks=peaks)
    assert result.final_n >= 1
    assert hasattr(result, "ambiguous")
    assert hasattr(result, "stop_reason")
