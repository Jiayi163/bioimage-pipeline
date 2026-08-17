"""Tests for Phase B: residual-guided split specification (no production wiring)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.residual_split import (
    ResidualSplitConfig,
    SplitLoopState,
    SplitProposal,
    evaluate_split_acceptance,
    find_structured_residual_peaks,
    is_positive_residual_structured,
    propose_n_plus_one_split,
    should_propose_split,
    should_stop_split_loop,
)
from bioimage_pipeline.puncta.types import GaussianComponent, MixtureFitResult
from bioimage_pipeline.puncta.validation.gmm_probe import make_clean_doublet_patch


def _default_split_config(**overrides: object) -> ResidualSplitConfig:
    params = {
        "min_peak_fraction_of_max": 0.35,
        "min_lobe_area_px": 4,
        "min_prominence_fraction": 0.15,
        "min_positive_mass_fraction": 0.08,
        "exclusion_radius_px": 1.5,
        "min_resolvability_sigma_units": 0.75,
        "bic_improvement_margin": 2.0,
        "min_residual_improvement_fraction": 0.05,
        "min_sigma": 0.5,
        "max_sigma": 4.0,
        "min_amplitude": 10.0,
        "max_components": 3,
        "max_split_iterations": 2,
    }
    params.update(overrides)
    return ResidualSplitConfig(**params)


def _gaussian_component(
    *,
    component_id: int,
    row: float,
    col: float,
    amplitude: float = 800.0,
    sigma: float = 2.2,
) -> GaussianComponent:
    return GaussianComponent(
        component_id=component_id,
        initial_row=row,
        initial_col=col,
        fitted_row=row,
        fitted_col=col,
        sigma_row=sigma,
        sigma_col=sigma,
        amplitude=amplitude,
        background=40.0,
        residual_rmse=15.0,
        residual_relative=0.12,
        r_squared=0.85,
        model_score=100.0,
        n_components_in_model=1,
        fit_succeeded=True,
    )


def _mixture_result(
    components: list[GaussianComponent],
    *,
    residual_rmse: float = 12.0,
    bic: float = 400.0,
    residual_patch: np.ndarray | None = None,
    predicted_patch: np.ndarray | None = None,
) -> MixtureFitResult:
    return MixtureFitResult(
        components=components,
        n_components=len(components),
        background=40.0,
        residual_rmse=residual_rmse,
        r_squared=0.90,
        aic=bic + 10.0,
        bic=bic,
        model_score=bic,
        fit_succeeded=True,
        residual_patch=residual_patch,
        predicted_patch=predicted_patch,
    )


def _single_gaussian_residual(
    shape: tuple[int, int] = (48, 48),
    center: tuple[float, float] = (24.0, 24.0),
    sigma: float = 2.2,
    amplitude: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate residual after 1-Gaussian fit on a doublet: one lobe remains."""
    height, width = shape
    rows, cols = np.indices((height, width))
    row_c, col_c = center
    # Second lobe centered offset to the right (hidden from detector)
    second_col = col_c + 4.0
    residual = amplitude * np.exp(
        -((rows - row_c) ** 2 + (cols - second_col) ** 2) / (2.0 * sigma**2)
    )
    mask = np.ones(shape, dtype=bool)
    return residual, mask


def _clean_single_residual(shape: tuple[int, int] = (48, 48)) -> tuple[np.ndarray, np.ndarray]:
    """Unstructured noise-like residual for a clean single."""
    rng = np.random.default_rng(42)
    residual = rng.normal(0, 3.0, size=shape)
    residual = np.clip(residual, 0, None)
    mask = np.ones(shape, dtype=bool)
    return residual, mask


def _third_lobe_residual(
    shape: tuple[int, int] = (48, 48),
    *,
    existing_centers: list[tuple[float, float]],
    third_center: tuple[float, float] = (24.0, 34.0),
    amplitude: float = 180.0,
    sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Structured third lobe after a 2-component fit."""
    height, width = shape
    rows, cols = np.indices((height, width))
    row_c, col_c = third_center
    residual = amplitude * np.exp(
        -((rows - row_c) ** 2 + (cols - col_c) ** 2) / (2.0 * sigma**2)
    )
    mask = np.ones(shape, dtype=bool)
    return residual, mask


# ---------------------------------------------------------------------------
# Required test 1: hidden doublet — detector sees 1 peak, residual proposes N=2
# ---------------------------------------------------------------------------


def test_hidden_doublet_proposes_n_plus_one_split() -> None:
    """1-component fit on overlapping doublet: residual split should propose N=2."""
    residual, mask = _single_gaussian_residual()
    config = _default_split_config()

    component = _gaussian_component(component_id=1, row=24.0, col=22.0)
    proposal = propose_n_plus_one_split(
        current_n=1,
        residual_patch=residual,
        object_mask=mask,
        existing_components=[component],
        background_level=40.0,
        config=config,
    )

    assert proposal is not None
    assert proposal.current_n == 1
    assert proposal.proposed_n == 2
    # Second true center is near col=28 for default helper
    assert proposal.new_center_col > 26.0
    assert abs(proposal.new_center_row - 24.0) < 2.0


def test_hidden_doublet_should_propose_split() -> None:
    residual, mask = _single_gaussian_residual()
    config = _default_split_config()
    component = _gaussian_component(component_id=1, row=24.0, col=22.0)
    state = SplitLoopState(current_n=1)

    ok, reason = should_propose_split(
        state=state,
        residual_patch=residual,
        object_mask=mask,
        existing_components=[component],
        config=config,
    )

    assert ok is True
    assert reason == "structured_residual_present"


# ---------------------------------------------------------------------------
# Required test 2: clean single — no split
# ---------------------------------------------------------------------------


def test_clean_single_gaussian_no_split_proposed() -> None:
    residual, mask = _clean_single_residual()
    config = _default_split_config()
    component = _gaussian_component(component_id=1, row=24.0, col=24.0)

    proposal = propose_n_plus_one_split(
        current_n=1,
        residual_patch=residual,
        object_mask=mask,
        existing_components=[component],
        config=config,
    )

    assert proposal is None
    assert not is_positive_residual_structured(residual, mask, config=config)


# ---------------------------------------------------------------------------
# Required test 3: 2-component model with structured third residual → N=3
# ---------------------------------------------------------------------------


def test_two_component_model_proposes_third_component() -> None:
    existing = [
        _gaussian_component(component_id=1, row=24.0, col=20.0),
        _gaussian_component(component_id=2, row=24.0, col=28.0),
    ]
    residual, mask = _third_lobe_residual(
        existing_centers=[(20.0, 24.0), (28.0, 24.0)],
        third_center=(24.0, 34.0),
    )
    config = _default_split_config()

    proposal = propose_n_plus_one_split(
        current_n=2,
        residual_patch=residual,
        object_mask=mask,
        existing_components=existing,
        config=config,
    )

    assert proposal is not None
    assert proposal.current_n == 2
    assert proposal.proposed_n == 3
    assert proposal.new_center_col > 30.0
    assert abs(proposal.new_center_row - 24.0) < 2.0


# ---------------------------------------------------------------------------
# Required test 4: noisy unstructured residual — no split
# ---------------------------------------------------------------------------


def test_noisy_unstructured_residual_no_split() -> None:
    rng = np.random.default_rng(99)
    residual = np.abs(rng.normal(0, 8.0, size=(48, 48)))
    mask = np.ones((48, 48), dtype=bool)
    config = _default_split_config(min_peak_fraction_of_max=0.40)

    peaks = find_structured_residual_peaks(
        residual,
        mask,
        [(24.0, 24.0)],
        config=config,
    )
    assert len(peaks) == 0


# ---------------------------------------------------------------------------
# Required test 5: proposed component too close — reject / exclude
# ---------------------------------------------------------------------------


def test_residual_peak_near_existing_center_excluded() -> None:
    """Residual peak within exclusion radius of existing center must not be proposed."""
    height, width = 48, 48
    rows, cols = np.indices((height, width))
    # Lobe very close to existing center at (24, 24)
    residual = 200.0 * np.exp(-((rows - 24.0) ** 2 + (cols - 24.5) ** 2) / (2 * 1.5**2))
    mask = np.ones((height, width), dtype=bool)
    config = _default_split_config(exclusion_radius_px=1.5)

    existing = [_gaussian_component(component_id=1, row=24.0, col=24.0)]
    peaks = find_structured_residual_peaks(
        residual,
        mask,
        [(24.0, 24.0)],
        config=config,
    )
    assert len(peaks) == 0

    proposal = propose_n_plus_one_split(
        current_n=1,
        residual_patch=residual,
        object_mask=mask,
        existing_components=existing,
        config=config,
    )
    assert proposal is None


# ---------------------------------------------------------------------------
# Required test 6: N+1 improves residual but violates physical validity — reject
# ---------------------------------------------------------------------------


def test_split_rejected_when_physically_invalid_despite_residual_improvement() -> None:
    baseline = _mixture_result(
        [_gaussian_component(component_id=1, row=24.0, col=22.0)],
        residual_rmse=20.0,
        bic=500.0,
    )
    # Improved residual/BIC but invalid sigma on new component
    invalid_new = _gaussian_component(
        component_id=2,
        row=24.0,
        col=28.0,
        sigma=0.1,
        amplitude=800.0,
    )
    candidate = _mixture_result(
        [
            _gaussian_component(component_id=1, row=24.0, col=22.0),
            invalid_new,
        ],
        residual_rmse=10.0,
        bic=450.0,
    )
    proposal = SplitProposal(
        current_n=1,
        proposed_n=2,
        new_center_row=24.0,
        new_center_col=28.0,
        seed_intensity=800.0,
        residual_peak=find_structured_residual_peaks(
            *_single_gaussian_residual(),
            [(22.0, 24.0)],
            config=_default_split_config(),
        )[0],
    )

    result = evaluate_split_acceptance(
        baseline=baseline,
        candidate=candidate,
        proposal=proposal,
        config=_default_split_config(),
    )

    assert result.accepted is False
    assert result.checks["residual_improved"] is True
    assert result.checks["bic_improved"] is True
    assert result.checks["sigma_valid"] is False
    assert result.reason == "invalid_sigma"


def test_split_rejected_when_not_resolvable() -> None:
    baseline = _mixture_result(
        [
            _gaussian_component(component_id=1, row=24.0, col=22.0),
        ],
        residual_rmse=20.0,
        bic=500.0,
    )
    too_close = _gaussian_component(component_id=2, row=24.0, col=22.8, sigma=2.2)
    candidate = _mixture_result(
        [
            _gaussian_component(component_id=1, row=24.0, col=22.0),
            too_close,
        ],
        residual_rmse=12.0,
        bic=460.0,
    )
    residual, mask = _single_gaussian_residual()
    peaks = find_structured_residual_peaks(
        residual,
        mask,
        [(22.0, 24.0)],
        config=_default_split_config(),
    )
    proposal = SplitProposal(
        current_n=1,
        proposed_n=2,
        new_center_row=peaks[0].row,
        new_center_col=peaks[0].col,
        seed_intensity=800.0,
        residual_peak=peaks[0],
    )

    result = evaluate_split_acceptance(
        baseline=baseline,
        candidate=candidate,
        proposal=proposal,
        config=_default_split_config(min_resolvability_sigma_units=1.5),
    )

    assert result.accepted is False
    assert result.checks["resolvable"] is False


# ---------------------------------------------------------------------------
# Required test 7: max_components respected
# ---------------------------------------------------------------------------


def test_max_components_stops_split_proposal() -> None:
    config = _default_split_config(max_components=3)
    state = SplitLoopState(current_n=3, iterations=0)

    stop, reason = should_stop_split_loop(state, config=config)
    assert stop is True
    assert reason == "max_components_reached"

    residual, mask = _third_lobe_residual(existing_centers=[(20.0, 24.0), (28.0, 24.0)])
    existing = [
        _gaussian_component(component_id=1, row=24.0, col=18.0),
        _gaussian_component(component_id=2, row=24.0, col=24.0),
        _gaussian_component(component_id=3, row=24.0, col=30.0),
    ]
    proposal = propose_n_plus_one_split(
        current_n=3,
        residual_patch=residual,
        object_mask=mask,
        existing_components=existing,
        config=config,
    )
    assert proposal is None


def test_max_split_iterations_guardrail() -> None:
    config = _default_split_config(max_split_iterations=2)
    state = SplitLoopState(current_n=2, iterations=2)

    stop, reason = should_stop_split_loop(state, config=config)
    assert stop is True
    assert reason == "max_split_iterations_reached"


# ---------------------------------------------------------------------------
# Required test 8: deterministic / repeatable split proposal
# ---------------------------------------------------------------------------


def test_split_proposal_is_deterministic() -> None:
    residual, mask = _single_gaussian_residual()
    config = _default_split_config()
    existing = [_gaussian_component(component_id=1, row=24.0, col=22.0)]

    p1 = propose_n_plus_one_split(
        current_n=1,
        residual_patch=residual,
        object_mask=mask,
        existing_components=existing,
        config=config,
    )
    p2 = propose_n_plus_one_split(
        current_n=1,
        residual_patch=residual,
        object_mask=mask,
        existing_components=existing,
        config=config,
    )

    assert p1 is not None and p2 is not None
    assert p1.new_center_row == p2.new_center_row
    assert p1.new_center_col == p2.new_center_col
    assert p1.seed_intensity == p2.seed_intensity
    assert p1.residual_peak.label_id == p2.residual_peak.label_id


# ---------------------------------------------------------------------------
# Integration-oriented: config factory from PunctaDeclumpConfig
# ---------------------------------------------------------------------------


def test_residual_split_config_from_puncta_config() -> None:
    puncta = PunctaDeclumpConfig(
        gmm_bic_improvement_margin=3.0,
        gmm_acceptance_min_separation=2.0,
        gmm_max_components=4,
    )
    split_cfg = ResidualSplitConfig.from_puncta_config(puncta)
    assert split_cfg.bic_improvement_margin == 3.0
    assert split_cfg.exclusion_radius_px == 2.0
    assert split_cfg.max_components == 4


# ---------------------------------------------------------------------------
# Synthetic patch integration helper (uses probe, not production pipeline)
# ---------------------------------------------------------------------------


def test_hidden_doublet_patch_residual_is_structured() -> None:
    """End-to-end helper: doublet patch minus single-Gaussian prediction leaves structure."""
    patch, _obj, true_centers = make_clean_doublet_patch(separation_px=4.0, sigma=2.2)
    rows, cols = np.indices(patch.corrected.shape)
    # Approximate 1-Gaussian prediction centered on first peak
    pred = 1800.0 * np.exp(
        -((rows - true_centers[0][1]) ** 2 + (cols - true_centers[0][0]) ** 2) / (2 * 2.2**2)
    )
    residual = np.clip(patch.corrected - pred, 0, None)
    config = _default_split_config()

    assert is_positive_residual_structured(residual, patch.object_mask, config=config)

    proposal = propose_n_plus_one_split(
        current_n=1,
        residual_patch=residual,
        object_mask=patch.object_mask,
        existing_components=[
            _gaussian_component(
                component_id=1,
                row=true_centers[0][1],
                col=true_centers[0][0],
            )
        ],
        config=config,
    )
    assert proposal is not None
    assert proposal.proposed_n == 2
    # Proposed center should be nearer second true center than first
    dist_to_second = math.hypot(
        proposal.new_center_col - true_centers[1][0],
        proposal.new_center_row - true_centers[1][1],
    )
    dist_to_first = math.hypot(
        proposal.new_center_col - true_centers[0][0],
        proposal.new_center_row - true_centers[0][1],
    )
    assert dist_to_second < dist_to_first
