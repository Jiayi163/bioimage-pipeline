"""Tests for conservative Phase C fallback after Phase B."""

from __future__ import annotations

import numpy as np
import pytest

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.gaussian_fitter import GaussianModelSelector
from bioimage_pipeline.puncta.object_processor import ObjectProcessor
from bioimage_pipeline.puncta.phase_c_fallback import evaluate_phase_c_fallback
from bioimage_pipeline.puncta.residual_refiner import ResidualSplitRefiner
from bioimage_pipeline.puncta.residual_split import ResidualSplitConfig
from bioimage_pipeline.puncta.types import GaussianComponent, MixtureFitResult, ObjectInfo, ObjectPatch, PeakCandidate
from tests.test_phase_b_integration import fit_single_component, make_hidden_doublet_patch, make_patch
from tests.test_phase_c_dynamic_model_order import _make_gaussian_patch, _make_multi_peak_patch


def _object_info(**overrides: object) -> ObjectInfo:
    params = {
        "label": 1,
        "area": 20.0,
        "equivalent_diameter": 5.0,
        "bbox": (0, 0, 20, 20),
        "centroid": (10.0, 10.0),
        "brightest_row": 10.0,
        "brightest_col": 10.0,
        "brightest_intensity": 300.0,
        "eccentricity": 0.2,
        "solidity": 0.9,
        "major_axis_length": 6.0,
        "minor_axis_length": 5.0,
        "elongation": 1.2,
    }
    params.update(overrides)
    return ObjectInfo(**params)


def _single_component(residual_patch: np.ndarray | None = None, **kwargs: object) -> GaussianComponent:
    defaults = {
        "component_id": 1,
        "initial_row": 12.0,
        "initial_col": 12.0,
        "fitted_row": 12.0,
        "fitted_col": 12.0,
        "amplitude": 200.0,
        "sigma_row": 1.5,
        "sigma_col": 1.5,
        "background": 100.0,
        "residual_rmse": 20.0,
        "residual_relative": 0.1,
        "r_squared": 0.8,
        "model_score": 0.8,
        "n_components_in_model": 1,
        "fit_succeeded": True,
        "residual_patch": residual_patch,
    }
    defaults.update(kwargs)
    return GaussianComponent(**defaults)


def _mixture(n: int = 2, *, residual_patch: np.ndarray | None = None) -> MixtureFitResult:
    components = [
        GaussianComponent(
            component_id=i + 1,
            initial_row=12.0 + i,
            initial_col=12.0 + i * 2,
            fitted_row=12.0 + i,
            fitted_col=12.0 + i * 2,
            amplitude=200.0 - 20 * i,
            sigma_row=1.5,
            sigma_col=1.5,
            background=100.0,
            residual_rmse=80.0,
            residual_relative=0.4,
            r_squared=0.3,
            model_score=0.3,
            n_components_in_model=n,
            fit_succeeded=True,
        )
        for i in range(n)
    ]
    return MixtureFitResult(
        n_components=n,
        components=components,
        background=100.0,
        r_squared=0.3,
        residual_rmse=80.0,
        bic=100.0,
        aic=90.0,
        model_score=0.3,
        fit_succeeded=True,
        residual_patch=residual_patch,
    )


def _structured_residual_patch(height: int = 25, width: int = 30) -> tuple[np.ndarray, np.ndarray]:
    residual = np.zeros((height, width), dtype=float)
    mask = np.zeros((height, width), dtype=bool)
    mask[5:20, 5:25] = True
    residual[10, 8] = 1.0
    residual[10, 22] = 0.9
    residual[14, 15] = 0.7
    return residual, mask


def test_dense_six_peak_unresolved_triggers_phase_c_fallback() -> None:
    config = PunctaDeclumpConfig(dynamic_model_order_fallback_enabled=True)
    obj = _object_info(area=166.0, equivalent_diameter=14.5)
    residual, mask = _structured_residual_patch()
    decision = evaluate_phase_c_fallback(
        config=config,
        obj=obj,
        single=_single_component(
            residual_patch=residual,
            r_squared=0.48,
            residual_relative=0.48,
        ),
        selected=_mixture(residual_patch=residual),
        patch=ObjectPatch(
            object_id=1,
            row_offset=0,
            col_offset=0,
            corrected=np.ones((25, 30)),
            object_mask=mask,
            background_level=100.0,
            global_bbox=(0, 0, 25, 30),
            raw=np.ones((25, 30)) * 200,
        ),
        n_filtered_peaks=6,
        n_accepted=0,
        under_split_suspect=True,
    )
    assert decision.trigger is True
    assert "multiplicity_gap=0/6" in decision.reason


def test_phase_b_success_does_not_trigger_fallback() -> None:
    config = PunctaDeclumpConfig(dynamic_model_order_fallback_enabled=True)
    obj = _object_info()
    mask = np.ones((20, 20), dtype=bool)
    decision = evaluate_phase_c_fallback(
        config=config,
        obj=obj,
        single=_single_component(r_squared=0.85, residual_relative=0.08),
        selected=_mixture(n=2),
        patch=ObjectPatch(
            object_id=1,
            row_offset=0,
            col_offset=0,
            corrected=np.ones((20, 20)),
            object_mask=mask,
            background_level=100.0,
            global_bbox=(0, 0, 20, 20),
            raw=np.ones((20, 20)),
        ),
        n_filtered_peaks=2,
        n_accepted=2,
        under_split_suspect=False,
    )
    assert decision.trigger is False
    assert decision.reason == "multiplicity_resolved"


def test_clean_single_never_triggers_fallback() -> None:
    config = PunctaDeclumpConfig(dynamic_model_order_fallback_enabled=True)
    obj = _object_info(area=12.0, equivalent_diameter=4.0)
    mask = np.zeros((17, 17), dtype=bool)
    mask[6:11, 6:11] = True
    decision = evaluate_phase_c_fallback(
        config=config,
        obj=obj,
        single=_single_component(r_squared=0.95, residual_relative=0.05),
        selected=_single_component(r_squared=0.95, residual_relative=0.05),
        patch=ObjectPatch(
            object_id=1,
            row_offset=0,
            col_offset=0,
            corrected=np.ones((17, 17)),
            object_mask=mask,
            background_level=100.0,
            global_bbox=(0, 0, 17, 17),
            raw=np.ones((17, 17)),
        ),
        n_filtered_peaks=1,
        n_accepted=1,
        under_split_suspect=False,
    )
    assert decision.trigger is False


def test_noisy_multi_peak_without_support_does_not_trigger_fallback() -> None:
    """Unresolved count alone is insufficient without under-split + evidence."""
    config = PunctaDeclumpConfig(dynamic_model_order_fallback_enabled=True)
    obj = _object_info(area=30.0, equivalent_diameter=6.0)
    mask = np.ones((20, 20), dtype=bool)
    flat_residual = np.random.default_rng(0).normal(0, 0.01, (20, 20))
    decision = evaluate_phase_c_fallback(
        config=config,
        obj=obj,
        single=_single_component(residual_patch=flat_residual, r_squared=0.9, residual_relative=0.05),
        selected=_mixture(residual_patch=flat_residual),
        patch=ObjectPatch(
            object_id=1,
            row_offset=0,
            col_offset=0,
            corrected=np.ones((20, 20)),
            object_mask=mask,
            background_level=100.0,
            global_bbox=(0, 0, 20, 20),
            raw=np.ones((20, 20)),
        ),
        n_filtered_peaks=4,
        n_accepted=1,
        under_split_suspect=False,
    )
    assert decision.trigger is False
    assert decision.reason == "not_under_split_suspect"


def test_fallback_uses_phase_c_limits_and_can_grow_beyond_phase_b() -> None:
    centers = [(12.0, 6.0), (12.0, 12.0), (12.0, 18.0), (12.0, 24.0)]
    patch, peaks = _make_multi_peak_patch(centers, peak_indices=[0, 2], seed=404)
    config = PunctaDeclumpConfig(
        residual_split_enabled=True,
        dynamic_model_order_fallback_enabled=True,
        dynamic_model_order_enabled=False,
    )
    fitter = GaussianModelSelector(config)
    single = fit_single_component(fitter.single_fitter, patch, peaks)
    comparison = fitter.select_balanced_model(
        patch,
        peaks,
        single_component=single,
        n_filtered_peaks=len(peaks),
        n_raw_peaks=len(peaks),
        obj=_object_info(area=80.0, equivalent_diameter=12.0),
    )
    split_b = ResidualSplitConfig.from_puncta_config(config)
    assert split_b.max_split_iterations == 1

    fallback = fitter.apply_phase_c_fallback_refinement(
        comparison,
        patch,
        peaks,
        trigger_reason="test",
    )
    split_c = ResidualSplitConfig.for_phase_c_fallback(config)
    assert split_c.max_split_iterations == 3
    assert split_c.max_components == 4
    assert "phase_c_fallback=" in fallback.selection_reason

    refiner = ResidualSplitRefiner(
        mixture_fitter=fitter.mixture_fitter,
        config=config,
        split_config=split_c,
    )
    init = fitter.mixture_fitter.fit_patch(patch, peaks, n_components=2, single_component=single)
    result = refiner.refine(initial_model=init, patch=patch, peaks=peaks)
    assert result.final_n <= 4
    assert len(result.split_attempts) <= 3


def test_fallback_disabled_when_global_phase_c_enabled() -> None:
    config = PunctaDeclumpConfig(
        dynamic_model_order_enabled=True,
        dynamic_model_order_fallback_enabled=True,
    )
    decision = evaluate_phase_c_fallback(
        config=config,
        obj=_object_info(area=166.0, equivalent_diameter=14.5),
        single=_single_component(r_squared=0.4, residual_relative=0.5),
        selected=_mixture(),
        patch=ObjectPatch(
            object_id=1,
            row_offset=0,
            col_offset=0,
            corrected=np.ones((10, 10)),
            object_mask=np.ones((10, 10), dtype=bool),
            background_level=100.0,
            global_bbox=(0, 0, 10, 10),
            raw=np.ones((10, 10)),
        ),
        n_filtered_peaks=6,
        n_accepted=0,
        under_split_suspect=True,
    )
    assert decision.trigger is False
    assert decision.reason == "phase_c_global_enabled"


def test_hidden_doublet_integration_does_not_trigger_fallback() -> None:
    patch, peaks = make_hidden_doublet_patch()
    obj = _object_info(area=20.0, equivalent_diameter=5.0)
    processor = ObjectProcessor(PunctaDeclumpConfig(dynamic_model_order_fallback_enabled=True))
    image = patch.raw
    mask = np.zeros(image.shape, dtype=bool)
    mask[patch.row_offset : patch.row_offset + patch.corrected.shape[0], patch.col_offset : patch.col_offset + patch.corrected.shape[1]] = (
        patch.object_mask
    )
    result = processor.process_suspicious(
        image,
        mask,
        obj,
        assigned_peaks=peaks,
        candidate_id_start=1,
    )
    accepted = [candidate for candidate in result.candidates if candidate.accepted and candidate.fit_status == "fit_ok"]
    assert len(accepted) >= 1
    assert "phase_c_fallback=" not in (result.debug.model_selection_reason or "")


def test_unresolved_can_mark_ambiguous_at_phase_c_cap() -> None:
    """Phase C fallback preserves ambiguous stop when growth cannot continue."""
    from bioimage_pipeline.puncta.residual_split import SplitLoopState, mark_ambiguous_if_needed
    from tests.test_phase_b_residual_guided_split import _third_lobe_residual

    residual, obj_mask = _third_lobe_residual(
        existing_centers=[(10.0, 12.0), (18.0, 12.0), (14.0, 12.0)],
        third_center=(14.0, 12.0),
    )
    state = SplitLoopState(current_n=4)
    mark_ambiguous_if_needed(
        state,
        rejection_reason="not_resolvable",
        residual_patch=residual,
        object_mask=obj_mask,
        existing_components=_mixture(n=3).components,
        config=ResidualSplitConfig.for_phase_c_fallback(PunctaDeclumpConfig()),
    )
    assert state.ambiguous is True
    assert state.stop_reason.startswith("ambiguous_")
