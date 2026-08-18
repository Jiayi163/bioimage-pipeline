"""Tests for conservative local peak recovery on fast-path assignment misses."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from bioimage_pipeline.puncta.connected_objects import ConnectedObjectAnalyzer
from bioimage_pipeline.puncta.maxima_detector import MaximaDetector
from bioimage_pipeline.puncta.object_processor import ObjectProcessor
from bioimage_pipeline.puncta.object_router import RouteDecision
from bioimage_pipeline.puncta.peak_assignment import assign_peaks_to_objects
from bioimage_pipeline.puncta.pipeline import run_puncta_declump
from bioimage_pipeline.puncta.types import PeakDetectionResult
from tests.test_puncta_declump import default_config, make_binary_disk, make_gaussian_spot


def _route() -> RouteDecision:
    return RouteDecision(route="ordinary_single", reasons=())


def _labels_and_objects(image: np.ndarray, mask: np.ndarray):
    return ConnectedObjectAnalyzer().analyze(mask.astype(bool), image)


def test_zero_assigned_one_local_peak_uses_recovered_single_path() -> None:
    """Single recovered peak uses fast_single path with no Gaussian fitting."""
    image = make_gaussian_spot((48, 48), (24.0, 24.0), sigma=1.3, amplitude=500.0)
    mask = make_binary_disk((48, 48), (24.0, 24.0), radius=4.0)
    labels, objects = _labels_and_objects(image, mask)
    obj = objects[0]
    processor = ObjectProcessor(default_config())

    recovery = processor.recover_local_peaks(image, labels == obj.label, obj)
    assert recovery.success
    assert recovery.filtered_count == 1
    assert recovery.peak_source == "recovered_local_detector"

    result = processor.process_recovered_single(
        obj, recovery, candidate_id_start=1, route=_route()
    )
    # CRITICAL: no Gaussian fit, fast_single path
    assert result.path == "fast_single"
    assert result.candidates[0].peak_source == "recovered_local_detector"
    assert result.candidates[0].fit_status == "fit_ok"
    assert result.candidates[0].accepted
    assert result.debug.tried_gmm is False
    assert result.debug.local_peak_recovery_attempted is True
    assert result.debug.local_peak_recovery_success is True
    # No Gaussian fit metrics should be recorded
    assert result.debug.one_gaussian_r_squared is None
    assert result.debug.one_gaussian_residual_relative is None
    assert result.debug.one_gaussian_sigma is None
    assert abs(result.candidates[0].final_row - 24.0) < 2.0


def test_zero_assigned_two_local_peaks_reroutes_suspicious() -> None:
    image = np.full((48, 64), 40.0, dtype=np.float64)
    image = np.maximum(image, make_gaussian_spot((48, 64), (24.0, 22.0), 1.2, 500.0))
    image = np.maximum(image, make_gaussian_spot((48, 64), (24.0, 30.0), 1.2, 500.0))
    mask = np.zeros((48, 64), dtype=bool)
    mask[18:31, 16:37] = True
    labels, objects = _labels_and_objects(image, mask)
    obj = objects[0]
    config = default_config(single_spot_max_diameter=20.0)
    processor = ObjectProcessor(config)

    recovery = processor.recover_local_peaks(image, labels == obj.label, obj)
    assert recovery.filtered_count >= 2
    assert recovery.peak_source == "recovered_local_detector"

    result = processor.process_suspicious(
        image,
        labels == obj.label,
        obj,
        assigned_peaks=recovery.peaks,
        candidate_id_start=1,
        peak_source="recovered_local_detector",
        recovery=recovery,
    )
    assert result.debug.local_peak_recovery_attempted is True
    assert result.debug.local_peak_recovery_filtered_count >= 2
    assert result.candidates[0].peak_source == "recovered_local_detector"
    assert result.debug.tried_gmm is True or result.path in {"gmm", "single", "fallback"}
    assert result.path != "fast_single"


def test_global_assigned_peak_behavior_unchanged() -> None:
    image = make_gaussian_spot((64, 64), (32.0, 32.0), sigma=1.5, amplitude=500.0)
    result = run_puncta_declump(image, default_config(single_spot_max_diameter=12.0))
    accepted = result.accepted[0]
    assert accepted.path == "fast_single"
    assert accepted.peak_source == "assigned_global"
    assert accepted.local_peak_recovery_attempted is False
    assert result.summary.local_peak_recovery_attempts == 0


def test_touching_neighbor_assignment_miss_recovers(monkeypatch) -> None:
    image = np.full((48, 48), 40.0, dtype=np.float64)
    image = np.maximum(image, make_gaussian_spot((48, 48), (20.0, 16.0), 1.2, 520.0))
    image = np.maximum(image, make_gaussian_spot((48, 48), (20.0, 26.0), 1.2, 510.0))
    mask = np.zeros((48, 48), dtype=bool)
    mask[16:25, 12:20] = True
    mask[16:25, 21:30] = True

    original = assign_peaks_to_objects

    def drop_right_object(labels, objects, peak_table, config):
        assigned = original(labels, objects, peak_table, config)
        right = max(objects, key=lambda item: item.centroid[1])
        assigned[right.label] = []
        return assigned

    monkeypatch.setattr(
        "bioimage_pipeline.puncta.pipeline.assign_peaks_to_objects",
        drop_right_object,
    )
    result = run_puncta_declump(
        image,
        default_config(single_spot_max_diameter=12.0, log_progress=False),
        external_mask=mask,
    )
    recovered = [
        candidate
        for candidate in result.candidates
        if candidate.local_peak_recovery_attempted
    ]
    assert recovered
    assert any(candidate.peak_source == "recovered_local_detector" for candidate in recovered)
    assert any(
        candidate.accepted and candidate.fit_status == "fit_ok" for candidate in recovered
    )
    left = [c for c in result.candidates if not c.local_peak_recovery_attempted]
    assert left
    assert all(c.peak_source == "assigned_global" for c in left)


def test_tiny_object_masked_argmax_recovery(monkeypatch) -> None:
    image = np.full((24, 24), 40.0, dtype=np.float64)
    image[10, 10] = 400.0
    image[10, 11] = 280.0
    image[11, 10] = 260.0
    mask = np.zeros((24, 24), dtype=bool)
    mask[10:12, 10:12] = True

    def empty_detect(self, patch_raw, patch_mask):
        return PeakDetectionResult(raw_peaks=[], filtered_peaks=[], method="mock_empty")

    monkeypatch.setattr(MaximaDetector, "detect", empty_detect)
    labels, objects = _labels_and_objects(image, mask)
    obj = objects[0]
    assert obj.area <= 9
    processor = ObjectProcessor(default_config())
    recovery = processor.recover_local_peaks(image, labels == obj.label, obj)
    assert recovery.success
    assert recovery.peak_source == "recovered_masked_argmax"
    assert len(recovery.peaks) == 1
    result = processor.process_recovered_single(
        obj, recovery, candidate_id_start=1, route=_route()
    )
    assert result.debug.peak_source in {"recovered_masked_argmax", "fallback"}
    assert result.candidates[0].local_peak_recovery_attempted is True
    assert abs(result.candidates[0].initial_row - 10.0) < 2.0
    assert abs(result.candidates[0].initial_col - 10.0) < 2.0


def test_noisy_object_without_local_evidence_remains_fallback(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    image = 50.0 + rng.normal(0.0, 0.4, size=(48, 48))
    mask = make_binary_disk((48, 48), (24.0, 24.0), radius=10.0)

    def empty_detect(self, patch_raw, patch_mask):
        return PeakDetectionResult(raw_peaks=[], filtered_peaks=[], method="mock_empty")

    monkeypatch.setattr(MaximaDetector, "detect", empty_detect)
    labels, objects = _labels_and_objects(image, mask)
    obj = objects[0]
    processor = ObjectProcessor(default_config())
    recovery = processor.recover_local_peaks(image, labels == obj.label, obj)
    assert recovery.success is False
    assert recovery.peak_source == "fallback"

    result = processor.process_fast(
        obj,
        [],
        candidate_id_start=1,
        route=_route(),
        recovery=recovery,
    )
    assert result.path == "fallback"
    assert result.candidates[0].peak_source == "fallback"
    assert result.candidates[0].fit_status == "fit_failed_fallback"
    assert result.debug.local_peak_recovery_attempted is True
    assert result.debug.local_peak_recovery_success is False


def test_recovery_does_not_hallucinate_extra_peaks_on_isolated_control() -> None:
    image = make_gaussian_spot((64, 64), (32.0, 32.0), sigma=1.5, amplitude=500.0)
    mask = make_binary_disk((64, 64), (32.0, 32.0), radius=5.0)
    labels, objects = _labels_and_objects(image, mask)
    obj = objects[0]
    processor = ObjectProcessor(default_config())
    recovery = processor.recover_local_peaks(image, labels == obj.label, obj)
    assert recovery.filtered_count == 1

    result = run_puncta_declump(image, default_config(log_progress=False))
    assert result.summary.local_peak_recovery_attempts == 0
    assert result.accepted[0].path == "fast_single"
    assert result.accepted[0].peak_source == "assigned_global"
    assert result.accepted[0].n_filtered_local_maxima == 1


def test_recovery_disabled_keeps_brightest_fallback(monkeypatch) -> None:
    image = make_gaussian_spot((48, 48), (24.0, 24.0), sigma=1.3, amplitude=500.0)
    mask = make_binary_disk((48, 48), (24.0, 24.0), radius=4.0)

    original = assign_peaks_to_objects

    def drop_all(labels, objects, peak_table, config):
        assigned = original(labels, objects, peak_table, config)
        return {label: [] for label in assigned}

    monkeypatch.setattr(
        "bioimage_pipeline.puncta.pipeline.assign_peaks_to_objects",
        drop_all,
    )
    result = run_puncta_declump(
        image,
        default_config(
            local_peak_recovery_enabled=False,
            log_progress=False,
        ),
        external_mask=mask,
    )
    assert result.summary.local_peak_recovery_attempts == 0
    assert result.accepted[0].path == "fallback"
    assert result.accepted[0].peak_source == "fallback"


def test_recovered_single_never_invokes_gaussian_fitter() -> None:
    """One recovered peak must not call GaussianModelSelector.single_fitter.fit_peak."""
    image = make_gaussian_spot((48, 48), (24.0, 24.0), sigma=1.3, amplitude=500.0)
    mask = make_binary_disk((48, 48), (24.0, 24.0), radius=4.0)
    labels, objects = _labels_and_objects(image, mask)
    obj = objects[0]
    processor = ObjectProcessor(default_config())
    recovery = processor.recover_local_peaks(image, labels == obj.label, obj)
    assert recovery.filtered_count == 1

    with patch.object(
        processor.model_selector.single_fitter,
        "fit_peak",
        side_effect=AssertionError("fit_peak must not run for recovered fast_single"),
    ):
        result = processor.process_recovered_single(
            obj, recovery, candidate_id_start=1, route=_route()
        )
    assert result.path == "fast_single"
    assert result.candidates[0].peak_source == "recovered_local_detector"


def test_pipeline_recovery_one_peak_is_not_unconditional_fallback(monkeypatch) -> None:
    """Recovered single peak stays on fast path; no Gaussian fit invoked."""
    image = make_gaussian_spot((48, 48), (24.0, 24.0), sigma=1.3, amplitude=500.0)
    mask = make_binary_disk((48, 48), (24.0, 24.0), radius=4.0)
    original = assign_peaks_to_objects

    def drop_all(labels, objects, peak_table, config):
        assigned = original(labels, objects, peak_table, config)
        return {label: [] for label in assigned}

    monkeypatch.setattr(
        "bioimage_pipeline.puncta.pipeline.assign_peaks_to_objects",
        drop_all,
    )
    result = run_puncta_declump(
        image,
        default_config(log_progress=False),
        external_mask=mask,
    )
    accepted = result.accepted[0]
    assert accepted.local_peak_recovery_attempted is True
    assert accepted.peak_source == "recovered_local_detector"
    # CRITICAL: fast_single path, not "single" which implies Gaussian fit
    assert accepted.path == "fast_single"
    assert accepted.fit_status == "fit_ok"
    # Verify no Gaussian fit occurred
    assert accepted.one_gaussian_r_squared is None
    assert accepted.one_gaussian_residual_relative is None
    assert result.summary.local_peak_recovery_one_peak == 1
    assert result.summary.fallback_objects == 0
