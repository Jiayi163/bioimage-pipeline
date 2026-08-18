"""Tests for image-only (maskless) puncta detection."""

from __future__ import annotations

import numpy as np
import pytest

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.peak_grouping import group_peaks
from bioimage_pipeline.puncta.pipeline import run_puncta_declump
from bioimage_pipeline.puncta.types import PeakCandidate
from tests.test_puncta_declump import default_config, make_binary_disk, make_gaussian_spot


def image_only_config(**overrides: object) -> PunctaDeclumpConfig:
    params = {
        "detection_mask_mode": "image_only",
        "diagnostic_mode": "summary",
        "min_amplitude": 5.0,
        "min_center_separation": 2.5,
        "image_only_min_snr": 2.0,
        "image_only_support_mad_multiplier": 2.0,
        "enable_gmm": True,
        "min_peak_distance": 2,
        "peak_min_relative_height": 0.15,
    }
    params.update(overrides)
    return PunctaDeclumpConfig(**params)


def test_isolated_bright_punctum_without_mask() -> None:
    image = make_gaussian_spot((64, 64), (32.0, 32.0), sigma=1.5, amplitude=500.0)
    result = run_puncta_declump(image, image_only_config())

    assert result.threshold_metadata["detection_mask_mode"] == "image_only"
    assert len(result.accepted) >= 1
    assert result.accepted[0].detection_provenance == "image_only_peak"
    assert result.image_only_diagnostics is not None
    assert len(result.image_only_diagnostics.validated_peaks) >= 1


def test_two_separated_puncta_no_gmm() -> None:
    image = np.full((64, 64), 40.0, dtype=np.float64)
    image += make_gaussian_spot((64, 64), (24.0, 24.0), sigma=1.2, amplitude=400.0, background=0.0)
    image += make_gaussian_spot((64, 64), (40.0, 40.0), sigma=1.2, amplitude=400.0, background=0.0)

    result = run_puncta_declump(image, image_only_config())

    accepted = result.accepted
    assert len(accepted) >= 2
    assert all(c.detection_provenance == "image_only_peak" for c in accepted)
    assert result.threshold_metadata["image_only_counts"]["gmm_groups"] == 0


def test_two_close_ambiguous_puncta_routed_to_gmm() -> None:
    """Two unresolved peaks in one group route to GMM when separation is below threshold."""
    image = np.full((64, 64), 40.0, dtype=np.float64)
    image += make_gaussian_spot((64, 64), (32.0, 28.0), sigma=0.85, amplitude=600.0, background=0.0)
    image += make_gaussian_spot((64, 64), (32.0, 30.2), sigma=0.85, amplitude=600.0, background=0.0)

    result = run_puncta_declump(
        image,
        image_only_config(
            image_only_rolling_ball_radius=4,
            image_only_group_link_distance=10.0,
            min_peak_distance=1,
            min_center_separation=2.5,
        ),
    )

    assert len(result.image_only_diagnostics.validated_peaks) >= 2
    assert result.threshold_metadata["image_only_counts"]["ambiguous_groups"] >= 1
    gmm_candidates = [
        c for c in result.candidates if c.detection_provenance == "gmm_unresolved_multi_peak"
    ]
    assert gmm_candidates
    assert any(c.tried_gmm for c in gmm_candidates)


def test_weak_punctum_retained_above_local_background() -> None:
    image = make_gaussian_spot((64, 64), (32.0, 32.0), sigma=1.2, amplitude=80.0, background=40.0)
    result = run_puncta_declump(
        image,
        image_only_config(image_only_min_snr=1.5, peak_min_relative_height=0.1),
    )

    assert len(result.accepted) >= 1


def test_pure_background_few_false_peaks() -> None:
    rng = np.random.default_rng(42)
    image = 40.0 + rng.normal(0, 2.0, size=(64, 64))
    result = run_puncta_declump(
        image,
        image_only_config(image_only_min_snr=4.0),
    )

    assert len(result.accepted) <= 1


def test_dense_small_puncta_produce_multiple_candidates() -> None:
    image = np.full((80, 80), 40.0, dtype=np.float64)
    centers = [(16 + 12 * c, 16 + 12 * r) for r in range(5) for c in range(5)]
    for center in centers:
        image += make_gaussian_spot(
            (80, 80),
            center,
            sigma=0.8,
            amplitude=200.0,
            background=0.0,
        )

    result = run_puncta_declump(
        image,
        image_only_config(
            image_only_rolling_ball_radius=4,
            image_only_min_snr=1.5,
            min_center_separation=2.0,
            image_only_group_link_distance=5.0,
        ),
    )

    assert result.image_only_diagnostics is not None
    assert len(result.image_only_diagnostics.validated_peaks) >= 10
    assert len(result.accepted) >= 10


def test_saturated_punctum_remains_detectable() -> None:
    image = make_gaussian_spot((64, 64), (32.0, 32.0), sigma=1.2, amplitude=500.0)
    image[32, 32] = 65535.0

    result = run_puncta_declump(
        image,
        image_only_config(image_only_rolling_ball_radius=4),
    )

    assert len(result.accepted) >= 1


def test_image_only_mode_works_without_mask() -> None:
    image = make_gaussian_spot((48, 48), (24.0, 24.0), sigma=1.2, amplitude=400.0)
    result = run_puncta_declump(image, image_only_config(), external_mask=None)

    assert result.threshold_metadata["detection_mask_mode"] == "image_only"
    assert len(result.accepted) >= 1


def test_image_only_progress_logging(capsys: pytest.CaptureFixture[str]) -> None:
    image = make_gaussian_spot((64, 64), (32.0, 32.0), sigma=1.5, amplitude=500.0)
    run_puncta_declump(image, image_only_config(log_progress=True))

    err = capsys.readouterr().err
    assert "[image_only] estimating background..." in err
    assert "[image_only] support map generated in" in err
    assert "[image_only] detecting peaks..." in err
    assert "[image_only] raw_peaks=" in err
    assert "[image_only] grouping peaks..." in err
    assert "[image_only] groups=" in err
    assert "[image_only] GMM fitting total=" in err


def test_image_only_progress_respects_no_progress(capsys: pytest.CaptureFixture[str]) -> None:
    image = make_gaussian_spot((64, 64), (32.0, 32.0), sigma=1.5, amplitude=500.0)
    run_puncta_declump(image, image_only_config(log_progress=False))

    err = capsys.readouterr().err
    assert "[image_only]" not in err


def test_external_mask_mode_unchanged() -> None:
    image = make_gaussian_spot((64, 64), (32.0, 32.0), sigma=1.5, amplitude=500.0)
    mask = make_binary_disk((64, 64), (32.0, 32.0), radius=4.0)

    baseline = run_puncta_declump(image, default_config(single_spot_max_diameter=12.0))
    external = run_puncta_declump(
        image,
        default_config(single_spot_max_diameter=12.0, detection_mask_mode="external"),
        external_mask=mask,
    )

    assert external.threshold_metadata.get("method") == "external_mask"
    assert len(external.accepted) == len(baseline.accepted)
    assert external.accepted[0].path == baseline.accepted[0].path


def routing_config(**overrides: object) -> PunctaDeclumpConfig:
    params = {
        "detection_mask_mode": "image_only",
        "min_center_separation": 2.5,
        "min_amplitude": 5.0,
        "image_only_group_link_distance": 20.0,
    }
    params.update(overrides)
    return PunctaDeclumpConfig(**params)


def test_three_well_separated_validated_peaks_route_direct() -> None:
    peaks = [
        PeakCandidate(10.0, 10.0, 100.0),
        PeakCandidate(10.0, 20.0, 100.0),
        PeakCandidate(10.0, 30.0, 100.0),
    ]
    groups = group_peaks(peaks, (64, 64), routing_config())

    assert len(groups) == 1
    assert groups[0].route == "direct"
    assert groups[0].routing_reason == "direct_resolved_multi_peak"


def test_five_well_separated_validated_peaks_route_direct() -> None:
    peaks = [PeakCandidate(10.0, 10.0 + 8.0 * i, 100.0) for i in range(5)]
    groups = group_peaks(peaks, (64, 64), routing_config())

    assert len(groups) == 1
    assert groups[0].route == "direct"
    assert groups[0].routing_reason == "direct_resolved_multi_peak"


def test_three_close_overlapping_peaks_route_gmm() -> None:
    peaks = [
        PeakCandidate(10.0, 10.0, 100.0),
        PeakCandidate(10.0, 12.0, 100.0),
        PeakCandidate(10.0, 14.0, 100.0),
    ]
    groups = group_peaks(peaks, (64, 64), routing_config())

    assert len(groups) == 1
    assert groups[0].route == "gmm"
    assert groups[0].routing_reason == "gmm_unresolved_multi_peak"


def test_mixed_ambiguous_separation_routes_gmm() -> None:
    peaks = [
        PeakCandidate(10.0, 10.0, 100.0),
        PeakCandidate(10.0, 12.0, 100.0),
        PeakCandidate(10.0, 30.0, 100.0),
    ]
    groups = group_peaks(
        peaks,
        (64, 64),
        routing_config(image_only_group_link_distance=25.0),
    )

    assert len(groups) == 1
    assert groups[0].route == "gmm"
    assert groups[0].routing_reason == "gmm_unresolved_multi_peak"


def test_single_and_two_peak_direct_routing_unchanged() -> None:
    single = group_peaks([PeakCandidate(10.0, 10.0, 100.0)], (64, 64), routing_config())
    assert single[0].route == "direct"
    assert single[0].routing_reason == "direct_single"

    pair = group_peaks(
        [PeakCandidate(10.0, 10.0, 100.0), PeakCandidate(10.0, 20.0, 100.0)],
        (64, 64),
        routing_config(),
    )
    assert pair[0].route == "direct"
    assert pair[0].routing_reason == "direct_resolved_multi_peak"

    close_pair = group_peaks(
        [PeakCandidate(10.0, 10.0, 100.0), PeakCandidate(10.0, 12.0, 100.0)],
        (64, 64),
        routing_config(),
    )
    assert close_pair[0].route == "gmm"
    assert close_pair[0].routing_reason == "gmm_unresolved_multi_peak"


def test_low_amplitude_multi_peak_group_routes_gmm() -> None:
    peaks = [
        PeakCandidate(10.0, 10.0, 100.0),
        PeakCandidate(10.0, 20.0, 1.0),
        PeakCandidate(10.0, 30.0, 100.0),
    ]
    groups = group_peaks(peaks, (64, 64), routing_config(min_amplitude=5.0))

    assert groups[0].route == "gmm"
    assert groups[0].routing_reason == "gmm_unresolved_multi_peak"


def test_three_separated_puncta_cluster_routes_direct_with_provenance() -> None:
    image = np.full((80, 80), 40.0, dtype=np.float64)
    for col in (20.0, 30.0, 40.0):
        image += make_gaussian_spot(
            (80, 80),
            (40.0, col),
            sigma=1.0,
            amplitude=400.0,
            background=0.0,
        )

    result = run_puncta_declump(
        image,
        image_only_config(
            image_only_group_link_distance=15.0,
            min_center_separation=2.5,
        ),
    )

    assert result.threshold_metadata["image_only_counts"]["gmm_groups"] == 0
    assert len(result.accepted) >= 3
    assert all(
        c.detection_provenance == "direct_resolved_multi_peak" for c in result.accepted
    )
