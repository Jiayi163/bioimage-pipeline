"""Tests for image-only (maskless) puncta detection."""

from __future__ import annotations

import numpy as np
import pytest

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.pipeline import run_puncta_declump
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
    """Two peaks in one group routed to GMM when peak count triggers mixture fitting."""
    image = np.full((64, 64), 40.0, dtype=np.float64)
    image += make_gaussian_spot((64, 64), (28.0, 32.0), sigma=1.0, amplitude=500.0, background=0.0)
    image += make_gaussian_spot((64, 64), (34.0, 32.0), sigma=1.0, amplitude=500.0, background=0.0)

    result = run_puncta_declump(
        image,
        image_only_config(
            image_only_rolling_ball_radius=4,
            min_reliable_peaks_for_routing=2,
            image_only_group_link_distance=10.0,
            min_peak_distance=2,
        ),
    )

    assert len(result.image_only_diagnostics.validated_peaks) >= 2
    assert result.threshold_metadata["image_only_counts"]["ambiguous_groups"] >= 1
    gmm_candidates = [
        c
        for c in result.candidates
        if c.detection_provenance in ("image_only_group", "image_only_gmm")
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
