"""Tests for puncta declumping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.export import ResultExporter
from bioimage_pipeline.puncta.pipeline import run_puncta_declump
from bioimage_pipeline.puncta.candidate_filter import CandidateFilter
from bioimage_pipeline.puncta.gaussian_fitter import GaussianFitter2D
from bioimage_pipeline.puncta.types import GaussianFitResult, ObjectInfo, PeakCandidate


def make_gaussian_spot(
    shape: tuple[int, int],
    center: tuple[float, float],
    sigma: float,
    amplitude: float,
    background: float = 40.0,
) -> np.ndarray:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    row_center, col_center = center
    distance_sq = (rows - row_center) ** 2 + (cols - col_center) ** 2
    image = background + amplitude * np.exp(-distance_sq / (2.0 * sigma**2))
    return image.astype(np.float64)


def make_binary_disk(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
) -> np.ndarray:
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    row_center, col_center = center
    mask = (rows - row_center) ** 2 + (cols - col_center) ** 2 <= radius**2
    return mask.astype(bool)


def default_config(**overrides: object) -> PunctaDeclumpConfig:
    params = {
        "threshold_method": "manual",
        "manual_threshold_value": 50.0,
        "min_object_area": 4,
        "single_spot_max_diameter": 12.0,
        "expected_single_spot_diameter": 3.0,
        "smoothing_sigma": 0.5,
        "min_peak_distance": 3,
        "fit_roi_radius": 5,
        "min_sigma": 0.3,
        "max_sigma": 5.0,
        "max_center_shift": 3.0,
        "min_amplitude": 5.0,
        "max_fit_residual": 100.0,
        "max_fit_residual_relative": 0.5,
        "min_center_separation": 3.0,
    }
    params.update(overrides)
    return PunctaDeclumpConfig(**params)


def test_single_gaussian_spot_detects_one_punctum() -> None:
    image = make_gaussian_spot((64, 64), (32.0, 32.0), sigma=1.5, amplitude=500.0)
    result = run_puncta_declump(image, default_config(single_spot_max_diameter=12.0))

    assert result.summary.total_mask_objects == 1
    assert result.summary.small_single_objects == 1
    assert len(result.accepted) == 1
    accepted = result.accepted[0]
    assert accepted.path == "single"
    assert abs(accepted.final_row - 32.0) < 2.0
    assert abs(accepted.final_col - 32.0) < 2.0


def test_two_merged_gaussians_declump_to_two_puncta() -> None:
    image = np.full((80, 80), 40.0, dtype=np.float64)
    spot_a = make_gaussian_spot((80, 80), (38.0, 30.0), sigma=1.5, amplitude=500.0)
    spot_b = make_gaussian_spot((80, 80), (38.0, 50.0), sigma=1.5, amplitude=500.0)
    image = np.maximum(image, spot_a)
    image = np.maximum(image, spot_b)

    # One connected mask blob spanning both peaks (threshold alone would split them).
    merged_mask = np.zeros((80, 80), dtype=bool)
    merged_mask[30:47, 22:58] = True

    config = default_config(
        single_spot_max_diameter=7.0,
        min_peak_distance=4,
        min_center_separation=4.0,
    )
    result = run_puncta_declump(image, config, external_mask=merged_mask)

    assert result.summary.total_mask_objects == 1
    assert result.summary.large_clumped_objects == 1
    assert len(result.accepted) == 2
    cols = sorted(candidate.final_col for candidate in result.accepted)
    assert cols[0] < 40.0
    assert cols[1] > 40.0


def test_size_gate_routes_small_vs_large_object() -> None:
    image = make_gaussian_spot((64, 64), (32.0, 32.0), sigma=1.2, amplitude=600.0)

    small_result = run_puncta_declump(
        image,
        default_config(single_spot_max_diameter=20.0),
    )
    large_result = run_puncta_declump(
        image,
        default_config(single_spot_max_diameter=3.0),
    )

    assert small_result.summary.small_single_objects == 1
    assert small_result.candidates[0].path == "single"
    assert large_result.summary.large_clumped_objects == 1
    assert any(candidate.path == "declump" for candidate in large_result.candidates)


def test_rejection_center_shift_too_large() -> None:
    config = default_config(max_center_shift=0.1)
    fitter = GaussianFitter2D(config)
    image = make_gaussian_spot((32, 32), (16.0, 16.0), sigma=1.5, amplitude=500.0)
    peak = PeakCandidate(row=16.0, col=16.0, intensity=float(image[16, 16]))
    fit = fitter.fit(image, peak)

    obj = ObjectInfo(
        label=1,
        area=20.0,
        equivalent_diameter=5.0,
        bbox=(8, 8, 24, 24),
        centroid=(16.0, 16.0),
        brightest_row=16.0,
        brightest_col=16.0,
        brightest_intensity=540.0,
    )
    candidate_filter = CandidateFilter(config)
    # Force a large apparent shift while keeping a valid fit payload.
    fit.fitted_row = 16.0
    fit.fitted_col = 20.0
    fit.fit_succeeded = True
    fit.amplitude = 400.0
    fit.residual_rmse = 5.0
    candidate = candidate_filter.evaluate(
        obj,
        peak,
        fit,
        candidate_id=1,
        path="declump",
        object_mask=np.ones((32, 32), dtype=bool),
    )

    assert not candidate.accepted
    assert candidate.rejection_reason == "center_shift_too_large"


def test_rejection_duplicate_centers() -> None:
    config = default_config(min_center_separation=5.0)
    candidate_filter = CandidateFilter(config)
    obj = ObjectInfo(
        label=1,
        area=40.0,
        equivalent_diameter=10.0,
        bbox=(0, 0, 20, 20),
        centroid=(10.0, 10.0),
        brightest_row=10.0,
        brightest_col=10.0,
        brightest_intensity=500.0,
    )

    first_peak = PeakCandidate(row=10.0, col=10.0, intensity=500.0)
    second_peak = PeakCandidate(row=10.5, col=10.5, intensity=480.0)
    good_fit = GaussianFitResult(
        fitted_row=10.0,
        fitted_col=10.0,
        sigma=1.5,
        width_fwhm=3.5,
        amplitude=200.0,
        background=40.0,
        residual_rmse=5.0,
        roi_touches_edge=False,
        fit_succeeded=True,
    )
    close_fit = GaussianFitResult(
        fitted_row=10.5,
        fitted_col=10.5,
        sigma=1.5,
        width_fwhm=3.5,
        amplitude=150.0,
        background=40.0,
        residual_rmse=5.0,
        roi_touches_edge=False,
        fit_succeeded=True,
    )

    first = candidate_filter.evaluate(
        obj,
        first_peak,
        good_fit,
        candidate_id=1,
        path="declump",
        object_mask=np.ones((20, 20), dtype=bool),
    )
    second = candidate_filter.evaluate(
        obj,
        second_peak,
        close_fit,
        candidate_id=2,
        path="declump",
        object_mask=np.ones((20, 20), dtype=bool),
    )

    assert first.accepted
    assert not second.accepted
    assert second.rejection_reason == "duplicate_center_too_close"


def test_fallback_when_no_maxima_survive() -> None:
    image = np.full((64, 64), 40.0, dtype=np.float64)
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:45, 20:45] = True
    image[mask] = 120.0

    config = default_config(
        single_spot_max_diameter=5.0,
        peak_noise_tolerance=1000.0,
        min_amplitude=1000.0,
    )
    result = run_puncta_declump(image, config, external_mask=mask)

    assert result.summary.fallback_objects == 1
    assert len(result.accepted) >= 1
    assert any(candidate.path == "fallback" for candidate in result.candidates)


def test_external_mask_mode_skips_threshold() -> None:
    image = make_gaussian_spot((48, 48), (24.0, 24.0), sigma=1.2, amplitude=400.0)
    mask = make_binary_disk((48, 48), (24.0, 24.0), radius=4.0)

    result = run_puncta_declump(image, default_config(), external_mask=mask)

    assert result.threshold_metadata["method"] == "external_mask"
    assert len(result.accepted) == 1


def test_export_writes_expected_files(tmp_path: Path) -> None:
    image = make_gaussian_spot((48, 48), (24.0, 24.0), sigma=1.2, amplitude=400.0)
    result = run_puncta_declump(image, default_config())

    exporter = ResultExporter()
    paths = exporter.export_all(tmp_path, result, stem="test", image_shape=image.shape)

    assert paths["csv"].exists()
    assert paths["summary"].exists()
    assert paths["seeds"].exists()
    assert paths["mask"].exists()
    assert paths["labels"].exists()

    dataframe = pd.read_csv(paths["csv"])
    expected_columns = {
        "object_id",
        "candidate_id",
        "path",
        "initial_row",
        "initial_col",
        "fitted_row",
        "fitted_col",
        "final_row",
        "final_col",
        "center_shift",
        "sigma",
        "width_fwhm",
        "amplitude",
        "background",
        "residual_rmse",
        "accepted",
        "rejection_reason",
        "warning",
    }
    assert expected_columns.issubset(set(dataframe.columns))

    summary_payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert "summary" in summary_payload
    assert summary_payload["summary"]["total_accepted"] >= 1


def test_config_validation_rejects_invalid_sigma_bounds() -> None:
    with pytest.raises(ValueError, match="sigma"):
        PunctaDeclumpConfig(min_sigma=4.0, max_sigma=2.0)


def test_load_grayscale_plane_from_stack() -> None:
    from bioimage_pipeline.puncta.ui import load_grayscale_plane

    stack = np.stack(
        [
            make_gaussian_spot((32, 32), (16.0, 16.0), sigma=1.0, amplitude=100.0),
            np.full((32, 32), 40.0),
        ],
        axis=0,
    )
    plane, metadata = load_grayscale_plane(stack, frame_index=0, source="stack.tif")

    assert plane.shape == (32, 32)
    assert metadata["source_shape"] == (2, 32, 32)
    assert metadata["extraction"] == "flattened_stack_plane"


def test_cli_accepts_stack_input_with_frame_index(tmp_path: Path) -> None:
    from argparse import Namespace

    from bioimage_pipeline.puncta.ui import run_cli

    stack = np.stack(
        [
            make_gaussian_spot((48, 48), (24.0, 24.0), sigma=1.2, amplitude=400.0),
            np.full((48, 48), 40.0),
        ],
        axis=0,
    )
    mask_stack = np.stack(
        [
            make_binary_disk((48, 48), (24.0, 24.0), radius=4.0),
            np.zeros((48, 48), dtype=bool),
        ],
        axis=0,
    )

    input_path = tmp_path / "stack.tif"
    mask_path = tmp_path / "mask_stack.tif"
    output_dir = tmp_path / "output"

    import tifffile

    tifffile.imwrite(input_path, stack.astype(np.uint16))
    tifffile.imwrite(mask_path, mask_stack.astype(np.uint8) * 255)

    args = Namespace(
        input=input_path,
        output_dir=output_dir,
        mask=mask_path,
        stem="puncta",
        threshold_method="otsu",
        manual_threshold=100.0,
        adaptive_block_size=51,
        adaptive_offset=0.0,
        sauvola_block_size=51,
        sauvola_k=0.2,
        min_object_area=4,
        max_object_area=10_000,
        expected_single_spot_diameter=5.0,
        single_spot_max_diameter=12.0,
        smoothing_sigma=0.75,
        min_peak_distance=3,
        peak_noise_tolerance=0.0,
        fit_roi_radius=5,
        min_sigma=0.5,
        max_sigma=4.0,
        max_center_shift=3.0,
        min_amplitude=5.0,
        max_fit_residual=100.0,
        max_fit_residual_relative=0.5,
        min_center_separation=3.0,
        frame_index=0,
        mask_frame_index=None,
        show_rejected=False,
    )

    output = run_cli(args)
    result = output["result"]

    assert result.threshold_metadata["image_plane"]["frame_index"] == 0
    assert len(result.accepted) >= 1
    assert (output_dir / "puncta_overlay.png").exists()
