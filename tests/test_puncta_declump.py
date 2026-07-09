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
from bioimage_pipeline.puncta.background import build_object_patch
from bioimage_pipeline.puncta.candidate_filter import CandidateFilter
from bioimage_pipeline.puncta.types import GaussianComponent, ObjectInfo, PeakCandidate


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
        "expected_single_spot_diameter": 5.0,
        "smoothing_sigma": 0.4,
        "min_peak_distance": 2,
        "fit_roi_radius": 5,
        "min_sigma": 0.3,
        "max_sigma": 5.0,
        "max_center_shift": 3.0,
        "min_amplitude": 5.0,
        "max_fit_residual": 100.0,
        "max_fit_residual_relative": 0.5,
        "min_center_separation": 2.5,
        "gmm_min_component_separation": 1.5,
        "residual_gmm_r_squared": 0.75,
        "residual_gmm_relative": 0.12,
        "diagnostic_mode": "summary",
    }
    params.update(overrides)
    return PunctaDeclumpConfig(**params)


def test_single_gaussian_spot_detects_one_punctum() -> None:
    image = make_gaussian_spot((64, 64), (32.0, 32.0), sigma=1.5, amplitude=500.0)
    result = run_puncta_declump(image, default_config(single_spot_max_diameter=12.0))

    assert result.summary.total_mask_objects == 1
    assert len(result.accepted) == 1
    accepted = result.accepted[0]
    assert accepted.path == "single"
    assert accepted.fit_status == "fit_ok"
    assert accepted.tried_gmm is False
    assert abs(accepted.final_row - 32.0) < 2.0
    assert abs(accepted.final_col - 32.0) < 2.0
    assert accepted.n_raw_local_maxima is not None


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

    assert small_result.candidates[0].path == "single"
    # Large diameter threshold forces GMM consideration even for a true single spot.
    assert large_result.candidates[0].tried_gmm is True
    assert large_result.candidates[0].gmm_trigger_reasons is not None
    assert "large_diameter" in large_result.candidates[0].gmm_trigger_reasons
    # Final accepted model may still be one Gaussian after model selection.
    assert large_result.candidates[0].fit_status == "fit_ok"


def test_rejection_center_shift_too_large() -> None:
    config = default_config(max_center_shift=0.1)
    image = make_gaussian_spot((32, 32), (16.0, 16.0), sigma=1.5, amplitude=500.0)
    mask = make_binary_disk((32, 32), (16.0, 16.0), radius=4.0)
    peak = PeakCandidate(row=16.0, col=16.0, intensity=float(image[16, 16]))

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
    patch = build_object_patch(image, mask, obj, config)
    component = GaussianComponent(
        component_id=1,
        initial_row=16.0,
        initial_col=16.0,
        fitted_row=16.0,
        fitted_col=20.0,
        sigma_row=1.5,
        sigma_col=1.5,
        amplitude=400.0,
        background=40.0,
        residual_rmse=5.0,
        residual_relative=0.01,
        r_squared=0.99,
        model_score=0.99,
        n_components_in_model=1,
        fit_succeeded=True,
    )
    candidate_filter = CandidateFilter(config)
    candidate = candidate_filter.evaluate_component(
        obj,
        peak,
        component,
        candidate_id=1,
        component_id=1,
        path="gmm",
        object_mask=patch.object_mask,
        patch=patch,
    )

    assert not candidate.accepted
    assert candidate.rejection_reason == "center_shift_too_large"
    assert candidate.fit_status == "rejected_bad_fit"


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

    image = np.full((20, 20), 40.0, dtype=np.float64)
    image[8:13, 8:13] = 240.0
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:13, 8:13] = True
    patch = build_object_patch(image, mask, obj, config)

    first_peak = PeakCandidate(row=10.0, col=10.0, intensity=500.0)
    second_peak = PeakCandidate(row=10.5, col=10.5, intensity=480.0)

    def make_component(row: float, col: float, amplitude: float) -> GaussianComponent:
        return GaussianComponent(
            component_id=1,
            initial_row=row,
            initial_col=col,
            fitted_row=row,
            fitted_col=col,
            sigma_row=1.5,
            sigma_col=1.5,
            amplitude=amplitude,
            background=40.0,
            residual_rmse=5.0,
            residual_relative=0.02,
            r_squared=0.98,
            model_score=0.98,
            n_components_in_model=2,
            fit_succeeded=True,
        )

    first = candidate_filter.evaluate_component(
        obj,
        first_peak,
        make_component(10.0, 10.0, 200.0),
        candidate_id=1,
        component_id=1,
        path="gmm",
        object_mask=patch.object_mask,
        patch=patch,
    )
    second = candidate_filter.evaluate_component(
        obj,
        second_peak,
        make_component(10.5, 10.5, 150.0),
        candidate_id=2,
        component_id=2,
        path="gmm",
        object_mask=patch.object_mask,
        patch=patch,
    )

    assert first.accepted
    assert not second.accepted
    assert second.rejection_reason == "duplicate_center_too_close"
    assert second.fit_status == "rejected_duplicate"


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


def test_fallback_has_no_fitted_coordinates() -> None:
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

    fallback = next(c for c in result.candidates if c.fit_status == "fit_failed_fallback")
    assert fallback.fitted_row is None
    assert fallback.fitted_col is None
    assert fallback.path == "fallback"
    assert fallback.warning == "fit_failed_used_brightest_pixel"


def test_small_object_with_two_peaks_uses_gmm() -> None:
    image = np.full((48, 48), 40.0, dtype=np.float64)
    spot_a = make_gaussian_spot((48, 48), (24.0, 20.0), sigma=1.2, amplitude=400.0)
    spot_b = make_gaussian_spot((48, 48), (24.0, 28.0), sigma=1.2, amplitude=400.0)
    image = np.maximum(image, spot_a)
    image = np.maximum(image, spot_b)

    mask = make_binary_disk((48, 48), (24.0, 24.0), radius=5.0)
    config = default_config(
        single_spot_max_diameter=20.0,
        min_peak_distance=2,
        min_center_separation=2.5,
        gmm_min_component_separation=1.5,
    )
    result = run_puncta_declump(image, config, external_mask=mask)

    assert result.summary.gmm_path_objects == 1
    assert len(result.gaussian_fitted) >= 2
    assert all(c.path == "gmm" for c in result.gaussian_fitted)
    assert result.gaussian_fitted[0].tried_gmm is True
    assert result.gaussian_fitted[0].n_raw_local_maxima is not None
    assert result.gaussian_fitted[0].n_raw_local_maxima >= 2


def test_residual_driven_gmm_retry_on_poor_single_fit() -> None:
    """Close overlapping spots with one mask blob should retry GMM after poor 1-Gaussian fit."""
    image = np.full((64, 64), 40.0, dtype=np.float64)
    spot_a = make_gaussian_spot((64, 64), (32.0, 30.0), sigma=1.4, amplitude=450.0)
    spot_b = make_gaussian_spot((64, 64), (32.0, 34.5), sigma=1.4, amplitude=420.0)
    image = np.maximum(image, spot_a)
    image = np.maximum(image, spot_b)
    mask = make_binary_disk((64, 64), (32.0, 32.0), radius=6.0)

    config = default_config(
        single_spot_max_diameter=30.0,
        min_peak_distance=2,
        min_center_separation=2.0,
        gmm_min_component_separation=1.2,
        gmm_trigger_r_squared=0.95,
        gmm_trigger_residual_relative=0.05,
    )
    result = run_puncta_declump(image, config, external_mask=mask)
    assert any(c.tried_gmm for c in result.candidates)
    assert any(
        c.gmm_trigger_reasons
        and (
            "low_one_gaussian_r2" in c.gmm_trigger_reasons
            or "filtered_peaks" in c.gmm_trigger_reasons
            or "raw_peaks" in c.gmm_trigger_reasons
        )
        for c in result.candidates
    )


def test_export_writes_undersplit_report(tmp_path: Path) -> None:
    image = make_gaussian_spot((48, 48), (24.0, 24.0), sigma=1.2, amplitude=400.0)
    result = run_puncta_declump(image, default_config())
    exporter = ResultExporter()
    paths = exporter.export_all(tmp_path, result, stem="test", image_shape=image.shape)
    assert paths["object_diagnostics"].exists()
    assert paths["undersplit_report"].exists()
    assert paths["undersplit_report_json"].exists()


def test_diagnostic_mode_summary_skips_pngs(tmp_path: Path) -> None:
    image = make_gaussian_spot((48, 48), (24.0, 24.0), sigma=1.2, amplitude=400.0)
    config = default_config(diagnostic_mode="summary")
    result = run_puncta_declump(
        image,
        config,
        diagnostics_dir=str(tmp_path / "diagnostics"),
    )
    assert result.diagnostic_artifacts == []
    assert not any(tmp_path.rglob("*.png"))


def test_diagnostic_mode_selected_objects_only(tmp_path: Path) -> None:
    image = np.full((64, 64), 40.0, dtype=np.float64)
    spot_a = make_gaussian_spot((64, 64), (32.0, 28.0), sigma=1.2, amplitude=400.0)
    spot_b = make_gaussian_spot((64, 64), (32.0, 36.0), sigma=1.2, amplitude=400.0)
    image = np.maximum(image, spot_a)
    image = np.maximum(image, spot_b)
    mask = make_binary_disk((64, 64), (32.0, 32.0), radius=6.0)

    config = default_config(
        diagnostic_mode="selected_objects",
        diagnostic_object_ids=(1,),
        max_diagnostic_objects=50,
    )
    result = run_puncta_declump(
        image,
        config,
        external_mask=mask,
        diagnostics_dir=str(tmp_path / "diagnostics"),
    )
    assert len(result.diagnostic_artifacts) >= 1
    assert all("object_0001" in path for path in result.diagnostic_artifacts)


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
    paths = exporter.export_all(
        tmp_path,
        result,
        stem="test",
        image_shape=image.shape,
        image=image,
        config=default_config(),
    )

    assert paths["csv"].exists()
    assert paths["summary"].exists()
    assert paths["seeds"].exists()
    assert paths["mask"].exists()
    assert paths["labels"].exists()
    assert paths["fit_ok_centers"].exists()
    assert paths["component_labels"].exists()
    assert paths["overlay_tiff"].exists()
    assert paths["gmm_object_labels"].exists()

    dataframe = pd.read_csv(paths["csv"])
    expected_columns = {
        "object_id",
        "candidate_id",
        "component_id",
        "path",
        "fit_status",
        "initial_row",
        "initial_col",
        "fitted_row",
        "fitted_col",
        "x_fit",
        "y_fit",
        "final_row",
        "final_col",
        "center_shift",
        "sigma",
        "sigma_row",
        "sigma_col",
        "sigma_x",
        "sigma_y",
        "width_fwhm",
        "amplitude",
        "background",
        "residual_rmse",
        "residual_relative",
        "r_squared",
        "model_score",
        "n_components_in_model",
        "accepted",
        "rejection_reason",
        "warning",
        "gmm_trigger_reasons",
        "n_raw_local_maxima",
        "n_filtered_local_maxima",
        "tried_gmm",
        "one_gaussian_r_squared",
        "model_selection_reason",
        "under_split_suspect",
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
        diagnostic_mode="summary",
        diagnostic_objects=None,
        max_diagnostic_objects=50,
        include_fallback_centers=False,
        no_fiji_tiffs=False,
        no_progress=True,
    )

    output = run_cli(args)
    result = output["result"]

    assert result.threshold_metadata["image_plane"]["frame_index"] == 0
    assert len(result.accepted) >= 1
    assert (output_dir / "puncta_overlay.png").exists()


def test_balanced_mode_skips_gmm_for_clean_single() -> None:
    image = make_gaussian_spot((64, 64), (32.0, 32.0), sigma=1.2, amplitude=600.0)
    config = default_config(
        single_spot_max_diameter=12.0,
        expected_single_spot_diameter=5.0,
        log_progress=False,
    )
    result = run_puncta_declump(image, config)
    assert result.candidates[0].tried_gmm is False
    assert result.summary.gmm_triggered_objects == 0


def test_balanced_mode_exports_diagnostics_for_suspicious_only(tmp_path: Path) -> None:
    image = np.full((64, 64), 40.0, dtype=np.float64)
    spot_a = make_gaussian_spot((64, 64), (32.0, 28.0), sigma=1.2, amplitude=400.0)
    spot_b = make_gaussian_spot((64, 64), (32.0, 36.0), sigma=1.2, amplitude=400.0)
    image = np.maximum(image, spot_a)
    image = np.maximum(image, spot_b)
    mask = make_binary_disk((64, 64), (32.0, 32.0), radius=6.0)

    config = default_config(
        diagnostic_mode="balanced",
        log_progress=False,
    )
    result = run_puncta_declump(
        image,
        config,
        external_mask=mask,
        diagnostics_dir=str(tmp_path / "diagnostics"),
    )
    assert result.summary.diagnostics_exported >= 1
    assert any(tmp_path.rglob("*.png"))


def test_runtime_summary_populated() -> None:
    image = make_gaussian_spot((48, 48), (24.0, 24.0), sigma=1.2, amplitude=400.0)
    config = default_config(log_progress=False)
    result = run_puncta_declump(image, config)
    assert result.summary.total_runtime_seconds > 0
    assert "runtime" in result.threshold_metadata
    runtime = result.threshold_metadata["runtime"]
    assert runtime["total_objects"] == 1
    assert runtime["gmm_triggered_objects"] == 0
