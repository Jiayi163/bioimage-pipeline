"""CLI helpers for puncta declumping."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bioimage_pipeline.io import extract_2d_plane, read_tiff
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig, ThresholdMethod
from bioimage_pipeline.puncta.export import ResultExporter
from bioimage_pipeline.puncta.overlay import OverlayRenderer
from bioimage_pipeline.puncta.pipeline import run_puncta_declump


def _is_channel_last_rgb(array: np.ndarray) -> bool:
    """Return True when the array looks like (H, W, C) with small C."""
    return (
        array.ndim == 3
        and array.shape[-1] in (3, 4)
        and array.shape[-1] < array.shape[0]
        and array.shape[-1] < array.shape[1]
    )


def load_grayscale_plane(
    array: np.ndarray,
    *,
    frame_index: int = 0,
    source: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Reduce a TIFF array to a 2D grayscale plane for puncta analysis."""
    arr = np.asarray(array)
    metadata: dict[str, object] = {
        "source": source,
        "source_shape": tuple(arr.shape),
        "frame_index": frame_index,
    }

    if arr.ndim == 2:
        return arr, metadata

    if _is_channel_last_rgb(arr):
        plane = arr[..., 0]
        metadata["extraction"] = "first_channel_from_hwc"
        return plane, metadata

    plane = extract_2d_plane(arr, frame_index=frame_index)
    metadata["extraction"] = "flattened_stack_plane"
    return plane, metadata


def load_mask_plane(
    array: np.ndarray,
    *,
    frame_index: int = 0,
    source: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Reduce a mask TIFF array to a 2D boolean plane."""
    plane, metadata = load_grayscale_plane(
        array,
        frame_index=frame_index,
        source=source,
    )
    metadata["source"] = source
    mask = plane != 0
    return mask, metadata


def parse_diagnostic_object_ids(value: str | None) -> tuple[int, ...]:
    """Parse comma-separated object IDs from CLI."""
    if not value:
        return ()
    ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            ids.append(int(part))
    return tuple(ids)


def build_config_from_args(args: argparse.Namespace) -> PunctaDeclumpConfig:
    """Construct config from parsed CLI arguments."""
    return PunctaDeclumpConfig(
        threshold_method=args.threshold_method,
        manual_threshold_value=args.manual_threshold,
        adaptive_block_size=args.adaptive_block_size,
        adaptive_offset=args.adaptive_offset,
        sauvola_block_size=args.sauvola_block_size,
        sauvola_k=args.sauvola_k,
        min_object_area=args.min_object_area,
        max_object_area=args.max_object_area,
        expected_single_spot_diameter=args.expected_single_spot_diameter,
        single_spot_max_diameter=args.single_spot_max_diameter,
        smoothing_sigma=args.smoothing_sigma,
        min_peak_distance=args.min_peak_distance,
        peak_noise_tolerance=args.peak_noise_tolerance,
        fit_roi_radius=args.fit_roi_radius,
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
        max_center_shift=args.max_center_shift,
        min_amplitude=args.min_amplitude,
        max_fit_residual=args.max_fit_residual,
        max_fit_residual_relative=args.max_fit_residual_relative,
        min_center_separation=args.min_center_separation,
        diagnostic_mode=args.diagnostic_mode,
        max_diagnostic_objects=args.max_diagnostic_objects,
        diagnostic_object_ids=parse_diagnostic_object_ids(args.diagnostic_objects),
        include_fallback_in_centers=args.include_fallback_centers,
        export_fiji_tiffs=not args.no_fiji_tiffs,
        log_progress=not args.no_progress,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the puncta declumping CLI parser."""
    parser = argparse.ArgumentParser(
        description="Detect and declump tiny fluorescent puncta using local maxima and Gaussian fitting.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input grayscale TIFF image.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for CSV, overlay, mask, labels, and summary outputs.",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional external binary mask TIFF (skips thresholding).",
    )
    parser.add_argument(
        "--stem",
        default="puncta",
        help="Output filename stem (default: puncta).",
    )
    parser.add_argument(
        "--threshold-method",
        choices=("otsu", "manual", "adaptive", "sauvola", "external_mask"),
        default="otsu",
        help="Threshold method for foreground mask (default: otsu).",
    )
    parser.add_argument("--manual-threshold", type=float, default=100.0)
    parser.add_argument("--adaptive-block-size", type=int, default=51)
    parser.add_argument("--adaptive-offset", type=float, default=0.0)
    parser.add_argument("--sauvola-block-size", type=int, default=51)
    parser.add_argument("--sauvola-k", type=float, default=0.2)
    parser.add_argument("--min-object-area", type=int, default=4)
    parser.add_argument("--max-object-area", type=int, default=10_000)
    parser.add_argument("--expected-single-spot-diameter", type=float, default=5.0)
    parser.add_argument("--single-spot-max-diameter", type=float, default=7.0)
    parser.add_argument("--smoothing-sigma", type=float, default=0.75)
    parser.add_argument("--min-peak-distance", type=int, default=3)
    parser.add_argument("--peak-noise-tolerance", type=float, default=0.0)
    parser.add_argument("--fit-roi-radius", type=int, default=5)
    parser.add_argument("--min-sigma", type=float, default=0.5)
    parser.add_argument("--max-sigma", type=float, default=4.0)
    parser.add_argument("--max-center-shift", type=float, default=4.0)
    parser.add_argument("--min-amplitude", type=float, default=10.0)
    parser.add_argument(
        "--max-fit-residual",
        type=float,
        default=None,
        help="Optional absolute fit RMSE limit. When omitted, only the relative limit is used.",
    )
    parser.add_argument(
        "--max-fit-residual-relative",
        type=float,
        default=0.25,
        help="Maximum fit RMSE divided by fitted amplitude (default: 0.25).",
    )
    parser.add_argument("--min-center-separation", type=float, default=3.0)
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="When the input is a stack or multi-page TIFF, use this 0-based plane index (default: 0).",
    )
    parser.add_argument(
        "--mask-frame-index",
        type=int,
        default=None,
        help="Plane index for the external mask TIFF. Defaults to --frame-index.",
    )
    parser.add_argument(
        "--show-rejected",
        action="store_true",
        help="Include rejected candidates as red crosses in the overlay.",
    )
    parser.add_argument(
        "--diagnostic-mode",
        choices=("off", "summary", "balanced", "suspicious_only", "selected_objects", "all"),
        default="balanced",
        help=(
            "Diagnostic PNG mode: off (none), summary (CSV/JSON only), "
            "balanced (default), suspicious_only (alias), selected_objects, or all."
        ),
    )
    parser.add_argument(
        "--diagnostic-objects",
        default=None,
        help="Comma-separated mask object IDs to always include in diagnostics (e.g. 126,159,713).",
    )
    parser.add_argument(
        "--max-diagnostic-objects",
        type=int,
        default=50,
        help="Maximum number of diagnostic PNGs to export (default: 50).",
    )
    parser.add_argument(
        "--include-fallback-centers",
        action="store_true",
        help="Include fallback (non-Gaussian) centers in puncta_fit_ok_centers.tif.",
    )
    parser.add_argument(
        "--no-fiji-tiffs",
        action="store_true",
        help="Skip Fiji TIFF exports (fit_ok_centers, component_labels, overlay.tif, gmm_labels).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress and runtime summary logs on stderr.",
    )
    return parser


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    """Execute puncta declumping from CLI arguments."""
    raw_image = read_tiff(args.input)
    image, image_plane_metadata = load_grayscale_plane(
        raw_image,
        frame_index=args.frame_index,
        source=str(args.input),
    )

    external_mask = None
    mask_plane_metadata: dict[str, object] | None = None
    threshold_method: ThresholdMethod = args.threshold_method
    if args.mask is not None:
        mask_frame_index = (
            args.mask_frame_index
            if args.mask_frame_index is not None
            else args.frame_index
        )
        raw_mask = read_tiff(args.mask)
        external_mask, mask_plane_metadata = load_mask_plane(
            raw_mask,
            frame_index=mask_frame_index,
            source=str(args.mask),
        )
        if external_mask.shape != image.shape:
            raise ValueError(
                f"Mask plane shape {external_mask.shape} must match image plane shape "
                f"{image.shape}. Try matching --frame-index and --mask-frame-index."
            )
        threshold_method = "external_mask"

    config = build_config_from_args(args)
    config.threshold_method = threshold_method

    diagnostics_dir: str | None = None
    if config.diagnostic_mode not in ("off", "summary"):
        diagnostics_dir = str(args.output_dir / "diagnostics")

    result = run_puncta_declump(
        image,
        config,
        external_mask=external_mask,
        diagnostics_dir=diagnostics_dir,
    )
    result.threshold_metadata["image_plane"] = image_plane_metadata
    if mask_plane_metadata is not None:
        result.threshold_metadata["mask_plane"] = mask_plane_metadata

    exporter = ResultExporter()
    paths = exporter.export_all(
        args.output_dir,
        result,
        stem=args.stem,
        image_shape=image.shape,
        image=image,
        config=config,
        show_rejected=args.show_rejected,
    )

    overlay_renderer = OverlayRenderer(cross_half_size=max(2, config.fit_roi_radius // 2))
    overlay_path = overlay_renderer.save(
        args.output_dir / f"{args.stem}_overlay.png",
        image,
        result,
        show_rejected=args.show_rejected,
    )
    paths["overlay"] = overlay_path

    gaussian_count = sum(1 for c in result.accepted if c.fit_status == "fit_ok")
    fallback_count = sum(1 for c in result.accepted if c.fit_status == "fit_failed_fallback")
    result.threshold_metadata["overlay_legend"] = {
        "green_cross_and_circle": "fit_ok (Gaussian fit accepted)",
        "cyan_line": "Seed-to-fit center shift (>0.25 px)",
        "yellow_cross": "fit_failed_fallback (integer seed only, no fitted coords)",
        "red_cross": "Rejected candidate (--show-rejected)",
        "gaussian_fit_count": gaussian_count,
        "fallback_count": fallback_count,
    }

    return {
        "result": result,
        "paths": paths,
    }
