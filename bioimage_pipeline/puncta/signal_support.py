"""Background estimation and permissive signal-support for image-only detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import morphology, measure

from bioimage_pipeline.preprocess import rolling_ball_subtract
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig


@dataclass(frozen=True)
class BackgroundEstimate:
    """Rolling-ball background estimate for image-only mode."""

    background: np.ndarray
    corrected: np.ndarray
    radius: int
    method: str = "rolling_ball"


@dataclass(frozen=True)
class SignalSupportResult:
    """Permissive signal-support map (not segmentation)."""

    support: np.ndarray
    labels: np.ndarray
    threshold: float
    noise_median: float
    noise_mad: float
    support_kind: str = "permissive_signal"
    method: str = "rolling_ball_mad"


def estimate_background(
    image: np.ndarray,
    config: PunctaDeclumpConfig,
) -> BackgroundEstimate:
    """Estimate uneven background with morphological rolling-ball opening."""
    image_arr = np.asarray(image, dtype=np.float64)
    height, width = image_arr.shape
    if config.image_only_rolling_ball_radius is not None:
        radius = config.image_only_rolling_ball_radius
    else:
        auto_radius = max(8, min(height, width) // 8)
        spot_radius = max(4, int(round(config.expected_single_spot_diameter)))
        radius = min(auto_radius, spot_radius)

    # Clip extreme outliers before rolling ball so saturated pixels do not
    # inflate the local background estimate.
    noise_median = float(np.median(image_arr))
    noise_mad = float(np.median(np.abs(image_arr - noise_median)))
    if noise_mad <= 0:
        noise_mad = float(np.std(image_arr))
    if noise_mad <= 0:
        noise_mad = 1.0
    robust_cap = noise_median + 20.0 * noise_mad
    clipped = np.clip(image_arr, 0.0, robust_cap)

    corrected_clipped = rolling_ball_subtract(clipped, radius=radius).astype(np.float64)
    background_est = clipped - corrected_clipped
    corrected = np.clip(image_arr - background_est, 0.0, None)
    return BackgroundEstimate(
        background=background_est,
        corrected=corrected,
        radius=radius,
    )


def build_signal_support(
    corrected: np.ndarray,
    config: PunctaDeclumpConfig,
) -> SignalSupportResult:
    """Build a permissive boolean support map from background-corrected intensity."""
    corrected_arr = np.asarray(corrected, dtype=np.float64)
    positive = corrected_arr[corrected_arr > 0]
    if positive.size >= 16:
        robust_cap = float(np.percentile(positive, 95))
        robust_sample = positive[positive <= robust_cap]
    elif positive.size > 0:
        robust_sample = positive
    else:
        robust_sample = corrected_arr.ravel()

    noise_median = float(np.median(robust_sample))
    deviations = np.abs(robust_sample - noise_median)
    noise_mad = float(np.median(deviations))
    if noise_mad <= 0:
        noise_mad = float(np.std(corrected_arr))
    if noise_mad <= 0:
        noise_mad = 1.0

    threshold = noise_median + config.image_only_support_mad_multiplier * noise_mad
    support = corrected_arr >= threshold
    support = morphology.opening(support, morphology.disk(1))

    if config.image_only_support_min_object_area > 1:
        labeled = measure.label(support, connectivity=1)
        if labeled.max() > 0:
            regions = measure.regionprops(labeled)
            keep = np.zeros_like(support, dtype=bool)
            for region in regions:
                if region.area >= config.image_only_support_min_object_area:
                    keep[labeled == region.label] = True
            support = keep

    labels = measure.label(support, connectivity=1).astype(np.int32)
    return SignalSupportResult(
        support=support,
        labels=labels,
        threshold=threshold,
        noise_median=noise_median,
        noise_mad=noise_mad,
    )
