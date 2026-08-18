"""Peak detection and validation for image-only puncta mode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.feature import peak_local_max

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.maxima_detector import MaximaDetector
from bioimage_pipeline.puncta.types import PeakCandidate, RejectedPeak


@dataclass(frozen=True)
class ImageOnlyPeakResult:
    """Raw and validated peaks from image-only detection."""

    raw_peaks: list[PeakCandidate]
    validated_peaks: list[PeakCandidate]
    rejected_peaks: list[RejectedPeak]
    response_method: str


def detect_raw_peaks(
    corrected: np.ndarray,
    support: np.ndarray,
    config: PunctaDeclumpConfig,
) -> tuple[list[PeakCandidate], str]:
    """Detect DoG peaks on corrected image within the support mask."""
    maxima = MaximaDetector(config)
    response, method = maxima._build_response(corrected)  # noqa: SLF001
    support_arr = np.asarray(support, dtype=bool)
    threshold_abs = maxima._compute_threshold_abs(response, support_arr)  # noqa: SLF001

    coords = peak_local_max(
        response,
        labels=support_arr.astype(np.int32),
        min_distance=max(1, config.min_peak_distance),
        threshold_abs=threshold_abs,
        exclude_border=False,
    )
    peaks = maxima._coords_to_peaks(coords, response)  # noqa: SLF001
    peaks = maxima._apply_relative_filters(peaks, response, support_arr)  # noqa: SLF001
    return peaks, method


def _local_snr(
    raw: np.ndarray,
    row: int,
    col: int,
    *,
    window_radius: int = 5,
) -> float:
    """Compute local SNR in an (2*window_radius+1)^2 window around a peak."""
    height, width = raw.shape
    r0 = max(0, row - window_radius)
    r1 = min(height, row + window_radius + 1)
    c0 = max(0, col - window_radius)
    c1 = min(width, col + window_radius + 1)
    window = raw[r0:r1, c0:c1].astype(np.float64)
    if window.size == 0:
        return 0.0
    local_bg = float(np.median(window))
    local_mad = float(np.median(np.abs(window - local_bg)))
    if local_mad <= 0:
        local_mad = float(np.std(window))
    if local_mad <= 0:
        local_mad = 1.0
    peak_value = float(raw[row, col])
    return (peak_value - local_bg) / local_mad


def validate_peaks(
    raw: np.ndarray,
    corrected: np.ndarray,
    support: np.ndarray,
    raw_peaks: list[PeakCandidate],
    config: PunctaDeclumpConfig,
    *,
    response_method: str = "",
) -> ImageOnlyPeakResult:
    """Apply image-only validation rules to raw DoG peaks."""
    support_arr = np.asarray(support, dtype=bool)
    maxima = MaximaDetector(config)
    response, method = maxima._build_response(corrected)  # noqa: SLF001
    if not response_method:
        response_method = method

    masked_response = response[support_arr]
    peak_max = float(np.max(masked_response)) if masked_response.size else 0.0
    min_height = config.peak_min_relative_height * peak_max if peak_max > 0 else 0.0

    validated: list[PeakCandidate] = []
    rejected: list[RejectedPeak] = []

    sorted_peaks = sorted(raw_peaks, key=lambda p: p.intensity, reverse=True)
    for peak in sorted_peaks:
        row = int(round(peak.row))
        col = int(round(peak.col))
        height, width = raw.shape
        if row < 0 or col < 0 or row >= height or col >= width:
            rejected.append(
                RejectedPeak(peak.row, peak.col, peak.intensity, "out_of_bounds")
            )
            continue
        if not support_arr[row, col]:
            rejected.append(
                RejectedPeak(peak.row, peak.col, peak.intensity, "outside_support")
            )
            continue
        dog_value = float(response[row, col])
        if peak_max > 0 and dog_value < min_height:
            rejected.append(
                RejectedPeak(peak.row, peak.col, peak.intensity, "low_relative_height")
            )
            continue
        snr = _local_snr(raw, row, col)
        if snr < config.image_only_min_snr:
            rejected.append(
                RejectedPeak(
                    peak.row,
                    peak.col,
                    peak.intensity,
                    f"low_snr:{snr:.2f}",
                )
            )
            continue
        too_close = False
        min_validation_sep = max(1.0, float(config.min_peak_distance))
        for accepted in validated:
            dist = float(np.hypot(peak.row - accepted.row, peak.col - accepted.col))
            if dist < min_validation_sep:
                too_close = True
                rejected.append(
                    RejectedPeak(
                        peak.row,
                        peak.col,
                        peak.intensity,
                        f"too_close:{dist:.2f}",
                    )
                )
                break
        if too_close:
            continue
        validated.append(peak)

    return ImageOnlyPeakResult(
        raw_peaks=list(raw_peaks),
        validated_peaks=validated,
        rejected_peaks=rejected,
        response_method=response_method,
    )


def detect_and_validate_peaks(
    raw: np.ndarray,
    corrected: np.ndarray,
    support: np.ndarray,
    config: PunctaDeclumpConfig,
) -> ImageOnlyPeakResult:
    """Detect raw peaks then apply image-only validation."""
    raw_peaks, method = detect_raw_peaks(corrected, support, config)
    return validate_peaks(
        raw,
        corrected,
        support,
        raw_peaks,
        config,
        response_method=method,
    )
