"""Configuration for puncta declumping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ThresholdMethod = Literal["otsu", "manual", "adaptive", "sauvola", "external_mask"]


@dataclass
class PunctaDeclumpConfig:
    """Tunable parameters for size-gated puncta declumping."""

    # Mask generation
    threshold_method: ThresholdMethod = "otsu"
    manual_threshold_value: float = 100.0
    adaptive_block_size: int = 51
    adaptive_offset: float = 0.0
    sauvola_block_size: int = 51
    sauvola_k: float = 0.2
    min_object_area: int = 4
    max_object_area: int = 10_000
    fill_holes: bool = True
    clear_border: bool = True

    # Size gate
    expected_single_spot_diameter: float = 5.0
    single_spot_max_diameter: float = 7.0

    # Maxima detection (large / clumped objects)
    smoothing_sigma: float = 0.75
    min_peak_distance: int = 3
    peak_noise_tolerance: float = 0.0

    # Gaussian fitting
    fit_roi_radius: int = 5
    min_sigma: float = 0.5
    max_sigma: float = 4.0
    max_center_shift: float = 4.0
    min_amplitude: float = 10.0
    max_fit_residual: float | None = None
    max_fit_residual_relative: float = 0.25

    # Deduplication
    min_center_separation: float = 3.0

    # Single-object path: accept brightest pixel when fit fails
    accept_brightest_on_fit_failure: bool = True

    def __post_init__(self) -> None:
        if self.single_spot_max_diameter <= 0:
            raise ValueError("single_spot_max_diameter must be positive")
        if self.fit_roi_radius < 1:
            raise ValueError("fit_roi_radius must be at least 1")
        if self.min_sigma <= 0 or self.max_sigma <= self.min_sigma:
            raise ValueError("Invalid sigma bounds")
        if self.min_center_separation < 0:
            raise ValueError("min_center_separation must be non-negative")
        if self.max_fit_residual_relative <= 0:
            raise ValueError("max_fit_residual_relative must be positive")
