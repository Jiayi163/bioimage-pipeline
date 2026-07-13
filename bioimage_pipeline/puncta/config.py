"""Configuration for puncta declumping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ThresholdMethod = Literal["otsu", "manual", "adaptive", "sauvola", "external_mask"]
DiagnosticMode = Literal[
    "off",
    "summary",
    "balanced",
    "suspicious_only",
    "selected_objects",
    "all",
]
CandidateDetectorMode = Literal["python_log", "fiji_find_maxima", "trackmate", "comparison"]
FijiBatchMode = Literal["per_image", "batch"]


@dataclass
class PunctaDeclumpConfig:
    """Tunable parameters for Gaussian / GMM puncta declumping."""

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

    # Size gate / expected spot geometry
    expected_single_spot_diameter: float = 5.0
    single_spot_max_diameter: float = 7.0
    expected_single_spot_area_factor: float = 1.8
    elongation_gmm_threshold: float = 1.6
    eccentricity_gmm_threshold: float = 0.65
    solidity_gmm_threshold: float = 0.80

    # Local background correction
    background_ring_width: int = 3
    background_margin: int = 4

    # Maxima detection (defaults tuned for close/overlapping spots)
    smoothing_sigma: float = 0.4
    min_peak_distance: int = 2
    peak_noise_tolerance: float = 0.0
    peak_relative_prominence: float = 0.08
    peak_min_relative_height: float = 0.25
    use_dog_peaks: bool = True
    dog_sigma_small: float = 0.6
    dog_sigma_large: float = 1.4
    min_reliable_peaks_for_gmm: int = 2
    min_reliable_peaks_for_routing: int = 3

    # Single-component elliptical fitting
    fit_roi_radius: int = 5
    min_sigma: float = 0.5
    max_sigma: float = 4.0
    max_center_shift: float = 4.0
    min_amplitude: float = 10.0
    max_fit_residual: float | None = None
    max_fit_residual_relative: float = 0.25
    min_r_squared: float = 0.3

    # Balanced GMM triage thresholds (strong warnings — any one triggers GMM)
    gmm_trigger_r_squared: float = 0.6
    gmm_trigger_residual_relative: float = 0.18
    gmm_trigger_sigma_factor: float = 1.4
    gmm_trigger_area_factor: float = 3.0
    gmm_weak_fit_r_squared: float = 0.75
    # Legacy residual thresholds (used in under-split reporting)
    residual_gmm_r_squared: float = 0.75
    residual_gmm_relative: float = 0.12
    residual_gmm_sigma_factor: float = 1.4

    # GMM / mixture fitting
    gmm_max_components: int = 3
    gmm_max_components_large: int = 5
    gmm_try_component_delta: int = 1
    gmm_min_component_separation: float = 1.5
    gmm_merge_amplitude_ratio: float = 0.12
    gmm_bic_improvement_margin: float = 2.0
    gmm_aic_improvement_margin: float = 2.0
    large_object_diameter_threshold: float = 10.0
    gmm_multi_start_enabled: bool = True
    gmm_max_multi_starts: int = 20
    gmm_multi_start_max_nfev: int = 3000
    gmm_multi_start_separations: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    gmm_acceptance_min_separation: float = 1.5
    gmm_use_mixture_acceptance_separation: bool = True

    # Selective routing / detectors
    enable_selective_routing: bool = True
    candidate_detector: CandidateDetectorMode = "python_log"
    fiji_batch_mode: FijiBatchMode = "batch"
    detector_cache_dir: str | None = None
    force_redetect: bool = False
    ordinary_area_factor: float = 2.0
    enable_watershed_declump: bool = True
    enable_gmm: bool = True

    # Deduplication
    min_center_separation: float = 2.5

    # Fallback / diagnostics (PNG exports are expensive; CSV debug columns always exported)
    accept_brightest_on_fit_failure: bool = True
    diagnostic_mode: DiagnosticMode = "balanced"
    max_diagnostic_objects: int = 50
    log_progress: bool = True
    progress_log_interval: int = 50
    diagnostic_object_ids: tuple[int, ...] = ()
    diagnostic_low_r_squared: float = 0.5
    diagnostic_high_residual_relative: float = 0.20
    under_split_report_top_n: int = 50

    # Fiji TIFF exports
    export_fiji_tiffs: bool = True
    include_fallback_in_centers: bool = False
    fiji_center_disk_radius: int = 2
    fiji_label_disk_radius: float = 2.0

    # Deprecated: use diagnostic_mode instead. Kept for backward compatibility.
    export_diagnostics: bool | None = None
    diagnostic_residual_threshold: float = 0.20
    export_under_split_diagnostics: bool | None = None

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
        if self.gmm_max_components < 1:
            raise ValueError("gmm_max_components must be at least 1")
        if self.max_diagnostic_objects < 1:
            raise ValueError("max_diagnostic_objects must be at least 1")
        if self.diagnostic_mode not in (
            "off",
            "summary",
            "balanced",
            "suspicious_only",
            "selected_objects",
            "all",
        ):
            raise ValueError(f"Invalid diagnostic_mode: {self.diagnostic_mode}")
        if self.progress_log_interval < 1:
            raise ValueError("progress_log_interval must be at least 1")
        if self.candidate_detector not in (
            "python_log",
            "fiji_find_maxima",
            "trackmate",
            "comparison",
        ):
            raise ValueError(f"Invalid candidate_detector: {self.candidate_detector}")
        if self.fiji_batch_mode not in ("per_image", "batch"):
            raise ValueError(f"Invalid fiji_batch_mode: {self.fiji_batch_mode}")
        if self.ordinary_area_factor <= 0:
            raise ValueError("ordinary_area_factor must be positive")
        if self.min_reliable_peaks_for_routing < 2:
            raise ValueError("min_reliable_peaks_for_routing must be at least 2")
        if self.gmm_max_components_large < self.gmm_max_components:
            raise ValueError("gmm_max_components_large must be >= gmm_max_components")
        if self.gmm_max_multi_starts < 0:
            raise ValueError("gmm_max_multi_starts must be non-negative")
        if self.gmm_multi_start_max_nfev < 100:
            raise ValueError("gmm_multi_start_max_nfev must be at least 100")
        if self.gmm_acceptance_min_separation < 0:
            raise ValueError("gmm_acceptance_min_separation must be non-negative")
        if not self.gmm_multi_start_separations:
            raise ValueError("gmm_multi_start_separations must not be empty")

        # Backward compatibility for legacy boolean flags.
        if self.export_diagnostics is False and self.diagnostic_mode in (
            "balanced",
            "suspicious_only",
        ):
            self.diagnostic_mode = "off"
        if self.export_under_split_diagnostics is False and self.diagnostic_mode in (
            "balanced",
            "suspicious_only",
        ):
            self.diagnostic_mode = "summary"

    @property
    def expected_single_spot_area(self) -> float:
        radius = self.expected_single_spot_diameter / 2.0
        return float(3.141592653589793 * radius * radius)
