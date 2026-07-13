"""Data types for puncta declumping results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

DetectionPath = Literal["single", "fast_single", "declump", "gmm", "fallback"]
FitStatus = Literal[
    "fit_ok",
    "fit_failed_fallback",
    "rejected_bad_fit",
    "rejected_duplicate",
    "rejected_outside_mask",
    "rejected_low_amplitude",
    "rejected_other",
]


@dataclass(frozen=True)
class ObjectInfo:
    """One connected foreground object from the threshold mask."""

    label: int
    area: float
    equivalent_diameter: float
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    brightest_row: float
    brightest_col: float
    brightest_intensity: float
    eccentricity: float = 0.0
    solidity: float = 1.0
    major_axis_length: float = 0.0
    minor_axis_length: float = 0.0
    elongation: float = 1.0


@dataclass
class ObjectPatch:
    """A background-corrected image patch for one mask object."""

    object_id: int
    row_offset: int
    col_offset: int
    corrected: np.ndarray
    object_mask: np.ndarray
    background_level: float
    global_bbox: tuple[int, int, int, int]
    raw: np.ndarray | None = None


@dataclass
class PeakCandidate:
    """A local intensity maximum used as a Gaussian fit seed."""

    row: float
    col: float
    intensity: float


@dataclass
class ImagePeakTable:
    """Image-level candidate coordinates from one detector run."""

    peaks: list[PeakCandidate]
    detector_name: str = "python_log"
    method: str = ""
    cache_hit: bool = False


@dataclass
class PeakDetectionResult:
    """Raw and filtered local maxima for one object ROI."""

    raw_peaks: list[PeakCandidate]
    filtered_peaks: list[PeakCandidate]
    method: str = "peak_local_max"


@dataclass
class GaussianComponent:
    """One fitted Gaussian component in global image coordinates."""

    component_id: int
    initial_row: float
    initial_col: float
    fitted_row: float
    fitted_col: float
    sigma_row: float
    sigma_col: float
    amplitude: float
    background: float
    residual_rmse: float
    residual_relative: float
    r_squared: float
    model_score: float
    n_components_in_model: int
    fit_succeeded: bool
    fit_error: str | None = None
    predicted_patch: np.ndarray | None = None
    residual_patch: np.ndarray | None = None

    @property
    def sigma(self) -> float:
        return float(np.sqrt(self.sigma_row * self.sigma_col))

    @property
    def width_fwhm_row(self) -> float:
        return self.sigma_row * 2.355

    @property
    def width_fwhm_col(self) -> float:
        return self.sigma_col * 2.355

    @property
    def center_shift(self) -> float:
        return float(
            np.hypot(self.fitted_row - self.initial_row, self.fitted_col - self.initial_col)
        )


@dataclass
class GmmInitAttemptDiagnostics:
    """One multi-start initialization attempt for production diagnostics."""

    strategy: str
    converged: bool
    post_merge_component_count: int
    merge_collapsed: bool
    merge_notes: list[str]
    fitted_centers: list[tuple[float, float]]
    fitted_amplitudes: list[float]
    fitted_sigma_x: list[float]
    fitted_sigma_y: list[float]
    rss: float | None
    bic: float | None
    selected: bool = False
    optimizer_runtime_s: float | None = None
    n_optimizer_evaluations: int | None = None


@dataclass
class MixtureFitResult:
    """Joint Gaussian mixture fit for one object ROI."""

    components: list[GaussianComponent]
    n_components: int
    background: float
    residual_rmse: float
    r_squared: float
    aic: float
    bic: float
    model_score: float
    fit_succeeded: bool
    fit_error: str | None = None
    predicted_patch: np.ndarray | None = None
    residual_patch: np.ndarray | None = None
    merge_notes: list[str] = field(default_factory=list)
    winning_init_strategy: str | None = None
    multi_start_attempts: int | None = None
    multi_start_converged: int | None = None
    init_attempts: list[GmmInitAttemptDiagnostics] = field(default_factory=list)
    optimizer_runtime_s: float | None = None
    optimizer_nfev: int | None = None
    early_stopped: bool = False
    search_mode: str | None = None


@dataclass
class ModelSelectionDebug:
    """Object-level debug metadata for GMM routing and model choice."""

    gmm_trigger_reasons: list[str] = field(default_factory=list)
    n_raw_local_maxima: int = 0
    n_filtered_local_maxima: int = 0
    tried_gmm: bool = False
    gmm_candidate_components: int = 0
    one_gaussian_r_squared: float | None = None
    one_gaussian_residual_relative: float | None = None
    one_gaussian_sigma: float | None = None
    one_gaussian_sigma_row: float | None = None
    one_gaussian_sigma_col: float | None = None
    one_gaussian_amplitude: float | None = None
    one_gaussian_center_shift: float | None = None
    best_gmm_r_squared: float | None = None
    best_gmm_residual_relative: float | None = None
    best_gmm_n_components: int | None = None
    gmm_winning_init_strategy: str | None = None
    gmm_multi_start_attempts: int | None = None
    gmm_multi_start_converged: int | None = None
    gmm_fitted_center_distance_px: float | None = None
    gmm_bic_delta_vs_single: float | None = None
    gmm_aic_delta_vs_single: float | None = None
    gmm_acceptance_min_separation_px: float | None = None
    gmm_search_mode: str | None = None
    gmm_spurious_split_rejected: bool = False
    model_selection_reason: str = ""
    rejected_component_reason: str | None = None
    single_path_reason: str | None = None
    under_split_suspect: bool = False
    under_split_reasons: list[str] = field(default_factory=list)


@dataclass
class GaussianFitResult:
    """Legacy circular Gaussian fit result (kept for compatibility)."""

    fitted_row: float
    fitted_col: float
    sigma: float
    width_fwhm: float
    amplitude: float
    background: float
    residual_rmse: float
    roi_touches_edge: bool
    fit_succeeded: bool
    fit_error: str | None = None
    r_squared: float | None = None
    sigma_row: float | None = None
    sigma_col: float | None = None


@dataclass
class PunctumCandidate:
    """One punctum detection candidate with fit metadata."""

    object_id: int
    candidate_id: int
    component_id: int
    path: DetectionPath
    fit_status: FitStatus
    initial_row: float
    initial_col: float
    fitted_row: float | None = None
    fitted_col: float | None = None
    center_shift: float | None = None
    sigma: float | None = None
    sigma_row: float | None = None
    sigma_col: float | None = None
    width_fwhm: float | None = None
    amplitude: float | None = None
    background: float | None = None
    residual_rmse: float | None = None
    residual_relative: float | None = None
    r_squared: float | None = None
    model_score: float | None = None
    n_components_in_model: int | None = None
    accepted: bool = False
    rejection_reason: str | None = None
    warning: str | None = None
    # Object shape / routing debug
    object_area: float | None = None
    object_equivalent_diameter: float | None = None
    object_eccentricity: float | None = None
    object_solidity: float | None = None
    object_major_axis_length: float | None = None
    object_minor_axis_length: float | None = None
    object_elongation: float | None = None
    gmm_trigger_reasons: str | None = None
    n_raw_local_maxima: int | None = None
    n_filtered_local_maxima: int | None = None
    tried_gmm: bool | None = None
    gmm_candidate_components: int | None = None
    one_gaussian_r_squared: float | None = None
    one_gaussian_residual_relative: float | None = None
    best_gmm_r_squared: float | None = None
    best_gmm_residual_relative: float | None = None
    best_gmm_n_components: int | None = None
    model_selection_reason: str | None = None
    rejected_component_reason: str | None = None
    under_split_suspect: bool | None = None
    under_split_reasons: str | None = None
    gmm_winning_init_strategy: str | None = None
    gmm_duplicate_threshold_px: float | None = None
    gmm_duplicate_distance_px: float | None = None
    gmm_bic_delta_vs_single: float | None = None
    gmm_aic_delta_vs_single: float | None = None
    gmm_search_mode: str | None = None
    gmm_spurious_split_rejected: bool | None = None
    gmm_multi_start_attempts: int | None = None
    gmm_multi_start_converged: int | None = None

    @property
    def has_gaussian_fit(self) -> bool:
        return self.fit_status == "fit_ok" and self.fitted_row is not None

    @property
    def final_row(self) -> float:
        if self.has_gaussian_fit:
            return float(self.fitted_row)
        return self.initial_row

    @property
    def final_col(self) -> float:
        if self.has_gaussian_fit:
            return float(self.fitted_col)
        return self.initial_col


@dataclass
class DeclumpSummary:
    """Aggregate counts from a declumping run."""

    total_mask_objects: int = 0
    single_path_objects: int = 0
    gmm_path_objects: int = 0
    fallback_objects: int = 0
    total_candidates: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    fit_ok_count: int = 0
    fit_failed_fallback_count: int = 0
    under_split_suspect_objects: int = 0
    gmm_triggered_objects: int = 0
    gmm_accepted_objects: int = 0
    fast_path_objects: int = 0
    suspicious_objects: int = 0
    fitted_objects: int = 0
    diagnostics_exported: int = 0
    total_runtime_seconds: float = 0.0

    @property
    def small_single_objects(self) -> int:
        return self.single_path_objects

    @property
    def large_clumped_objects(self) -> int:
        return self.gmm_path_objects


@dataclass
class DeclumpResult:
    """Full output of puncta declumping."""

    candidates: list[PunctumCandidate] = field(default_factory=list)
    summary: DeclumpSummary = field(default_factory=DeclumpSummary)
    mask: np.ndarray | None = None
    labels: np.ndarray | None = None
    objects: list[ObjectInfo] = field(default_factory=list)
    threshold_metadata: dict[str, object] = field(default_factory=dict)
    diagnostic_artifacts: list[str] = field(default_factory=list)
    under_split_report: list[dict[str, object]] = field(default_factory=list)
    timing: dict[str, object] = field(default_factory=dict)
    peak_table: ImagePeakTable | None = None

    @property
    def accepted(self) -> list[PunctumCandidate]:
        return [c for c in self.candidates if c.accepted]

    @property
    def rejected(self) -> list[PunctumCandidate]:
        return [c for c in self.candidates if not c.accepted]

    @property
    def gaussian_fitted(self) -> list[PunctumCandidate]:
        return [c for c in self.candidates if c.fit_status == "fit_ok" and c.accepted]
