"""Data types for puncta declumping results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

DetectionPath = Literal["single", "declump", "fallback"]


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


@dataclass
class GaussianFitResult:
    """Result of a 2D circular Gaussian fit."""

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


@dataclass
class PeakCandidate:
    """A local intensity maximum used as a Gaussian fit seed."""

    row: float
    col: float
    intensity: float


@dataclass
class PunctumCandidate:
    """One punctum detection candidate with optional fit and acceptance status."""

    object_id: int
    candidate_id: int
    path: DetectionPath
    initial_row: float
    initial_col: float
    fitted_row: float | None = None
    fitted_col: float | None = None
    center_shift: float | None = None
    sigma: float | None = None
    width_fwhm: float | None = None
    amplitude: float | None = None
    background: float | None = None
    residual_rmse: float | None = None
    accepted: bool = False
    rejection_reason: str | None = None
    warning: str | None = None

    @property
    def final_row(self) -> float:
        return self.fitted_row if self.fitted_row is not None else self.initial_row

    @property
    def final_col(self) -> float:
        return self.fitted_col if self.fitted_col is not None else self.initial_col


@dataclass
class DeclumpSummary:
    """Aggregate counts from a declumping run."""

    total_mask_objects: int = 0
    small_single_objects: int = 0
    large_clumped_objects: int = 0
    total_candidates: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    fallback_objects: int = 0


@dataclass
class DeclumpResult:
    """Full output of puncta declumping."""

    candidates: list[PunctumCandidate] = field(default_factory=list)
    summary: DeclumpSummary = field(default_factory=DeclumpSummary)
    mask: np.ndarray | None = None
    labels: np.ndarray | None = None
    objects: list[ObjectInfo] = field(default_factory=list)
    threshold_metadata: dict[str, object] = field(default_factory=dict)

    @property
    def accepted(self) -> list[PunctumCandidate]:
        return [c for c in self.candidates if c.accepted]

    @property
    def rejected(self) -> list[PunctumCandidate]:
        return [c for c in self.candidates if not c.accepted]
