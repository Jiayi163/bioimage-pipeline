"""Conservative local peak recovery for fast-path objects with no assigned peaks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.types import (
    ModelSelectionDebug,
    ObjectInfo,
    ObjectPatch,
    PeakCandidate,
    PeakDetectionResult,
)

PeakSource = Literal[
    "assigned_global",
    "recovered_local_detector",
    "recovered_masked_argmax",
    "fallback",
]


@dataclass
class LocalPeakRecoveryAttempt:
    """Outcome of one object-level MaximaDetector / masked-argmax recovery."""

    attempted: bool
    success: bool
    raw_count: int
    filtered_count: int
    peak_source: PeakSource
    peaks: list[PeakCandidate] = field(default_factory=list)
    detection: PeakDetectionResult | None = None
    patch: ObjectPatch | None = None


@dataclass
class LocalPeakRecoveryStats:
    """Aggregate cheap recovery timing for one pipeline run."""

    attempts: int = 0
    success: int = 0
    one_peak: int = 0
    multi_peak: int = 0
    total_time: float = 0.0

    @property
    def mean_time(self) -> float:
        return self.total_time / max(self.attempts, 1)

    def to_dict(self) -> dict[str, object]:
        return {
            "local_peak_recovery_attempts": self.attempts,
            "local_peak_recovery_success": self.success,
            "local_peak_recovery_one_peak": self.one_peak,
            "local_peak_recovery_multi_peak": self.multi_peak,
            "local_peak_recovery_time": self.total_time,
            "local_peak_recovery_mean_time": self.mean_time,
        }


def is_tiny_recovery_object(obj: ObjectInfo, config: PunctaDeclumpConfig) -> bool:
    """True when masked-argmax fallback is allowed (tiny objects only)."""
    return (
        float(obj.area) <= config.local_peak_recovery_tiny_max_area
        and float(obj.equivalent_diameter) <= config.local_peak_recovery_tiny_max_diameter
    )


def masked_argmax_seed(patch: ObjectPatch) -> PeakCandidate | None:
    """Brightest masked pixel in the object patch, in global coordinates."""
    image = patch.raw if patch.raw is not None else patch.corrected
    mask = np.asarray(patch.object_mask, dtype=bool)
    if image is None or not mask.any():
        return None
    masked = np.where(mask, image, -np.inf)
    flat = int(np.argmax(masked))
    row, col = np.unravel_index(flat, image.shape)
    if not bool(mask[int(row), int(col)]):
        return None
    return PeakCandidate(
        row=float(row) + patch.row_offset,
        col=float(col) + patch.col_offset,
        intensity=float(image[int(row), int(col)]),
    )


def apply_recovery_to_debug(
    debug: ModelSelectionDebug,
    recovery: LocalPeakRecoveryAttempt,
    *,
    peak_source: PeakSource | None = None,
) -> None:
    debug.local_peak_recovery_attempted = recovery.attempted
    debug.local_peak_recovery_success = recovery.success
    debug.local_peak_recovery_raw_count = recovery.raw_count
    debug.local_peak_recovery_filtered_count = recovery.filtered_count
    debug.peak_source = peak_source or recovery.peak_source


def finalize_recovery(
    detection: PeakDetectionResult,
    patch: ObjectPatch,
    obj: ObjectInfo,
    config: PunctaDeclumpConfig,
) -> LocalPeakRecoveryAttempt:
    """Choose recovered detector peaks, tiny masked-argmax, or no recovery."""
    raw_count = len(detection.raw_peaks)
    filtered_count = len(detection.filtered_peaks)
    if filtered_count >= 1:
        return LocalPeakRecoveryAttempt(
            attempted=True,
            success=True,
            raw_count=raw_count,
            filtered_count=filtered_count,
            peak_source="recovered_local_detector",
            peaks=list(detection.filtered_peaks),
            detection=detection,
            patch=patch,
        )
    if is_tiny_recovery_object(obj, config):
        seed = masked_argmax_seed(patch)
        if seed is not None:
            return LocalPeakRecoveryAttempt(
                attempted=True,
                success=True,
                raw_count=raw_count,
                filtered_count=0,
                peak_source="recovered_masked_argmax",
                peaks=[seed],
                detection=detection,
                patch=patch,
            )
    return LocalPeakRecoveryAttempt(
        attempted=True,
        success=False,
        raw_count=raw_count,
        filtered_count=filtered_count,
        peak_source="fallback",
        peaks=[],
        detection=detection,
        patch=patch,
    )
