"""Pre-fit routing: ordinary vs suspicious mask objects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.peak_assignment import count_reliable_assigned_peaks
from bioimage_pipeline.puncta.types import ObjectInfo, PeakCandidate

ObjectRoute = Literal["ordinary_single", "suspicious"]


@dataclass(frozen=True)
class RouteDecision:
    """Routing outcome for one connected object."""

    route: ObjectRoute
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RoutingSummary:
    """Aggregate routing counts for logging."""

    ordinary: int
    suspicious: int
    reason_counts: dict[str, int]


class ObjectRouter:
    """Classify objects before expensive candidate detection or fitting."""

    def __init__(self, config: PunctaDeclumpConfig) -> None:
        self.config = config

    def classify(
        self,
        obj: ObjectInfo,
        assigned_peaks: list[PeakCandidate],
        *,
        manual_ids: tuple[int, ...] | None = None,
    ) -> RouteDecision:
        manual_ids = manual_ids if manual_ids is not None else self.config.diagnostic_object_ids
        reasons: list[str] = []
        cfg = self.config

        if obj.label in manual_ids:
            reasons.append("manual_id")
            return RouteDecision(route="suspicious", reasons=tuple(reasons))

        separated_peak_count = count_reliable_assigned_peaks(assigned_peaks, cfg)
        oversized = obj.equivalent_diameter > cfg.single_spot_max_diameter

        if separated_peak_count >= cfg.min_reliable_peaks_for_routing:
            reasons.append(f"assigned_peaks={separated_peak_count}")
            return RouteDecision(route="suspicious", reasons=tuple(reasons))

        if separated_peak_count >= cfg.min_reliable_peaks_for_gmm:
            shape_suspicious = self._shape_suspicious(obj)
            if oversized or shape_suspicious:
                reasons.append(f"two_peaks_separated={separated_peak_count}")
                if oversized:
                    reasons.append("two_peaks_oversized")
                if shape_suspicious:
                    reasons.append("two_peaks_irregular_shape")
                return RouteDecision(route="suspicious", reasons=tuple(reasons))

        if oversized:
            reasons.append("large_diameter")
            return RouteDecision(route="suspicious", reasons=tuple(reasons))

        max_ordinary_area = self._max_ordinary_area()
        if obj.area > max_ordinary_area:
            reasons.append("large_area")
            return RouteDecision(route="suspicious", reasons=tuple(reasons))

        if self._shape_suspicious(obj) and (
            separated_peak_count >= cfg.min_reliable_peaks_for_gmm or oversized
        ):
            if obj.solidity < cfg.solidity_gmm_threshold:
                reasons.append("low_solidity")
            if obj.elongation >= cfg.elongation_gmm_threshold:
                reasons.append("elongated")
            if obj.eccentricity >= cfg.eccentricity_gmm_threshold:
                reasons.append("eccentric")
            return RouteDecision(route="suspicious", reasons=tuple(reasons))

        return RouteDecision(route="ordinary_single", reasons=tuple(reasons))

    def summarize(
        self,
        objects: list[ObjectInfo],
        assigned_by_label: dict[int, list[PeakCandidate]],
    ) -> RoutingSummary:
        reason_counts: dict[str, int] = {}
        ordinary = 0
        suspicious = 0
        for obj in objects:
            decision = self.classify(obj, assigned_by_label.get(obj.label, []))
            if decision.route == "ordinary_single":
                ordinary += 1
            else:
                suspicious += 1
            for reason in decision.reasons:
                key = reason.split("=")[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
        return RoutingSummary(
            ordinary=ordinary,
            suspicious=suspicious,
            reason_counts=reason_counts,
        )

    def _max_ordinary_area(self) -> float:
        radius = self.config.single_spot_max_diameter / 2.0
        return float(math.pi * radius * radius * self.config.ordinary_area_factor)

    def _shape_suspicious(self, obj: ObjectInfo) -> bool:
        cfg = self.config
        return (
            obj.solidity < cfg.solidity_gmm_threshold
            or obj.elongation >= cfg.elongation_gmm_threshold
            or obj.eccentricity >= cfg.eccentricity_gmm_threshold
        )
