"""Stage timing metrics for puncta declumping."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class PunctaTimingMetrics:
    """Per-run timing breakdown for performance analysis."""

    preprocessing_time: float = 0.0
    connected_component_time: float = 0.0
    candidate_detection_time: float = 0.0
    gaussian_fit_time: float = 0.0
    watershed_time: float = 0.0
    diagnostic_export_time: float = 0.0
    total_time: float = 0.0
    number_of_objects: int = 0
    number_of_suspicious_objects: int = 0
    number_of_fitted_objects: int = 0
    number_of_fast_path_objects: int = 0
    detector_name: str = "python_log"
    cache_hit: bool = False
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(self.extra)
        return payload
