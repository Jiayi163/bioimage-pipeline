"""Candidate detector protocol and factory."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.types import ImagePeakTable


class CandidateDetector(Protocol):
    name: str

    def detect(
        self,
        image: np.ndarray,
        *,
        config: PunctaDeclumpConfig,
        cache_dir: str | None = None,
        source_path: str | None = None,
        stem: str = "puncta",
    ) -> ImagePeakTable: ...


def get_detector(config: PunctaDeclumpConfig) -> CandidateDetector:
    from bioimage_pipeline.puncta.candidate_detectors.comparison import ComparisonDetector
    from bioimage_pipeline.puncta.candidate_detectors.fiji_find_maxima import FijiFindMaximaDetector
    from bioimage_pipeline.puncta.candidate_detectors.python_log import PythonLoGDetector
    from bioimage_pipeline.puncta.candidate_detectors.trackmate import TrackMateLoGDetector

    mapping = {
        "python_log": PythonLoGDetector,
        "fiji_find_maxima": FijiFindMaximaDetector,
        "trackmate": TrackMateLoGDetector,
        "comparison": ComparisonDetector,
    }
    factory = mapping.get(config.candidate_detector)
    if factory is None:
        raise ValueError(f"Unknown candidate_detector: {config.candidate_detector}")
    return factory()
