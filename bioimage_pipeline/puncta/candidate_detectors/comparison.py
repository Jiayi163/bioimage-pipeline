"""Run all detectors for benchmarking without triple fitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bioimage_pipeline.puncta.candidate_detectors.fiji_find_maxima import FijiFindMaximaDetector
from bioimage_pipeline.puncta.candidate_detectors.python_log import PythonLoGDetector
from bioimage_pipeline.puncta.candidate_detectors.trackmate import TrackMateLoGDetector
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.detector_cache import write_peak_table_cache
from bioimage_pipeline.puncta.types import ImagePeakTable


class ComparisonDetector:
    """Run python_log, fiji_find_maxima, and trackmate; return python_log peaks."""

    name = "comparison"

    def detect(
        self,
        image: np.ndarray,
        *,
        config: PunctaDeclumpConfig,
        cache_dir: str | None = None,
        source_path: str | None = None,
        stem: str = "puncta",
    ) -> ImagePeakTable:
        python_table = PythonLoGDetector().detect(
            image,
            config=config,
            cache_dir=cache_dir,
            source_path=source_path,
            stem=f"{stem}_python_log",
        )

        if cache_dir is not None:
            cache_path = Path(cache_dir)
            try:
                fiji_table = FijiFindMaximaDetector().detect(
                    image,
                    config=config,
                    cache_dir=str(cache_path),
                    source_path=source_path,
                    stem=f"{stem}_fiji_find_maxima",
                )
                write_peak_table_cache(
                    fiji_table,
                    cache_dir=cache_path,
                    stem=f"{stem}_comparison_fiji",
                    config=config,
                    source_path=Path(source_path) if source_path else None,
                )
            except Exception:
                pass

            try:
                tm_table = TrackMateLoGDetector().detect(
                    image,
                    config=config,
                    cache_dir=str(cache_path),
                    source_path=source_path,
                    stem=f"{stem}_trackmate",
                )
                write_peak_table_cache(
                    tm_table,
                    cache_dir=cache_path,
                    stem=f"{stem}_comparison_trackmate",
                    config=config,
                    source_path=Path(source_path) if source_path else None,
                )
            except Exception:
                pass

        python_table.detector_name = self.name
        python_table.method = "comparison_primary_python_log"
        return python_table
