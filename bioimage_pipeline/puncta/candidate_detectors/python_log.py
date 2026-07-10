"""Python LoG / DoG image-level candidate detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.feature import peak_local_max

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.detector_cache import (
    evaluate_detector_cache,
    load_peak_table_cache,
    write_peak_table_cache,
)
from bioimage_pipeline.puncta.maxima_detector import MaximaDetector
from bioimage_pipeline.puncta.types import ImagePeakTable, PeakCandidate


class PythonLoGDetector:
    """Detect candidates once on the full image using DoG/LoG response."""

    name = "python_log"

    def detect(
        self,
        image: np.ndarray,
        *,
        config: PunctaDeclumpConfig,
        cache_dir: str | None = None,
        source_path: str | None = None,
        stem: str = "puncta",
    ) -> ImagePeakTable:
        source = Path(source_path) if source_path else None
        if cache_dir is not None:
            cache_path = Path(cache_dir)
            is_fresh, csv_path, _ = evaluate_detector_cache(
                source_path=source,
                cache_dir=cache_path,
                stem=stem,
                config=config,
            )
            if is_fresh:
                table = load_peak_table_cache(csv_path, self.name)
                table.cache_hit = True
                return table

        image_arr = np.asarray(image, dtype=np.float64)
        mask = np.ones(image_arr.shape, dtype=bool)
        maxima = MaximaDetector(config)
        response, method = maxima._build_response(image_arr)  # noqa: SLF001
        threshold_abs = maxima._compute_threshold_abs(response, mask)  # noqa: SLF001

        coords = peak_local_max(
            response,
            labels=mask.astype(np.int32),
            min_distance=max(1, config.min_peak_distance),
            threshold_abs=threshold_abs,
            exclude_border=False,
        )
        peaks = maxima._coords_to_peaks(coords, response)  # noqa: SLF001
        peaks = maxima._apply_relative_filters(peaks, response, mask)  # noqa: SLF001

        table = ImagePeakTable(
            peaks=[
                PeakCandidate(row=peak.row, col=peak.col, intensity=peak.intensity)
                for peak in peaks
            ],
            detector_name=self.name,
            method=method,
            cache_hit=False,
        )

        if cache_dir is not None:
            write_peak_table_cache(
                table,
                cache_dir=Path(cache_dir),
                stem=stem,
                config=config,
                source_path=source,
            )
        return table
