"""TrackMate LoG detector wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.detector_cache import (
    evaluate_detector_cache,
    load_peak_table_cache,
    write_peak_table_cache,
)
from bioimage_pipeline.puncta.trackmate_runner import (
    peaks_from_trackmate_csv,
    run_trackmate_on_image,
)
from bioimage_pipeline.puncta.types import ImagePeakTable


class TrackMateLoGDetector:
    name = "trackmate"

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
        cache_path = Path(cache_dir) if cache_dir else None
        csv_path = (cache_path / f"{stem}_trackmate.csv") if cache_path else None

        if cache_path is not None:
            is_fresh, cached_csv, _ = evaluate_detector_cache(
                source_path=source,
                cache_dir=cache_path,
                stem=f"{stem}_trackmate",
                config=config,
            )
            if is_fresh and cached_csv.is_file():
                table = load_peak_table_cache(cached_csv, self.name)
                table.cache_hit = True
                return table

        if source is not None and source.is_file():
            image_path = source
        else:
            if cache_path is None:
                raise ValueError("TrackMate detector requires cache_dir or source_path")
            cache_path.mkdir(parents=True, exist_ok=True)
            image_path = cache_path / f"{stem}_input.tif"
            tifffile.imwrite(image_path, np.asarray(image))

        if csv_path is None:
            csv_path = image_path.with_suffix(".trackmate.csv")

        run_trackmate_on_image(image_path, csv_path, config=config)
        table = peaks_from_trackmate_csv(csv_path)

        if cache_path is not None:
            write_peak_table_cache(
                table,
                cache_dir=cache_path,
                stem=f"{stem}_trackmate",
                config=config,
                source_path=source if source and source.is_file() else image_path,
            )
        return table
