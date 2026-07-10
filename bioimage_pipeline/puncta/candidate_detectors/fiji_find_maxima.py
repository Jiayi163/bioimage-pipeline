"""Fiji Find Maxima image-level detector."""

from __future__ import annotations

import csv
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import tifffile

from bioimage_pipeline.fiji_runner import run_fiji_macro
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.detector_cache import (
    evaluate_detector_cache,
    load_peak_table_cache,
    write_peak_table_cache,
)
from bioimage_pipeline.puncta.types import ImagePeakTable, PeakCandidate


class FijiFindMaximaDetector:
    name = "fiji_find_maxima"

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

        with tempfile.TemporaryDirectory(prefix="puncta_fiji_maxima_") as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / f"{stem}.tif"
            csv_path = tmp_path / f"{stem}_maxima.csv"
            tifffile.imwrite(image_path, np.asarray(image))
            macro_path = _write_find_maxima_macro(
                image_path=image_path,
                output_csv=csv_path,
                prominence=config.peak_relative_prominence,
            )
            result = run_fiji_macro(macro_path, headless=True, timeout=600.0)
            if not csv_path.is_file():
                raise RuntimeError(
                    f"Fiji Find Maxima failed: {result.stderr[:500]}"
                )
            peaks = _read_fiji_maxima_csv(csv_path)

        table = ImagePeakTable(
            peaks=peaks,
            detector_name=self.name,
            method="fiji_find_maxima",
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


def _read_fiji_maxima_csv(csv_path: Path) -> list[PeakCandidate]:
    peaks: list[PeakCandidate] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "X" in row and "Y" in row:
                peaks.append(
                    PeakCandidate(
                        row=float(row["Y"]),
                        col=float(row["X"]),
                        intensity=float(row.get("Value", row.get("Max", 0.0)) or 0.0),
                    )
                )
            elif "row" in row and "col" in row:
                peaks.append(
                    PeakCandidate(
                        row=float(row["row"]),
                        col=float(row["col"]),
                        intensity=float(row.get("intensity", 0.0)),
                    )
                )
    return peaks


def _write_find_maxima_macro(
    *,
    image_path: Path,
    output_csv: Path,
    prominence: float,
) -> Path:
    macro_dir = output_csv.parent / ".fiji_macros"
    macro_dir.mkdir(parents=True, exist_ok=True)
    macro_path = macro_dir / "find_maxima_generated.ijm"
    image_str = str(image_path).replace("\\", "/")
    csv_str = str(output_csv).replace("\\", "/")
    content = textwrap.dedent(
        f"""
        open("{image_str}");
        run("Find Maxima...", "prominence={prominence} exclude light output=[Single Points]");
        saveAs("Results", "{csv_str}");
        close();
        run("Close All");
        """
    ).strip()
    macro_path.write_text(content, encoding="utf-8")
    return macro_path
