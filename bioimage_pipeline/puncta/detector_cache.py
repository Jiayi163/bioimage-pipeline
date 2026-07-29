"""Cache image-level candidate detector outputs by path, mtime, and settings."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.types import ImagePeakTable, PeakCandidate

CACHE_MTIME_TOLERANCE_SECONDS = 2.0
META_SUFFIX = ".candidates.meta.json"
CSV_SUFFIX = "_candidates.csv"


def detector_settings_hash(config: PunctaDeclumpConfig) -> str:
    """Stable hash of detector-relevant settings."""
    payload = {
        "candidate_detector": config.candidate_detector,
        "expected_single_spot_diameter": config.expected_single_spot_diameter,
        "smoothing_sigma": config.smoothing_sigma,
        "min_peak_distance": config.min_peak_distance,
        "peak_noise_tolerance": config.peak_noise_tolerance,
        "peak_relative_prominence": config.peak_relative_prominence,
        "peak_min_relative_height": config.peak_min_relative_height,
        "use_dog_peaks": config.use_dog_peaks,
        "dog_sigma_small": config.dog_sigma_small,
        "dog_sigma_large": config.dog_sigma_large,
        "trackmate_radius": config.expected_single_spot_diameter / 2.0,
        "trackmate_threshold": config.peak_min_relative_height,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def cache_paths(cache_dir: Path, stem: str) -> tuple[Path, Path]:
    csv_path = cache_dir / f"{stem}{CSV_SUFFIX}"
    meta_path = cache_dir / f"{stem}{META_SUFFIX}"
    return csv_path, meta_path


def evaluate_detector_cache(
    *,
    source_path: Path | None,
    cache_dir: Path,
    stem: str,
    config: PunctaDeclumpConfig,
) -> tuple[bool, Path, Path]:
    """Return (is_fresh, csv_path, meta_path)."""
    csv_path, meta_path = cache_paths(cache_dir, stem)
    if config.force_redetect:
        return False, csv_path, meta_path
    if not meta_path.is_file() or not csv_path.is_file():
        return False, csv_path, meta_path

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, csv_path, meta_path

    if meta.get("detector_name") != config.candidate_detector:
        return False, csv_path, meta_path
    if meta.get("settings_hash") != detector_settings_hash(config):
        return False, csv_path, meta_path

    if source_path is not None and source_path.is_file():
        input_mtime = source_path.stat().st_mtime_ns
        stored_mtime = meta.get("source_mtime_ns")
        if stored_mtime != input_mtime:
            return False, csv_path, meta_path
        output_mtime = csv_path.stat().st_mtime
        input_mtime_s = input_mtime / 1e9
        if output_mtime + CACHE_MTIME_TOLERANCE_SECONDS < input_mtime_s:
            return False, csv_path, meta_path

    return True, csv_path, meta_path


def write_peak_table_cache(
    peak_table: ImagePeakTable,
    *,
    cache_dir: Path | str,
    stem: str,
    config: PunctaDeclumpConfig,
    source_path: Path | None = None,
) -> Path:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = cache_paths(cache_root, stem)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, peak_table)
    meta: dict[str, Any] = {
        "detector_name": peak_table.detector_name,
        "settings_hash": detector_settings_hash(config),
        "source_path": str(source_path) if source_path else None,
        "source_mtime_ns": source_path.stat().st_mtime_ns if source_path and source_path.is_file() else None,
        "peak_count": len(peak_table.peaks),
        "written_at": time.time(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return csv_path


def load_peak_table_cache(csv_path: Path, detector_name: str) -> ImagePeakTable:
    peaks: list[PeakCandidate] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            peaks.append(
                PeakCandidate(
                    row=float(row["row"]),
                    col=float(row["col"]),
                    intensity=float(row.get("intensity", 0.0)),
                )
            )
    return ImagePeakTable(peaks=peaks, detector_name=detector_name, cache_hit=True)


def _write_csv(csv_path: Path, peak_table: ImagePeakTable) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row", "col", "intensity", "quality", "radius"])
        writer.writeheader()
        for peak in peak_table.peaks:
            writer.writerow(
                {
                    "row": peak.row,
                    "col": peak.col,
                    "intensity": peak.intensity,
                    "quality": peak.intensity,
                    "radius": "",
                }
            )
