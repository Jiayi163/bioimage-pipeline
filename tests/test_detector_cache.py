"""Tests for puncta detector cache helpers."""

from __future__ import annotations

from pathlib import Path

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.detector_cache import write_peak_table_cache
from bioimage_pipeline.puncta.types import ImagePeakTable, PeakCandidate


def _default_config() -> PunctaDeclumpConfig:
    return PunctaDeclumpConfig(
        threshold_method="manual",
        manual_threshold_value=50.0,
        candidate_detector="python_log",
    )


def test_write_peak_table_cache_creates_missing_cache_directory(tmp_path: Path) -> None:
    cache_dir = tmp_path / "results" / "false_split_case" / ".puncta_cache"
    assert not cache_dir.exists()

    table = ImagePeakTable(
        peaks=[PeakCandidate(row=10.0, col=12.0, intensity=500.0)],
        detector_name="python_log",
    )
    csv_path = write_peak_table_cache(
        table,
        cache_dir=cache_dir,
        stem="false_split_case",
        config=_default_config(),
    )

    meta_path = cache_dir / "false_split_case.candidates.meta.json"
    assert cache_dir.is_dir()
    assert csv_path.is_file()
    assert meta_path.is_file()


def test_write_peak_table_cache_accepts_string_cache_dir(tmp_path: Path) -> None:
    cache_dir = tmp_path / "nested" / ".puncta_cache"
    table = ImagePeakTable(peaks=[], detector_name="python_log")

    csv_path = write_peak_table_cache(
        table,
        cache_dir=str(cache_dir),
        stem="sample",
        config=_default_config(),
    )

    assert cache_dir.is_dir()
    assert csv_path.is_file()
    assert (cache_dir / "sample.candidates.meta.json").is_file()
