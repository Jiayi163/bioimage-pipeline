"""Tests for puncta detector cache helpers."""

from __future__ import annotations

import sys
from pathlib import Path

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.detector_cache import (
    META_SUFFIX,
    cache_paths,
    write_peak_table_cache,
)
from bioimage_pipeline.puncta.types import ImagePeakTable, PeakCandidate

FALSE_SPLIT_CASE = (
    "false_split_sig2p0_noisemedium_amphigh_gradoff_ellipelongated_seed101"
)


def _default_config() -> PunctaDeclumpConfig:
    return PunctaDeclumpConfig(
        threshold_method="manual",
        manual_threshold_value=50.0,
        candidate_detector="python_log",
    )


def _benchmark_cache_layout(tmp_path: Path) -> tuple[str, Path]:
    """Mirror run_synthetic_benchmarks output/cache layout for false_split cases."""
    output_dir = (
        tmp_path
        / "synthetic_test_data"
        / "results"
        / f"{FALSE_SPLIT_CASE}_stage2_full"
    )
    cache_dir = output_dir / ".puncta_cache"
    return str(output_dir / ".puncta_cache"), cache_dir / f"{FALSE_SPLIT_CASE}{META_SUFFIX}"


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


def test_write_peak_table_cache_false_split_meta_json_path(tmp_path: Path) -> None:
    """Regression: false_split cache meta paths exceed Windows MAX_PATH without long-path support."""
    cache_dir, meta_path = _benchmark_cache_layout(tmp_path)
    assert not Path(cache_dir).exists()

    if sys.platform == "win32":
        # Pre-fix this path length exceeded MAX_PATH for the meta JSON filename.
        assert len(str(meta_path.resolve(strict=False))) > 260

    table = ImagePeakTable(
        peaks=[PeakCandidate(row=24.0, col=24.0, intensity=1800.0)],
        detector_name="python_log",
    )
    csv_path = write_peak_table_cache(
        table,
        cache_dir=cache_dir,
        stem=FALSE_SPLIT_CASE,
        config=_default_config(),
    )

    resolved_csv_path, resolved_meta_path = cache_paths(cache_dir, FALSE_SPLIT_CASE)
    assert resolved_meta_path.parent.is_dir()
    assert csv_path.is_file()
    assert resolved_meta_path.is_file()
    assert resolved_meta_path.read_text(encoding="utf-8").strip()
