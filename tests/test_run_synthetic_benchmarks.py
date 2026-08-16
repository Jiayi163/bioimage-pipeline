"""Tests for synthetic benchmark runner helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import tifffile

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.types import DeclumpResult
from scripts.run_synthetic_benchmarks import _run_pipeline_case, _select_target_cases


def _write_minimal_synthetic_case(data_root: Path, case_name: str) -> None:
    shape = (48, 48)
    image = np.zeros(shape, dtype=np.float32)
    mask = np.ones(shape, dtype=np.uint8)
    (data_root / "images" / case_name).mkdir(parents=True)
    (data_root / "masks" / case_name).mkdir(parents=True)
    tifffile.imwrite(data_root / "images" / case_name / "synthetic_noisy.tif", image)
    tifffile.imwrite(data_root / "masks" / case_name / "synthetic_mask.tif", mask)


def test_run_pipeline_case_recreates_out_dir_before_exports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Regression: cache hits may skip creating the result dir before benchmark exports."""
    data_root = tmp_path / "synthetic_test_data"
    case_name = "bench_case"
    _write_minimal_synthetic_case(data_root, case_name)

    def fake_declump(_image, _config, *, external_mask, output_dir, stem):
        del external_mask, stem
        result = DeclumpResult(
            threshold_metadata={
                "gmm_init_diagnostics": [{"multi_start_attempts": 1, "attempts": []}],
                "gmm_config": {"gmm_multi_start_mode": "full"},
            },
            timing={"total_seconds": 0.5},
        )
        out = Path(output_dir)
        if out.is_dir():
            shutil.rmtree(out)
        return result

    monkeypatch.setattr(
        "scripts.run_synthetic_benchmarks.run_puncta_declump",
        fake_declump,
    )

    out_dir, _runtime, meta = _run_pipeline_case(
        data_root,
        case_name,
        run_suffix="stage2_full",
        config=PunctaDeclumpConfig(
            diagnostic_mode="summary",
            export_fiji_tiffs=False,
            candidate_detector="python_log",
        ),
    )

    assert out_dir.is_dir()
    assert (out_dir / f"{case_name}_measurements.csv").is_file()
    summary_path = out_dir / f"{case_name}_summary.json"
    assert summary_path.is_file()
    assert (out_dir / f"{case_name}_gmm_init_diagnostics.json").is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["timing"] == {"total_seconds": 0.5}
    assert meta["timing"] == {"total_seconds": 0.5}


def test_select_target_cases_returns_exact_match() -> None:
    cases = ["sep_benchmark_sep3_seed101", "sep_benchmark_sep3_seed202"]
    assert _select_target_cases(cases, "sep_benchmark_sep3_seed101") == [
        "sep_benchmark_sep3_seed101"
    ]


def test_select_target_cases_passthrough_when_case_omitted() -> None:
    cases = ["sep_benchmark_sep3_seed101", "sep_benchmark_sep3_seed202"]
    assert _select_target_cases(cases, None) == cases


def test_select_target_cases_raises_when_missing() -> None:
    cases = ["sep_benchmark_sep3_seed101"]
    try:
        _select_target_cases(cases, "sep_benchmark_sep3_seed1010")
    except SystemExit as exc:
        assert "sep_benchmark_sep3_seed1010" in str(exc)
        assert "sep_benchmark_sep3_seed101" in str(exc)
    else:
        raise AssertionError("Expected SystemExit for missing case")
