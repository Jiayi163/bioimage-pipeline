"""Tests for GMM oracle diagnostics and pre-merge attempt reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.gaussian_fitter import GaussianMixtureFitter
from bioimage_pipeline.puncta.types import ObjectPatch, PeakCandidate
from bioimage_pipeline.puncta.validation.gmm_oracle import (
    _oracle_attempt_diagnostics,
    fit_oracle_mixture_from_init_peaks,
    run_ground_truth_oracle_experiment,
)


def _two_peak_patch(*, separation_px: float = 1.0) -> tuple[ObjectPatch, list[PeakCandidate]]:
    shape = (25, 25)
    corrected = np.zeros(shape, dtype=np.float64)
    row = 12.0
    col_a = 12.0 - separation_px / 2.0
    col_b = 12.0 + separation_px / 2.0
    corrected[int(row), int(col_a)] = 800.0
    corrected[int(row), int(col_b)] = 780.0
    patch = ObjectPatch(
        object_id=1,
        row_offset=0,
        col_offset=0,
        corrected=corrected,
        object_mask=np.ones(shape, dtype=bool),
        background_level=0.0,
        global_bbox=(0, 0, shape[0], shape[1]),
    )
    peaks = [
        PeakCandidate(row=row, col=col_a, intensity=800.0),
        PeakCandidate(row=row, col=col_b, intensity=780.0),
    ]
    return patch, peaks


def test_oracle_attempt_diagnostics_preserve_pre_merge_centers_when_merge_collapses() -> None:
    config = PunctaDeclumpConfig(gmm_min_component_separation=1.5)
    patch, peaks = _two_peak_patch(separation_px=1.0)
    fitter = GaussianMixtureFitter(config)
    details = fit_oracle_mixture_from_init_peaks(
        fitter,
        patch,
        peaks,
        n_components=2,
        initialization_method="test_close_init",
    )
    assert len(details.pre_merge_components) == 2

    attempt = _oracle_attempt_diagnostics(
        "test_close_init",
        details,
        initial_centers=[(peak.col, peak.row) for peak in peaks],
    )

    assert len(attempt.pre_merge_fitted_centers) == 2
    assert attempt.pre_merge_center_separation_px is not None
    assert attempt.merge_collapsed
    assert attempt.post_merge_component_count <= 1
    assert len(attempt.fitted_centers) <= 1


def _write_separation_case(data_root: Path, case_name: str, *, seed: int) -> None:
    from dataclasses import replace

    from scripts.generate_synthetic_puncta import build_separation_benchmark_case, write_case_outputs

    case = replace(build_separation_benchmark_case(3, seed), name=case_name)
    write_case_outputs(case, data_root)


def test_oracle_experiment_on_generated_case(tmp_path: Path) -> None:
    case_name = "sep_benchmark_sep3_seed1010"
    _write_separation_case(tmp_path, case_name, seed=1010)
    report = run_ground_truth_oracle_experiment(
        data_root=tmp_path,
        case_name=case_name,
        config=PunctaDeclumpConfig(
            gmm_multi_start_enabled=True,
            gmm_use_mixture_acceptance_separation=True,
            gmm_acceptance_min_separation=1.5,
            enable_selective_routing=False,
            candidate_detector="python_log",
        ),
    )
    payload = json.loads(json.dumps(report.to_dict(), default=str))
    assert payload["case_name"] == case_name
    assert len(payload["true_centers"]) == 2
    assert payload["optimizer_success"] is True
    assert payload["pre_merge_component_count"] == 2
