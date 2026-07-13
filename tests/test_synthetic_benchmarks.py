"""Tests for benchmark seed utilities and multi-start ordering."""

from __future__ import annotations

import numpy as np

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.gmm_multi_start import ordered_multi_start_strategies
from bioimage_pipeline.puncta.types import PeakCandidate
from scripts.benchmark_seed_utils import generate_seed_list
from scripts.generate_synthetic_puncta import generate_seed_list as gen_seed_list_alias


def test_generate_seed_list_length_and_step() -> None:
    seeds = generate_seed_list(5)
    assert seeds == [101, 202, 303, 404, 505]
    assert gen_seed_list_alias(3) == [101, 202, 303]


def test_ordered_multi_start_prioritizes_staged_strategies() -> None:
    peaks = [PeakCandidate(row=5.0, col=5.0, intensity=1000.0)]
    init_sets = {
        "offset_x_sep2": peaks * 2,
        "detector_based": peaks * 2,
        "symmetric_x_sep2": peaks * 2,
        "residual_peak": peaks * 2,
        "major_axis": peaks * 2,
    }
    order = ordered_multi_start_strategies(init_sets, config=PunctaDeclumpConfig())
    assert order[:3] == ["detector_based", "residual_peak", "major_axis"]
