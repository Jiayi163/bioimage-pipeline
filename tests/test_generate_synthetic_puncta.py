"""Smoke tests for the synthetic puncta generator script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "generate_synthetic_puncta.py"


def _load_generator_module():
    spec = importlib.util.spec_from_file_location("generate_synthetic_puncta", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_synthetic_puncta"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load_generator_module()


def _count_components(mask: np.ndarray) -> int:
    _, count = ndimage.label(mask)
    return int(count)


def test_basic_case1_writes_all_files(tmp_path: Path, gen) -> None:
    case = gen.BASIC_CASES["case1_isolated"]
    case.config.random_seed = 0
    paths = gen.write_case_outputs(case, tmp_path)

    assert paths["noisy"].is_file()
    assert paths["clean"].is_file()
    assert paths["mask"].is_file()
    assert paths["seeds"].is_file()
    assert paths["ground_truth"].is_file()

    truth = json.loads(paths["ground_truth"].read_text(encoding="utf-8"))
    assert truth["true_spot_count"] == 1

    mask = tifffile.imread(paths["mask"]) > 0
    assert _count_components(mask) == 1

    seeds = tifffile.imread(paths["seeds"])
    assert int(seeds.max()) == 1


def test_case2_separated_produces_two_mask_components(tmp_path: Path, gen) -> None:
    case = gen.BASIC_CASES["case2_separated"]
    case.config.random_seed = 0
    paths = gen.write_case_outputs(case, tmp_path)

    truth = json.loads(paths["ground_truth"].read_text(encoding="utf-8"))
    assert truth["true_spot_count"] == 2

    mask = tifffile.imread(paths["mask"]) > 0
    assert _count_components(mask) == 2


def test_case3_overlapping_produces_single_mask_component(tmp_path: Path, gen) -> None:
    case = gen.BASIC_CASES["case3_overlapping"]
    case.config.random_seed = 0
    paths = gen.write_case_outputs(case, tmp_path)

    truth = json.loads(paths["ground_truth"].read_text(encoding="utf-8"))
    assert truth["true_spot_count"] == 2
    assert truth["mask"]["merged_single_object"] is True

    mask = tifffile.imread(paths["mask"]) > 0
    assert _count_components(mask) == 1

    seeds = tifffile.imread(paths["seeds"])
    assert int(seeds.max()) == 2


def test_batch_dry_run_manifest(tmp_path: Path, gen) -> None:
    conditions = gen.iter_batch_conditions(limit=3)
    manifest_path = gen.write_batch_manifest(tmp_path, conditions)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["listed_conditions"] == 3
    assert len(payload["conditions"]) == 3
    assert "grid_definition" in payload


def test_render_clean_peak_at_spot_center(gen) -> None:
    case = gen.BASIC_CASES["case1_isolated"]
    clean = gen.render_clean_image(case.spots, case.config)
    row = int(round(case.spots[0].y))
    col = int(round(case.spots[0].x))
    background = case.config.background
    assert clean[row, col] > background + case.spots[0].amplitude * 0.9
