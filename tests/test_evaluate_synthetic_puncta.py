"""Tests for synthetic puncta evaluation against ground truth."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_synthetic_puncta.py"


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("evaluate_synthetic_puncta", EVAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_synthetic_puncta"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ev():
    return _load_eval_module()


def _write_ground_truth(path: Path, spots: list[dict[str, float]]) -> None:
    payload = {
        "case_name": path.parent.name,
        "true_spot_count": len(spots),
        "spots": [
            {
                "id": index,
                "x": spot["x"],
                "y": spot["y"],
                "amplitude": 1800.0,
                "sigma_x": 2.2,
                "sigma_y": 2.2,
            }
            for index, spot in enumerate(spots, start=1)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_measurements(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_row_column_handling_uses_final_col_as_x(tmp_path: Path, ev) -> None:
    gt_path = tmp_path / "ground_truth" / "case_x" / "synthetic_ground_truth.json"
    _write_ground_truth(gt_path, [{"x": 10.0, "y": 20.0}])

    results_dir = tmp_path / "results" / "case_x"
    _write_measurements(
        results_dir / "case_x_measurements.csv",
        [
            {
                "accepted": True,
                "final_col": 10.4,
                "final_row": 20.1,
                "path": "fast_single",
                "fit_status": "fit_ok",
                "tried_gmm": False,
            }
        ],
    )

    metrics = ev.evaluate_run(
        run_name="case_x",
        ground_truth_case="case_x",
        measurements_stem="case_x",
        data_root=tmp_path,
        tolerance_px=2.0,
    )
    assert metrics.true_positives == 1
    assert metrics.pass_criterion is True


def test_one_to_one_matching_prevents_double_assignment(tmp_path: Path, ev) -> None:
    gt_path = tmp_path / "ground_truth" / "case_pair" / "synthetic_ground_truth.json"
    _write_ground_truth(
        gt_path,
        [
            {"x": 60.0, "y": 64.0},
            {"x": 67.0, "y": 64.0},
        ],
    )

    results_dir = tmp_path / "results" / "case_pair"
    _write_measurements(
        results_dir / "case_pair_measurements.csv",
        [
            {
                "accepted": True,
                "final_col": 60.2,
                "final_row": 64.0,
                "path": "gmm",
                "fit_status": "fit_ok",
                "tried_gmm": True,
            },
            {
                "accepted": True,
                "final_col": 66.8,
                "final_row": 64.0,
                "path": "gmm",
                "fit_status": "fit_ok",
                "tried_gmm": True,
            },
        ],
    )

    metrics = ev.evaluate_run(
        run_name="case_pair",
        ground_truth_case="case_pair",
        measurements_stem="case_pair",
        data_root=tmp_path,
    )
    assert metrics.true_positives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0


def test_unmatched_predictions_are_false_positives(tmp_path: Path, ev) -> None:
    gt_path = tmp_path / "ground_truth" / "case_fp" / "synthetic_ground_truth.json"
    _write_ground_truth(gt_path, [{"x": 64.0, "y": 64.0}])

    results_dir = tmp_path / "results" / "case_fp"
    _write_measurements(
        results_dir / "case_fp_measurements.csv",
        [
            {
                "accepted": True,
                "final_col": 64.0,
                "final_row": 64.0,
                "path": "fast_single",
                "fit_status": "fit_ok",
                "tried_gmm": False,
            },
            {
                "accepted": True,
                "final_col": 80.0,
                "final_row": 80.0,
                "path": "fast_single",
                "fit_status": "fit_ok",
                "tried_gmm": False,
            },
        ],
    )

    metrics = ev.evaluate_run(
        run_name="case_fp",
        ground_truth_case="case_fp",
        measurements_stem="case_fp",
        data_root=tmp_path,
    )
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.pass_criterion is False


def test_unmatched_truth_points_are_false_negatives(tmp_path: Path, ev) -> None:
    gt_path = tmp_path / "ground_truth" / "case_fn" / "synthetic_ground_truth.json"
    _write_ground_truth(
        gt_path,
        [
            {"x": 61.0, "y": 64.0},
            {"x": 67.0, "y": 64.0},
        ],
    )

    results_dir = tmp_path / "results" / "case_fn"
    _write_measurements(
        results_dir / "case_fn_measurements.csv",
        [
            {
                "accepted": True,
                "final_col": 61.0,
                "final_row": 64.0,
                "path": "single",
                "fit_status": "fit_ok",
                "tried_gmm": True,
            }
        ],
    )

    metrics = ev.evaluate_run(
        run_name="case_fn",
        ground_truth_case="case_fn",
        measurements_stem="case_fn",
        data_root=tmp_path,
    )
    assert metrics.true_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.under_split is True
    assert metrics.pass_criterion is False


def test_count_correct_but_localization_failed(tmp_path: Path, ev) -> None:
    gt_path = tmp_path / "ground_truth" / "case_loc" / "synthetic_ground_truth.json"
    _write_ground_truth(gt_path, [{"x": 64.0, "y": 64.0}])

    results_dir = tmp_path / "results" / "case_loc"
    _write_measurements(
        results_dir / "case_loc_measurements.csv",
        [
            {
                "accepted": True,
                "final_col": 68.0,
                "final_row": 64.0,
                "path": "fast_single",
                "fit_status": "fit_ok",
                "tried_gmm": False,
            }
        ],
    )

    metrics = ev.evaluate_run(
        run_name="case_loc",
        ground_truth_case="case_loc",
        measurements_stem="case_loc",
        data_root=tmp_path,
        tolerance_px=2.0,
    )
    assert metrics.exact_count_correct is True
    assert metrics.true_positives == 0
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.pass_criterion is False


def test_empty_predictions(tmp_path: Path, ev) -> None:
    gt_path = tmp_path / "ground_truth" / "case_empty" / "synthetic_ground_truth.json"
    _write_ground_truth(gt_path, [{"x": 64.0, "y": 64.0}])

    results_dir = tmp_path / "results" / "case_empty"
    _write_measurements(
        results_dir / "case_empty_measurements.csv",
        [{"accepted": False, "final_col": 64.0, "final_row": 64.0}],
    )

    metrics = ev.evaluate_run(
        run_name="case_empty",
        ground_truth_case="case_empty",
        measurements_stem="case_empty",
        data_root=tmp_path,
    )
    assert metrics.predicted_accepted_count == 0
    assert metrics.false_negatives == 1
    assert metrics.pass_criterion is False


def test_case3_shared_ground_truth_mapping(tmp_path: Path, ev) -> None:
    gt_path = tmp_path / "ground_truth" / "case3_overlapping" / "synthetic_ground_truth.json"
    _write_ground_truth(
        gt_path,
        [
            {"x": 63.0, "y": 64.0},
            {"x": 65.0, "y": 64.0},
        ],
    )

    for run_name, stem, x_values in (
        ("case3_overlapping_normal", "case3_overlapping_normal", [63.0, 65.0]),
        ("case3_overlapping_forced_gmm", "case3_overlapping_forced_gmm", [63.1, 64.9]),
    ):
        rows = [
            {
                "accepted": True,
                "final_col": x,
                "final_row": 64.0,
                "path": "gmm",
                "fit_status": "fit_ok",
                "tried_gmm": True,
            }
            for x in x_values
        ]
        results_dir = tmp_path / "results" / run_name
        _write_measurements(results_dir / f"{stem}_measurements.csv", rows)

        metrics = ev.evaluate_run(
            run_name=run_name,
            ground_truth_case="case3_overlapping",
            measurements_stem=stem,
            data_root=tmp_path,
        )
        assert metrics.true_spot_count == 2
        assert metrics.predicted_accepted_count == 2


def test_multiple_predictions_near_same_true_point(tmp_path: Path, ev) -> None:
    gt_path = tmp_path / "ground_truth" / "case_near" / "synthetic_ground_truth.json"
    _write_ground_truth(gt_path, [{"x": 64.0, "y": 64.0}])

    results_dir = tmp_path / "results" / "case_near"
    _write_measurements(
        results_dir / "case_near_measurements.csv",
        [
            {
                "accepted": True,
                "final_col": 64.1,
                "final_row": 64.0,
                "path": "gmm",
                "fit_status": "fit_ok",
                "tried_gmm": True,
            },
            {
                "accepted": True,
                "final_col": 64.5,
                "final_row": 64.0,
                "path": "gmm",
                "fit_status": "fit_ok",
                "tried_gmm": True,
            },
        ],
    )

    metrics = ev.evaluate_run(
        run_name="case_near",
        ground_truth_case="case_near",
        measurements_stem="case_near",
        data_root=tmp_path,
    )
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.over_split is True
