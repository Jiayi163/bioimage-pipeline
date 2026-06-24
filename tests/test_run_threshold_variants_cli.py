"""Tests for the threshold variant CLI example."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "run_threshold_variants.py"

SAMPLE_CPPIPE = """CellProfiler Pipeline: http://www.cellprofiler.org
Version:5

Images:[module_num:1|svn_version:'Unknown'|variable_revision_number:1|show_window:False|notes:[]]
Filter images?:No

IdentifyPrimaryObjects:[module_num:2|svn_version:'Unknown'|variable_revision_number:1|show_window:False|notes:[]]
Select the input image:Green
Name the primary objects to be identified:Spots
Threshold strategy:Adaptive
Thresholding method:Otsu

ExportToSpreadsheet:[module_num:3|svn_version:'Unknown'|variable_revision_number:1|show_window:False|notes:[]]
Select the column delimiter:Comma
"""


def test_run_threshold_variants_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "trial", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--subset-count" in completed.stdout


@patch("bioimage_pipeline.threshold_recommender.run_threshold_recommender_trial")
def test_run_threshold_variants_default_trial_command(
    mock_trial: MagicMock,
    tmp_path: Path,
) -> None:
    import importlib.util

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "img_001.tif").write_bytes(b"fake")
    cppipe = tmp_path / "pipeline.cppipe"
    cppipe.write_text(SAMPLE_CPPIPE, encoding="utf-8")

    trial_result = MagicMock()
    trial_result.recommender_root = tmp_path / "out" / "threshold_recommender"
    trial_result.subset_manifest.image_names = ["img_001.tif"]
    trial_result.subset_dir = trial_result.recommender_root / "subset"
    trial_result.comparison_paths = {"csv": tmp_path / "comparison.csv"}
    trial_result.ranking_paths = {"csv": tmp_path / "ranking.csv"}
    trial_result.summaries = []
    trial_result.ranked_scores = []
    trial_result.run_results = []
    trial_result.gt_ranked_scores = []
    mock_trial.return_value = trial_result

    spec = importlib.util.spec_from_file_location("run_threshold_variants", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    argv = [
        str(SCRIPT),
        "--cppipe",
        str(cppipe),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(tmp_path / "out"),
        "--max-variants",
        "1",
        "--subset-count",
        "1",
    ]
    with patch.object(module, "run_threshold_recommender_trial", mock_trial):
        with patch.object(sys, "argv", argv):
            assert module.main() == 0
    mock_trial.assert_called_once()


def test_run_threshold_variants_apply_requires_confirm(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "apply",
            "--output-dir",
            str(tmp_path),
            "--input-dir",
            str(tmp_path),
            "--variant-id",
            "001_baseline",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "--confirm" in completed.stderr


def test_run_threshold_variants_compare_only_from_session(tmp_path: Path) -> None:
    recommender_root = tmp_path / "threshold_recommender"
    ranking_csv = recommender_root / "threshold_variants" / "threshold_variant_ranking.csv"
    comparison_csv = recommender_root / "threshold_variants" / "threshold_variant_comparison.csv"
    ranking_csv.parent.mkdir(parents=True, exist_ok=True)
    ranking_csv.write_text("rank,variant_id\n", encoding="utf-8")
    comparison_csv.write_text("variant_id\n", encoding="utf-8")
    (recommender_root / "recommender_session.json").write_text(
        json.dumps(
            {
                "imported_cppipe_path": str(tmp_path / "pipeline.cppipe"),
                "input_dir": str(tmp_path / "input"),
                "output_dir": str(tmp_path),
                "recommender_root": str(recommender_root),
                "subset_manifest": {
                    "source_dir": str(tmp_path / "input"),
                    "staged_dir": str(recommender_root / "subset"),
                    "mode": "auto",
                    "sample_count": 1,
                    "sample_method": "even",
                    "image_names": ["a.tif"],
                },
                "comparison_paths": {"csv": str(comparison_csv)},
                "ranking_paths": {"csv": str(ranking_csv)},
                "artifacts": [],
                "ranking": [],
                "preview_index": {},
            }
        ),
        encoding="utf-8",
    )

    argv = [
        str(SCRIPT),
        "trial",
        "--cppipe",
        str(tmp_path / "pipeline.cppipe"),
        "--input-dir",
        str(tmp_path / "input"),
        "--output-dir",
        str(tmp_path),
        "--compare-only",
    ]
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_threshold_variants", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    with patch.object(sys, "argv", argv):
        assert module.main() == 0
