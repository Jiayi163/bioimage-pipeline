"""CLI smoke tests for stack batch processing (Phases S.6-S.7)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_SCRIPT = REPO_ROOT / "examples" / "run_stack_batch.py"
EXAMPLE_SCRIPT = REPO_ROOT / "examples" / "run_stack_example.py"
RECIPE_DEMO = REPO_ROOT / "examples" / "stack_batch_recipe.json"


def _run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI_SCRIPT), *args],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_demo_mode_creates_expected_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo_out"
    result = _run_cli("--demo", "--output", str(output_dir), cwd=REPO_ROOT)

    assert result.returncode == 0, result.stderr or result.stdout
    assert (output_dir / "all_measurements.csv").exists()
    mask_files = list(output_dir.glob("*_mask.tif"))
    assert len(mask_files) >= 1


def test_cli_demo_with_qc_and_processed(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo_qc"
    result = _run_cli(
        "--demo",
        "--output",
        str(output_dir),
        "--export-processed",
        "--generate-qc",
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert list(output_dir.glob("*_processed.tif"))
    assert list(output_dir.glob("*_qc_mask_overlay.png"))


def test_cli_missing_input_shows_helpful_error() -> None:
    result = _run_cli("--input", "path/to/nonexistent", "--output", "out")

    assert result.returncode != 0
    assert "does not exist" in (result.stderr + result.stdout)


def test_cli_recipe_demo_mode(tmp_path: Path) -> None:
    import json

    output_dir = tmp_path / "recipe_out"
    recipe_path = tmp_path / "demo_recipe.json"
    recipe_path.write_text(
        json.dumps({"demo": True, "output": str(output_dir), "generate_qc": True}),
        encoding="utf-8",
    )

    result = _run_cli("--recipe", str(recipe_path), cwd=REPO_ROOT)

    assert result.returncode == 0, result.stderr or result.stdout
    assert (output_dir / "all_measurements.csv").exists()
    assert list(output_dir.glob("*_qc_label_overlay.png"))


def test_stack_example_script_runs() -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_example_recipe_file_exists() -> None:
    assert RECIPE_DEMO.is_file()
