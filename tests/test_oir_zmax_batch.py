"""Tests for OIR Z-max batch workflow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from bioimage_pipeline.fiji_runner import default_fiji_headless, extract_fiji_errors
from bioimage_pipeline.oir_zmax_batch import (
    DEFAULT_MACRO_PATH,
    build_manual_oir_zmax_macro,
    run_oir_zmax_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_SCRIPT = REPO_ROOT / "examples" / "run_oir_zmax_batch.py"
FOLDER_MACRO_FILE = REPO_ROOT / "examples" / "fiji_macros" / "stacking_zmax.ijm"


def test_default_macro_exists() -> None:
    assert DEFAULT_MACRO_PATH.is_file()
    assert FOLDER_MACRO_FILE.is_file()


def test_run_oir_zmax_batch_empty_folder(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    result = run_oir_zmax_batch(input_dir, output_dir, engine="python")
    assert result.engine == "python"
    assert result.processed == []
    assert result.failed == []


def test_run_oir_zmax_batch_missing_input_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_oir_zmax_batch(tmp_path / "missing", tmp_path / "out", engine="python")


def test_run_oir_zmax_batch_default_engine_writes_manual_macro(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "sample.oir").write_bytes(b"")

    result = run_oir_zmax_batch(input_dir, output_dir)

    assert result.engine == "fiji-manual-macro"
    assert result.manual_macro_path is not None
    assert result.manual_macro_path.is_file()
    macro_text = result.manual_macro_path.read_text(encoding="utf-8")
    assert "Bio-Formats Windowless Importer" in macro_text
    assert "sample.oir" not in macro_text  # file discovery happens inside Fiji
    assert str(input_dir.resolve()).replace("\\", "/") in macro_text
    assert str(output_dir.resolve()).replace("\\", "/") in macro_text
    assert "File.separator" not in macro_text
    assert 'return path + "/";' in macro_text


def test_generated_macro_processes_only_oir_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    macro_text = build_manual_oir_zmax_macro(input_dir, output_dir)

    assert 'endsWith(lower, ".oir")' in macro_text
    assert 'endsWith(lower, ".tif")' not in macro_text
    assert 'endsWith(lower, ".tiff")' not in macro_text


def test_run_oir_zmax_batch_builds_file_pairs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "sample.oir").write_bytes(b"")

    result = run_oir_zmax_batch(input_dir, output_dir, engine="python")
    assert len(result.file_pairs) == 1
    assert result.file_pairs[0].input_oir.name == "sample.oir"
    assert result.file_pairs[0].output_tif == output_dir / "sample.tif"
    assert "Bio-Formats Windowless Importer" in result.file_pairs[0].bioformats_import_command
    assert "open=[" in result.file_pairs[0].bioformats_import_command
    assert "\\" not in result.file_pairs[0].bioformats_import_command


def test_extract_fiji_errors_detects_verify_error() -> None:
    errors = extract_fiji_errors("", "java.lang.VerifyError: Bad type on operand stack")
    assert any("VerifyError" in line for line in errors)


def test_default_fiji_headless_is_boolean() -> None:
    assert isinstance(default_fiji_headless(), bool)


def test_cli_help_exposes_headless_flags() -> None:
    result = subprocess.run(
        [sys.executable, str(CLI_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--headless" in result.stdout
    assert "--no-headless" in result.stdout


def test_cli_missing_input_returns_error() -> None:
    result = subprocess.run(
        [sys.executable, str(CLI_SCRIPT), "--input", "missing_dir", "--output", "out"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "does not exist" in (result.stderr + result.stdout)
