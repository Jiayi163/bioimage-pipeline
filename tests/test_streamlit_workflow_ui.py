"""Tests for Phase 15.0 workflow UI helpers."""

from pathlib import Path

import pandas as pd

from bioimage_pipeline.workflow_ui import (
    list_qc_pngs,
    load_measurements_for_display,
    read_text_tail,
    save_uploaded_cppipe,
    validate_workflow_inputs,
)


class _FakeUpload:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


def test_read_text_tail_returns_full_small_file(tmp_path: Path) -> None:
    log_file = tmp_path / "stdout.log"
    log_file.write_text("line one\nline two\n", encoding="utf-8")
    assert read_text_tail(log_file) == "line one\nline two\n"


def test_read_text_tail_truncates_large_file(tmp_path: Path) -> None:
    log_file = tmp_path / "stdout.log"
    log_file.write_text("x" * 100, encoding="utf-8")
    tail = read_text_tail(log_file, max_chars=20)
    assert "truncated" in tail
    assert tail.endswith("x" * 20)


def test_read_text_tail_missing_file(tmp_path: Path) -> None:
    assert "missing file" in read_text_tail(tmp_path / "missing.log")


def test_list_qc_pngs_sorted(tmp_path: Path) -> None:
    qc_dir = tmp_path / "qc"
    qc_dir.mkdir()
    (qc_dir / "b_overlay.png").write_bytes(b"png")
    (qc_dir / "a_overlay.png").write_bytes(b"png")
    assert [path.name for path in list_qc_pngs(qc_dir)] == [
        "a_overlay.png",
        "b_overlay.png",
    ]


def test_load_measurements_prefers_merged_csv(tmp_path: Path) -> None:
    measurements_dir = tmp_path / "measurements"
    measurements_dir.mkdir()
    pd.DataFrame({"ImageNumber": [1], "AreaShape_Area": [42]}).to_csv(
        measurements_dir / "other.csv",
        index=False,
    )
    pd.DataFrame({"ImageNumber": [1], "AreaShape_Area": [99]}).to_csv(
        measurements_dir / "merged_measurements.csv",
        index=False,
    )
    loaded = load_measurements_for_display(measurements_dir)
    assert loaded is not None
    assert loaded.iloc[0]["AreaShape_Area"] == 99


def test_save_uploaded_cppipe_writes_file(tmp_path: Path) -> None:
    uploaded = _FakeUpload(b'{"modules":[]}')
    saved = save_uploaded_cppipe(uploaded, "pipeline.cppipe", tmp_path / "cache")
    assert saved.name == "pipeline.cppipe"
    assert saved.read_text(encoding="utf-8") == '{"modules":[]}'


def test_validate_workflow_inputs_reports_missing_paths(tmp_path: Path) -> None:
    errors = validate_workflow_inputs(
        tmp_path / "missing_input",
        "",
        None,
    )
    assert any("Input image folder" in error for error in errors)
    assert any("Output folder is required" in error for error in errors)
    assert any("pipeline (.cppipe) is required" in error for error in errors)


def test_validate_workflow_inputs_accepts_valid_paths(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    pipeline = tmp_path / "pipeline.cppipe"
    pipeline.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "results"
    assert validate_workflow_inputs(input_dir, output_dir, pipeline) == []


def test_validate_workflow_inputs_rejects_non_cppipe(tmp_path: Path) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    pipeline = tmp_path / "pipeline.txt"
    pipeline.write_text("{}", encoding="utf-8")
    errors = validate_workflow_inputs(input_dir, tmp_path / "out", pipeline)
    assert any("must end with .cppipe" in error for error in errors)
