"""Tests for OIR projection lifecycle snapshots and audit logging."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from bioimage_pipeline.oir_projection_lifecycle import (
    AUDIT_LOG,
    OirProjectionLifecycleRecorder,
    compare_cross_run_snapshots,
    load_previous_workflow_end,
)
from bioimage_pipeline.oir_zmax_batch import (
    OirFilePair,
    _ensure_projection_output_dir,
    _reconcile_fiji_outputs,
)


def test_compare_cross_run_snapshots_warns_when_tiffs_disappear() -> None:
    previous_end = {
        "projection_output_dir": "C:/output/oir_projection",
        "existing_tifs": [
            {"path": "C:/output/oir_projection/a.tif"},
            {"path": "C:/output/oir_projection/b.tif"},
        ],
    }
    current_start = {
        "projection_output_dir": "C:/output/oir_projection",
        "existing_tifs": [],
    }

    warnings = compare_cross_run_snapshots(previous_end, current_start)

    assert any("2 TIFF(s) present at previous workflow_end" in warning for warning in warnings)
    assert any("missing: C:/output/oir_projection/a.tif" in warning for warning in warnings)


def test_compare_cross_run_snapshots_warns_on_projection_dir_change() -> None:
    previous_end = {
        "projection_output_dir": "C:/old/oir_projection",
        "existing_tifs": [{"path": "C:/old/oir_projection/a.tif"}],
    }
    current_start = {
        "projection_output_dir": "C:/new/oir_projection",
        "existing_tifs": [],
    }

    warnings = compare_cross_run_snapshots(previous_end, current_start)

    assert any("projection_output_dir changed between runs" in warning for warning in warnings)


def test_lifecycle_recorder_persists_ordered_stages(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    logs_dir = results_dir / "logs"
    projection_dir = results_dir / "oir_projection"
    logs_dir.mkdir(parents=True)
    projection_dir.mkdir()

    recorder = OirProjectionLifecycleRecorder(
        logs_dir=logs_dir,
        results_dir=results_dir,
        projection_dir=projection_dir,
    )
    recorder.record_workflow_start()
    recorder.record_stage("setup_directories")
    recorder.record_stage("prepare_input_entry")
    recorder.record_stage("before_cache_evaluation")
    recorder.record_stage("after_projection")
    recorder.record_stage("workflow_end")

    payload = json.loads((logs_dir / "oir_projection_lifecycle.json").read_text(encoding="utf-8"))
    stage_names = [stage["stage"] for stage in payload["stages"]]

    assert stage_names == [
        "workflow_start",
        "setup_directories",
        "prepare_input_entry",
        "before_cache_evaluation",
        "after_projection",
        "workflow_end",
    ]


def test_cross_run_warning_loaded_from_previous_lifecycle_file(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    logs_dir = results_dir / "logs"
    projection_dir = results_dir / "oir_projection"
    logs_dir.mkdir(parents=True)
    projection_dir.mkdir()

    previous = OirProjectionLifecycleRecorder(
        logs_dir=logs_dir,
        results_dir=results_dir,
        projection_dir=projection_dir,
    )
    previous.record_workflow_start()
    previous.record_stage(
        "workflow_end",
        snapshot={
            "projection_output_dir": str(projection_dir.resolve()),
            "projection_output_dir_exists": True,
            "existing_tifs": [{"path": str((projection_dir / "sample.tif").resolve())}],
        },
    )

    end_snapshot = load_previous_workflow_end(logs_dir / "oir_projection_lifecycle.json")
    assert end_snapshot is not None
    assert len(end_snapshot["existing_tifs"]) == 1

    current = OirProjectionLifecycleRecorder(
        logs_dir=logs_dir,
        results_dir=results_dir,
        projection_dir=projection_dir,
    )
    current.record_workflow_start()

    assert current.cross_run_warnings
    assert any("missing at workflow_start" in warning for warning in current.cross_run_warnings)


def test_reconcile_fiji_outputs_logs_move_within_projection_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "oir_projection"
    logs_dir = tmp_path / "logs"
    output_dir.mkdir()
    misnamed = output_dir / "wrong_name.tif"
    misnamed.write_bytes(b"tiff")
    pair = OirFilePair(
        input_oir=(tmp_path / "sample.oir").resolve(),
        output_tif=(output_dir / "sample.tif").resolve(),
    )
    (tmp_path / "sample.oir").write_bytes(b"oir")

    processed, failed, _files_created, remapped = _reconcile_fiji_outputs(
        output_dir,
        [pair],
        logs_dir=logs_dir,
    )

    assert processed == ["sample.tif"]
    assert not failed
    assert remapped
    audit_text = (logs_dir / AUDIT_LOG).read_text(encoding="utf-8")
    assert "move_within_projection_dir" in audit_text
    assert "sample.tif" in audit_text


def test_ensure_projection_output_dir_logs_only_on_create(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    projection_dir = tmp_path / "oir_projection"

    _ensure_projection_output_dir(projection_dir, logs_dir=logs_dir)
    first_log = (logs_dir / AUDIT_LOG).read_text(encoding="utf-8")
    assert "mkdir_projection_dir" in first_log

    _ensure_projection_output_dir(projection_dir, logs_dir=logs_dir)
    second_log = (logs_dir / AUDIT_LOG).read_text(encoding="utf-8")
    assert second_log.count("mkdir_projection_dir") == 1


def test_save_tiff_audit_log_for_oir_projection_write(tmp_path: Path) -> None:
    from bioimage_pipeline.io import save_tiff
    import numpy as np

    logs_dir = tmp_path / "logs"
    output_path = tmp_path / "oir_projection" / "sample.tif"
    output_path.parent.mkdir()

    save_tiff(output_path, np.zeros((4, 4), dtype=np.uint8), audit_logs_dir=logs_dir)

    audit_text = (logs_dir / AUDIT_LOG).read_text(encoding="utf-8")
    assert "write_tif" in audit_text
    assert "sample.tif" in audit_text


def test_prepare_input_workflow_records_lifecycle_stages(tmp_path: Path) -> None:
    from bioimage_pipeline.analysis import (
        RESULTS_OIR_PROJECTION_DIR,
        _prepare_cellprofiler_input_dir,
        _prepare_workflow_directories,
        resolve_workflow_output_dir,
    )
    from bioimage_pipeline.oir_projection_lifecycle import OirProjectionLifecycleRecorder

    input_dir = tmp_path / "input"
    nested = input_dir / "plate"
    nested.mkdir(parents=True)
    (nested / "sample.oir").write_bytes(b"oir")

    results_dir = resolve_workflow_output_dir(tmp_path / "results")
    directories = _prepare_workflow_directories(results_dir)
    lifecycle = OirProjectionLifecycleRecorder(
        logs_dir=directories["logs"],
        results_dir=directories["results"],
        projection_dir=directories["oir_projection"],
    )
    lifecycle.record_workflow_start()
    lifecycle.record_stage("setup_directories")

    with patch("bioimage_pipeline.oir_zmax_batch.run_oir_zmax_batch") as mock_batch:
        mock_batch.return_value = type(
            "Result",
            (),
            {
                "input_dir": input_dir.resolve(),
                "output_dir": (results_dir / RESULTS_OIR_PROJECTION_DIR).resolve(),
                "engine": "python",
                "processed": ["sample.tif"],
                "failed": [],
                "files_created": [],
                "remapped_outputs": [],
                "file_pairs": [],
                "fiji_executable": None,
                "fiji_headless": None,
                "fiji_returncode": None,
                "generated_macro_path": None,
                "fiji_log_files": {},
                "file_profiles": [],
                "cache_hits": [],
                "reprojected": ["sample.tif"],
                "force_oir_reproject": False,
            },
        )()
        _prepare_cellprofiler_input_dir(
            input_dir,
            results_dir=directories["results"],
            logs_dir=directories["logs"],
            oir_projection_engine="python",
            lifecycle=lifecycle,
        )

    payload = json.loads(
        (directories["logs"] / "oir_projection_lifecycle.json").read_text(encoding="utf-8")
    )
    stage_names = [stage["stage"] for stage in payload["stages"]]
    assert "prepare_input_entry" in stage_names
    mock_batch.assert_called_once()
    assert mock_batch.call_args.kwargs["lifecycle"] is lifecycle
