"""Tests for stack batch workflow orchestration."""

from pathlib import Path

import numpy as np

from bioimage_pipeline.io import save_tiff
from bioimage_pipeline.stack_batch import run_stack_batch_workflow


def test_run_stack_batch_workflow_processes_folder(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        img = np.zeros((24, 32), dtype=np.uint8)
        img[:, 16:] = 200
        save_tiff(src / f"img_{i}.tif", img)

    result = run_stack_batch_workflow(
        src,
        tmp_path / "out",
        export_processed=True,
        generate_qc=True,
    )

    assert result.stack.frame_count == 3
    assert len(result.processed) == 3
    assert (tmp_path / "out" / "all_measurements.csv").exists()
    assert list((tmp_path / "out").glob("*_processed.tif"))
    assert len(result.qc_artifacts) == 3
