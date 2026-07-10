"""Tests for selective routing, detector cache, watershed, and timing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.detector_cache import (
    evaluate_detector_cache,
    write_peak_table_cache,
)
from bioimage_pipeline.puncta.object_router import ObjectRouter
from bioimage_pipeline.puncta.peak_assignment import assign_peaks_to_objects
from bioimage_pipeline.puncta.pipeline import run_puncta_declump
from bioimage_pipeline.puncta.trackmate_runner import parse_trackmate_csv
from bioimage_pipeline.puncta.types import ImagePeakTable, ObjectInfo, PeakCandidate
from bioimage_pipeline.puncta.watershed_declump import apply_watershed_declump
from tests.test_puncta_declump import (
    default_config,
    make_binary_disk,
    make_gaussian_spot,
)


def _object_info(**overrides: object) -> ObjectInfo:
    params = {
        "label": 1,
        "area": 20.0,
        "equivalent_diameter": 5.0,
        "bbox": (10, 10, 20, 20),
        "centroid": (15.0, 15.0),
        "brightest_row": 15.0,
        "brightest_col": 15.0,
        "brightest_intensity": 500.0,
        "eccentricity": 0.2,
        "solidity": 0.95,
        "elongation": 1.1,
    }
    params.update(overrides)
    return ObjectInfo(**params)


def test_router_marks_small_round_object_ordinary() -> None:
    router = ObjectRouter(default_config())
    decision = router.classify(_object_info(), [PeakCandidate(15.0, 15.0, 500.0)])
    assert decision.route == "ordinary_single"


def test_router_diameter_seven_area_stays_ordinary() -> None:
    import math

    area = math.pi * 3.5**2
    router = ObjectRouter(default_config())
    obj = _object_info(equivalent_diameter=7.0, area=area)
    decision = router.classify(obj, [PeakCandidate(15.0, 15.0, 500.0)])
    assert decision.route == "ordinary_single"


def test_router_two_close_peaks_on_small_object_stays_ordinary() -> None:
    router = ObjectRouter(default_config())
    peaks = [
        PeakCandidate(14.0, 14.0, 500.0),
        PeakCandidate(16.0, 16.0, 450.0),
    ]
    decision = router.classify(_object_info(), peaks)
    assert decision.route == "ordinary_single"


def test_router_two_separated_peaks_on_large_blob_is_suspicious() -> None:
    router = ObjectRouter(default_config(single_spot_max_diameter=7.0))
    peaks = [
        PeakCandidate(10.0, 10.0, 500.0),
        PeakCandidate(20.0, 20.0, 450.0),
    ]
    obj = _object_info(equivalent_diameter=8.5, area=80.0)
    decision = router.classify(obj, peaks)
    assert decision.route == "suspicious"


def test_router_three_separated_peaks_is_suspicious() -> None:
    router = ObjectRouter(default_config())
    peaks = [
        PeakCandidate(10.0, 10.0, 500.0),
        PeakCandidate(15.0, 15.0, 450.0),
        PeakCandidate(20.0, 20.0, 400.0),
    ]
    decision = router.classify(_object_info(), peaks)
    assert decision.route == "suspicious"


def test_router_low_solidity_alone_stays_ordinary() -> None:
    router = ObjectRouter(default_config())
    obj = _object_info(solidity=0.7, eccentricity=0.7, elongation=1.7)
    decision = router.classify(obj, [PeakCandidate(15.0, 15.0, 500.0)])
    assert decision.route == "ordinary_single"


def test_router_low_solidity_with_two_separated_peaks_is_suspicious() -> None:
    router = ObjectRouter(default_config())
    obj = _object_info(solidity=0.7, eccentricity=0.7, elongation=1.7)
    peaks = [
        PeakCandidate(10.0, 10.0, 500.0),
        PeakCandidate(20.0, 20.0, 450.0),
    ]
    decision = router.classify(obj, peaks)
    assert decision.route == "suspicious"


def test_router_bulk_majority_ordinary() -> None:
    router = ObjectRouter(default_config())
    objects = [_object_info(label=index, area=25.0, equivalent_diameter=6.0) for index in range(1, 1020)]
    assigned = {index: [PeakCandidate(15.0, 15.0, 500.0)] for index in range(1, 1020)}
    for index in range(1, 51):
        assigned[index] = [
            PeakCandidate(10.0, 10.0, 500.0),
            PeakCandidate(15.0, 15.0, 450.0),
            PeakCandidate(20.0, 20.0, 400.0),
        ]
    summary = router.summarize(objects, assigned)
    assert summary.ordinary > summary.suspicious
    assert summary.suspicious / len(objects) < 0.2


def test_peak_assignment_maps_global_peaks_to_objects() -> None:
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[10:18, 10:18] = 1
    labels[20:28, 20:28] = 2
    objects = [
        _object_info(label=1, bbox=(10, 10, 18, 18)),
        _object_info(label=2, bbox=(20, 20, 28, 28), equivalent_diameter=5.0),
    ]
    table = ImagePeakTable(
        peaks=[
            PeakCandidate(14.0, 14.0, 500.0),
            PeakCandidate(24.0, 24.0, 480.0),
        ],
        detector_name="python_log",
    )
    assigned = assign_peaks_to_objects(labels, objects, table, default_config())
    assert len(assigned[1]) == 1
    assert len(assigned[2]) == 1


def test_detector_cache_roundtrip(tmp_path: Path) -> None:
    config = default_config(candidate_detector="python_log")
    source = tmp_path / "input.tif"
    source.write_bytes(b"x")
    table = ImagePeakTable(
        peaks=[PeakCandidate(1.0, 2.0, 100.0)],
        detector_name="python_log",
    )
    write_peak_table_cache(
        table,
        cache_dir=tmp_path / "cache",
        stem="sample",
        config=config,
        source_path=source,
    )
    is_fresh, csv_path, _ = evaluate_detector_cache(
        source_path=source,
        cache_dir=tmp_path / "cache",
        stem="sample",
        config=config,
    )
    assert is_fresh
    assert csv_path.is_file()


def test_selective_routing_uses_fast_path_for_single_spot(tmp_path: Path) -> None:
    image = make_gaussian_spot((64, 64), (32.0, 32.0), sigma=1.5, amplitude=500.0)
    config = default_config(
        single_spot_max_diameter=12.0,
        enable_selective_routing=True,
        diagnostic_mode="summary",
    )
    result = run_puncta_declump(
        image,
        config,
        output_dir=str(tmp_path),
        stem="spot",
    )
    assert result.summary.total_mask_objects == 1
    assert result.summary.fast_path_objects + result.summary.single_path_objects >= 1
    assert result.timing
    assert "candidate_detection_time" in result.timing


def test_watershed_splits_multi_center_object() -> None:
    from bioimage_pipeline.puncta.types import PunctumCandidate

    image = make_gaussian_spot((64, 64), (28.0, 28.0), sigma=1.2, amplitude=500.0)
    image += make_gaussian_spot((64, 64), (36.0, 36.0), sigma=1.2, amplitude=450.0)
    mask = make_binary_disk((64, 64), (32.0, 32.0), radius=8.0)
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[mask] = 1
    obj = _object_info(
        label=1,
        bbox=(24, 24, 40, 40),
        equivalent_diameter=14.0,
        area=float(mask.sum()),
    )
    candidates = {
        1: [
            PunctumCandidate(
                object_id=1,
                candidate_id=1,
                component_id=1,
                path="gmm",
                fit_status="fit_ok",
                initial_row=28.0,
                initial_col=28.0,
                fitted_row=28.0,
                fitted_col=28.0,
                accepted=True,
            ),
            PunctumCandidate(
                object_id=1,
                candidate_id=2,
                component_id=2,
                path="gmm",
                fit_status="fit_ok",
                initial_row=36.0,
                initial_col=36.0,
                fitted_row=36.0,
                fitted_col=36.0,
                accepted=True,
            ),
        ]
    }
    new_labels, next_label = apply_watershed_declump(
        labels,
        image,
        [obj],
        candidates,
    )
    assert next_label >= 3
    assert len(np.unique(new_labels[mask])) >= 2


def test_parse_trackmate_csv_fixture(tmp_path: Path) -> None:
    csv_path = tmp_path / "spots.csv"
    csv_path.write_text(
        "LABEL,ID,QUALITY,POSITION_X,POSITION_Y,POSITION_Z\n"
        "0,1,0.95,10.5,20.25,0.0\n",
        encoding="utf-8",
    )
    peaks = parse_trackmate_csv(csv_path)
    assert len(peaks) == 1
    assert peaks[0].col == pytest.approx(10.5)
    assert peaks[0].row == pytest.approx(20.25)


def test_batch_processes_two_images(tmp_path: Path) -> None:
    import tifffile

    from bioimage_pipeline.puncta.batch import run_puncta_batch

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for index, center in enumerate([(20.0, 20.0), (30.0, 30.0)]):
        image = make_gaussian_spot((48, 48), center, sigma=1.2, amplitude=400.0)
        tifffile.imwrite(input_dir / f"img_{index}.tif", image.astype(np.uint16))

    config = default_config(diagnostic_mode="summary", log_progress=False)
    batch = run_puncta_batch(input_dir, output_dir, config)
    assert len(batch.processed) == 2
    assert not batch.failed
    summary_path = output_dir / "img_0" / "img_0_summary.json"
    assert summary_path.is_file()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "timing" in payload
