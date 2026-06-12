"""Tests for prepare_input profiling."""

from __future__ import annotations

from pathlib import Path

from bioimage_pipeline.prepare_input_profile import (
    PrepareInputFileRecord,
    PrepareInputProfile,
    PrepareInputScanResult,
    build_investigation_notes,
    format_prepare_input_report,
    parse_fiji_oir_file_records,
    scan_prepare_input_folder,
)


def test_scan_prepare_input_folder_finds_oir_and_tiff(tmp_path: Path) -> None:
    root = tmp_path / "input"
    nested = root / "plate"
    nested.mkdir(parents=True)
    (root / "a.tif").write_bytes(b"tiff")
    (nested / "sample.oir").write_bytes(b"oir")

    scan = scan_prepare_input_folder(root)

    assert scan.oir_count == 1
    assert scan.tiff_count == 1
    assert scan.directories_scanned >= 2


def test_parse_fiji_oir_file_records_extracts_timings() -> None:
    stdout = """
[OIR] input path: C:/data/sample.oir
[OIR] timing read_seconds: 12.5
[OIR] timing conversion_seconds: 1.2
[OIR] saveAs target: C:/out/sample.tif
[OIR] timing write_seconds: 0.8
[OIR] saved file verified on disk: C:/out/sample.tif
"""
    records = parse_fiji_oir_file_records(stdout)

    assert len(records) == 1
    assert records[0].detected_type == "oir"
    assert records[0].read_seconds == 12.5
    assert records[0].conversion_seconds == 1.2
    assert records[0].write_seconds == 0.8
    assert records[0].total_seconds == 14.5


def test_build_investigation_notes_for_oir_projection() -> None:
    scan = PrepareInputScanResult(
        input_dir="/input",
        directories_scanned=8,
        oir_files=[Path("/input/a.oir")],
        tiff_files=[Path("/input/a.tif")],
        scan_seconds=0.2,
    )
    profile = PrepareInputProfile(
        input_dir="/input",
        action="oir_projection_python",
        scan=scan,
        engine="python",
        projection_seconds=41.0,
        file_records=[
            PrepareInputFileRecord(
                input_path="/input/a.oir",
                detected_type="oir",
                output_path="/out/a.tif",
                input_bytes=20_000_000,
                read_seconds=20.0,
                conversion_seconds=10.0,
                write_seconds=11.0,
                total_seconds=41.0,
                output_existed_before_run=True,
            )
        ],
    )

    notes = build_investigation_notes(profile)

    assert any("OIR projection is active" in note for note in notes)
    assert any("already had projected TIFF outputs" in note for note in notes)
    assert any("visited 8 directories" in note for note in notes)


def test_build_investigation_notes_for_cache_hit() -> None:
    scan = PrepareInputScanResult(
        input_dir="/input",
        directories_scanned=1,
        oir_files=[Path("/input/a.oir")],
        tiff_files=[],
        scan_seconds=0.01,
    )
    profile = PrepareInputProfile(
        input_dir="/input",
        action="oir_projection_cache_hit",
        scan=scan,
        engine="fiji",
        projection_seconds=0.05,
        file_records=[
            PrepareInputFileRecord(
                input_path="/input/a.oir",
                detected_type="oir",
                output_path="/out/a.tif",
                skipped=True,
                skip_reason="projection_cache_hit",
            )
        ],
    )

    notes = build_investigation_notes(profile)

    assert any("reused from oir_projection/" in note for note in notes)
    assert any("All projected TIFFs were reused" in note for note in notes)
    assert not any("processed again" in note for note in notes)


def test_format_prepare_input_report_lists_per_file_fields() -> None:
    profile = PrepareInputProfile(
        input_dir="/input",
        action="passthrough_tiff",
        scan=PrepareInputScanResult(
            input_dir="/input",
            directories_scanned=1,
            oir_files=[],
            tiff_files=[Path("/input/a.tif")],
            scan_seconds=0.01,
        ),
        file_records=[
            PrepareInputFileRecord(
                input_path="/input/a.tif",
                detected_type="tiff",
                output_path="/input/a.tif",
                input_bytes=1024,
            )
        ],
        investigation_notes=["No .oir files found"],
    )

    text = format_prepare_input_report(profile)

    assert "detected_type: tiff" in text
    assert "read_seconds:" in text
    assert "Investigation notes:" in text
