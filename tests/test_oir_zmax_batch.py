"""Tests for OIR Z-max batch workflow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from unittest.mock import MagicMock, patch

import pytest

from bioimage_pipeline.fiji_runner import default_fiji_headless, extract_fiji_errors
from bioimage_pipeline.oir_zmax_batch import (
    DEFAULT_MACRO_PATH,
    FIJI_OIR_COMMAND_LOG,
    FIJI_OIR_STDERR_LOG,
    FIJI_OIR_STDOUT_LOG,
    GENERATED_MACRO_NAME,
    build_manual_oir_zmax_macro,
    build_oir_zmax_macro,
    resolve_oir_projection_engine,
    run_oir_zmax_batch,
    write_manual_oir_zmax_macro,
    write_oir_zmax_generated_macro,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_SCRIPT = REPO_ROOT / "examples" / "run_oir_zmax_batch.py"
FOLDER_MACRO_FILE = REPO_ROOT / "examples" / "fiji_macros" / "stacking_zmax.ijm"


def test_resolve_oir_projection_engine_auto_prefers_fiji(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fiji_exe = tmp_path / "ImageJ-win64.exe"
    fiji_exe.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(
        "bioimage_pipeline.oir_zmax_batch.python_oir_dependencies_available",
        lambda: True,
    )

    assert resolve_oir_projection_engine("auto", fiji_executable=fiji_exe) == "fiji"
    assert resolve_oir_projection_engine("fiji", fiji_executable=fiji_exe) == "fiji"
    assert resolve_oir_projection_engine("python", fiji_executable=fiji_exe) == "python"


def test_build_oir_zmax_macro_uses_bioformats_windowless_importer(tmp_path: Path) -> None:
    macro_text = build_oir_zmax_macro(tmp_path / "input", tmp_path / "output")
    assert 'run("Bio-Formats Windowless Importer"' in macro_text
    assert 'run("Z Project..."' in macro_text
    assert 'saveAs("Tiff"' in macro_text


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


def test_run_oir_zmax_batch_default_engine_uses_fiji_when_python_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "sample.oir").write_bytes(b"")

    monkeypatch.setattr(
        "bioimage_pipeline.oir_zmax_batch.python_oir_dependencies_available",
        lambda: False,
    )
    fiji_exe = tmp_path / "ImageJ-win64.exe"
    fiji_exe.write_text("stub", encoding="utf-8")

    with patch(
        "bioimage_pipeline.oir_zmax_batch._run_fiji_oir_zmax_batch",
        return_value=MagicMock(
            input_dir=input_dir.resolve(),
            output_dir=output_dir.resolve(),
            engine="fiji",
            processed=["sample.tif"],
            failed=[],
            file_pairs=[],
            fiji_executable=fiji_exe.resolve(),
        ),
    ) as fiji_batch:
        result = run_oir_zmax_batch(
            input_dir,
            output_dir,
            fiji_executable=fiji_exe,
        )

    fiji_batch.assert_called_once()
    assert result.engine == "fiji"


def test_run_oir_zmax_batch_auto_prefers_fiji_when_python_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "sample.oir").write_bytes(b"")

    monkeypatch.setattr(
        "bioimage_pipeline.oir_zmax_batch.python_oir_dependencies_available",
        lambda: True,
    )
    fiji_exe = tmp_path / "ImageJ-win64.exe"
    fiji_exe.write_text("stub", encoding="utf-8")

    with patch(
        "bioimage_pipeline.oir_zmax_batch._run_fiji_oir_zmax_batch",
        return_value=MagicMock(
            input_dir=input_dir.resolve(),
            output_dir=output_dir.resolve(),
            engine="fiji",
            processed=["sample.tif"],
            failed=[],
            file_pairs=[],
            fiji_executable=fiji_exe.resolve(),
        ),
    ) as fiji_batch:
        result = run_oir_zmax_batch(
            input_dir,
            output_dir,
            engine="auto",
            fiji_executable=fiji_exe,
        )

    fiji_batch.assert_called_once()
    assert result.engine == "fiji"


def test_run_oir_zmax_batch_fiji_engine_never_uses_python_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "sample.oir").write_bytes(b"")

    monkeypatch.setattr(
        "bioimage_pipeline.oir_zmax_batch.python_oir_dependencies_available",
        lambda: True,
    )
    fiji_exe = tmp_path / "ImageJ-win64.exe"
    fiji_exe.write_text("stub", encoding="utf-8")

    with patch(
        "bioimage_pipeline.oir_zmax_batch.process_oir_file_python_timed",
    ) as process_timed, patch(
        "bioimage_pipeline.fiji_runner.run_fiji_macro",
    ) as run_macro:
        from bioimage_pipeline.fiji_runner import FijiRunResult

        (output_dir / "sample.tif").write_bytes(b"tiff")
        run_macro.return_value = FijiRunResult(
            command=[str(fiji_exe), "-macro"],
            returncode=0,
            stdout="",
            stderr="",
            macro_path=DEFAULT_MACRO_PATH,
            executable=fiji_exe.resolve(),
        )

        result = run_oir_zmax_batch(
            input_dir,
            output_dir,
            engine="fiji",
            fiji_executable=fiji_exe,
            logs_dir=tmp_path / "logs",
            force_oir_reproject=True,
            projection_method="median",
        )

    process_timed.assert_not_called()
    assert run_macro.call_args.args[0].name == GENERATED_MACRO_NAME
    macro_text = run_macro.call_args.args[0].read_text(encoding="utf-8")
    assert 'projection=[Median]' in macro_text
    assert result.engine == "fiji"


def test_run_oir_zmax_batch_python_engine_requires_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "sample.oir").write_bytes(b"")

    monkeypatch.setattr(
        "bioimage_pipeline.oir_zmax_batch.python_oir_dependencies_available",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match="aicsimageio/bfio"):
        run_oir_zmax_batch(input_dir, output_dir, engine="python")


def test_run_oir_zmax_batch_fiji_engine_uses_configured_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "sample.oir").write_bytes(b"")

    monkeypatch.setattr(
        "bioimage_pipeline.oir_zmax_batch.python_oir_dependencies_available",
        lambda: True,
    )
    fiji_exe = tmp_path / "ImageJ-win64.exe"
    fiji_exe.write_text("stub", encoding="utf-8")

    with patch(
        "bioimage_pipeline.fiji_runner.run_fiji_macro",
    ) as run_macro:
        from bioimage_pipeline.fiji_runner import FijiRunResult

        (output_dir / "sample.tif").write_bytes(b"tiff")
        run_macro.return_value = FijiRunResult(
            command=[str(fiji_exe), "--headless"],
            returncode=0,
            stdout="",
            stderr="",
            macro_path=DEFAULT_MACRO_PATH,
            executable=fiji_exe.resolve(),
        )

        result = run_oir_zmax_batch(
            input_dir,
            output_dir,
            engine="fiji",
            fiji_executable=fiji_exe,
            logs_dir=tmp_path / "logs",
            force_oir_reproject=True,
        )

    run_macro.assert_called_once()
    assert run_macro.call_args.args[0].name == GENERATED_MACRO_NAME
    assert run_macro.call_args.kwargs["fiji_executable"] == fiji_exe.resolve()
    assert run_macro.call_args.kwargs["headless"] is False
    assert result.engine == "fiji"
    assert result.fiji_executable == fiji_exe.resolve()
    assert result.output_dir == output_dir.resolve()


def test_run_oir_zmax_batch_fiji_engine_honors_fiji_executable_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "sample.oir").write_bytes(b"")

    fiji_exe = tmp_path / "ImageJ-win64.exe"
    fiji_exe.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("FIJI_EXECUTABLE", str(fiji_exe))

    with patch(
        "bioimage_pipeline.fiji_runner.run_fiji_macro",
    ) as run_macro:
        from bioimage_pipeline.fiji_runner import FijiRunResult

        run_macro.return_value = FijiRunResult(
            command=[str(fiji_exe), "--headless"],
            returncode=0,
            stdout="",
            stderr="",
            macro_path=DEFAULT_MACRO_PATH,
            executable=fiji_exe.resolve(),
        )
        (output_dir / "sample.tif").write_bytes(b"tiff")

        result = run_oir_zmax_batch(
            input_dir,
            output_dir,
            engine="fiji",
            logs_dir=tmp_path / "logs",
            force_oir_reproject=True,
        )

    assert run_macro.call_args.kwargs["fiji_executable"] == fiji_exe.resolve()
    assert result.fiji_executable == fiji_exe.resolve()


def test_build_oir_zmax_macro_embeds_paths_for_special_filenames(tmp_path: Path) -> None:
    input_dir = tmp_path / "input with spaces"
    output_dir = tmp_path / "output+folder"
    input_dir.mkdir()
    output_dir.mkdir()

    macro_text = build_oir_zmax_macro(input_dir, output_dir)

    assert "getArgument();" not in macro_text
    assert str(input_dir.resolve()).replace("\\", "/") in macro_text
    assert str(output_dir.resolve()).replace("\\", "/") in macro_text
    assert 'open=[" + inputPath + "]' in macro_text
    assert "[OIR] input path:" in macro_text
    assert "[OIR] saveAs called" in macro_text


@pytest.mark.parametrize(
    ("method", "fiji_label"),
    [
        ("max", "Max Intensity"),
        ("min", "Min Intensity"),
        ("average", "Average Intensity"),
        ("sum", "Sum Slices"),
        ("standard", "Standard Deviation"),
        ("median", "Median"),
    ],
)
def test_build_oir_zmax_macro_uses_selected_projection_method(
    tmp_path: Path,
    method: str,
    fiji_label: str,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    macro_text = build_oir_zmax_macro(
        input_dir,
        output_dir,
        projection_method=method,
    )

    assert f"projection=[{fiji_label}]" in macro_text
    assert f"Z Project ({fiji_label})" in macro_text


def test_projection_cache_is_stale_when_method_changes(tmp_path: Path) -> None:
    import time

    from bioimage_pipeline.oir_zmax_batch import (
        OirFilePair,
        projection_cache_is_fresh,
        write_stored_projection_method,
    )

    oir_path = tmp_path / "sample.oir"
    tif_path = tmp_path / "sample.tif"
    oir_path.write_bytes(b"oir-data")
    time.sleep(0.05)
    tif_path.write_bytes(b"tiff-data")
    write_stored_projection_method(tmp_path, "max")

    pair = OirFilePair(input_oir=oir_path, output_tif=tif_path)
    assert projection_cache_is_fresh(
        pair,
        projection_method="max",
        projection_output_dir=tmp_path,
    ) is True
    assert projection_cache_is_fresh(
        pair,
        projection_method="average",
        projection_output_dir=tmp_path,
    ) is False


def test_reconcile_fiji_outputs_renames_mismatched_tif(tmp_path: Path) -> None:
    from bioimage_pipeline.oir_zmax_batch import OirFilePair, _reconcile_fiji_outputs

    output_dir = tmp_path / "oir_projection"
    output_dir.mkdir()
    oir_path = tmp_path / "DQMI+4CHI+Ploy A_0007.oir"
    expected_tif = output_dir / "DQMI+4CHI+Ploy A_0007.tif"
    alternate_tif = output_dir / "alternate_name.tif"
    alternate_tif.write_bytes(b"tiff")

    pair = OirFilePair(input_oir=oir_path, output_tif=expected_tif)
    processed, failed, files_created, remapped = _reconcile_fiji_outputs(
        output_dir,
        [pair],
    )

    assert failed == []
    assert processed == ["DQMI+4CHI+Ploy A_0007.tif"]
    assert expected_tif.is_file()
    assert not alternate_tif.is_file()
    assert remapped == [
        {"from": str(alternate_tif), "to": str(expected_tif)},
    ]


def test_run_fiji_batch_writes_projection_logs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "oir_projection"
    logs_dir = tmp_path / "logs"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "sample.oir").write_bytes(b"")

    fiji_exe = tmp_path / "ImageJ-win64.exe"
    fiji_exe.write_text("stub", encoding="utf-8")

    with patch("bioimage_pipeline.fiji_runner.run_fiji_macro") as run_macro:
        from bioimage_pipeline.fiji_runner import FijiRunResult

        generated = write_oir_zmax_generated_macro(logs_dir, input_dir, output_dir)
        (output_dir / "sample.tif").write_bytes(b"tiff")
        run_macro.return_value = FijiRunResult(
            command=[str(fiji_exe), "-macro", str(generated)],
            returncode=0,
            stdout="[OIR] saveAs called",
            stderr="",
            macro_path=generated,
            executable=fiji_exe.resolve(),
        )

        result = run_oir_zmax_batch(
            input_dir,
            output_dir,
            engine="fiji",
            logs_dir=logs_dir,
            fiji_executable=fiji_exe,
            force_oir_reproject=True,
        )

    assert result.generated_macro_path == generated
    assert generated.is_file()
    assert (logs_dir / FIJI_OIR_STDOUT_LOG).read_text(encoding="utf-8") == "[OIR] saveAs called"
    assert (logs_dir / FIJI_OIR_STDERR_LOG).exists()
    assert (logs_dir / FIJI_OIR_COMMAND_LOG).exists()
    assert result.fiji_log_files["stdout"] == logs_dir / FIJI_OIR_STDOUT_LOG


def test_run_oir_zmax_batch_manual_macro_helper(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "sample.oir").write_bytes(b"")

    result = write_manual_oir_zmax_macro(input_dir, output_dir)
    macro_text = result.read_text(encoding="utf-8")

    assert result.is_file()
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


def test_projection_cache_is_fresh_when_tif_is_newer(tmp_path: Path) -> None:
    import time

    from bioimage_pipeline.oir_zmax_batch import OirFilePair, projection_cache_is_fresh

    oir_path = tmp_path / "sample.oir"
    tif_path = tmp_path / "sample.tif"
    oir_path.write_bytes(b"oir-data")
    time.sleep(0.05)
    tif_path.write_bytes(b"tiff-data")

    pair = OirFilePair(input_oir=oir_path, output_tif=tif_path)
    assert projection_cache_is_fresh(pair) is True


def test_projection_cache_is_stale_when_oir_is_newer(tmp_path: Path) -> None:
    import time

    from bioimage_pipeline.oir_zmax_batch import OirFilePair, projection_cache_is_fresh

    oir_path = tmp_path / "sample.oir"
    tif_path = tmp_path / "sample.tif"
    tif_path.write_bytes(b"tiff-data")
    time.sleep(0.05)
    oir_path.write_bytes(b"oir-data")
    # Ensure input mtime is clearly beyond tolerance.
    time.sleep(2.1)
    oir_path.write_bytes(b"oir-data-updated")

    pair = OirFilePair(input_oir=oir_path, output_tif=tif_path)
    assert projection_cache_is_fresh(pair) is False


def test_projection_cache_tolerates_small_input_mtime_lead(tmp_path: Path) -> None:
    import os

    from bioimage_pipeline.oir_zmax_batch import OirFilePair, projection_cache_is_fresh

    oir_path = tmp_path / "sample.oir"
    tif_path = tmp_path / "sample.tif"
    oir_path.write_bytes(b"oir")
    tif_path.write_bytes(b"tiff")
    base = 1_700_000_000.0
    os.utime(tif_path, (base, base))
    os.utime(oir_path, (base, base + 1.0))

    pair = OirFilePair(input_oir=oir_path, output_tif=tif_path)
    assert projection_cache_is_fresh(pair) is True

    os.utime(oir_path, (base, base + 3.0))
    assert projection_cache_is_fresh(pair) is False


def test_projection_cache_misses_on_zero_byte_tif(tmp_path: Path) -> None:
    from bioimage_pipeline.oir_zmax_batch import OirFilePair, projection_cache_is_fresh

    oir_path = tmp_path / "sample.oir"
    tif_path = tmp_path / "sample.tif"
    oir_path.write_bytes(b"oir")
    tif_path.write_bytes(b"")

    pair = OirFilePair(input_oir=oir_path, output_tif=tif_path)
    assert projection_cache_is_fresh(pair) is False


def test_run_oir_zmax_batch_python_skips_fresh_cache(tmp_path: Path) -> None:
    from bioimage_pipeline.oir_zmax_batch import run_oir_zmax_batch

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    oir_path = input_dir / "sample.oir"
    oir_path.write_bytes(b"oir")
    projected = output_dir / "sample.tif"
    projected.parent.mkdir(parents=True, exist_ok=True)
    projected.write_bytes(b"projected")

    with patch(
        "bioimage_pipeline.oir_zmax_batch.process_oir_file_python_timed",
    ) as process_timed:
        result = run_oir_zmax_batch(input_dir, output_dir, engine="python")

    process_timed.assert_not_called()
    assert result.cache_hits == ["sample.tif"]
    assert result.reprojected == []
    assert result.processed == ["sample.tif"]
    assert result.file_profiles[0].skipped is True
    assert result.file_profiles[0].skip_reason == "projection_cache_hit"


def test_run_oir_zmax_batch_python_force_reproject(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    oir_path = input_dir / "sample.oir"
    oir_path.write_bytes(b"oir")
    projected = output_dir / "sample.tif"
    projected.parent.mkdir(parents=True, exist_ok=True)
    projected.write_bytes(b"projected")

    with patch(
        "bioimage_pipeline.oir_zmax_batch.process_oir_file_python_timed",
        return_value=(projected.resolve(), MagicMock()),
    ) as process_timed:
        result = run_oir_zmax_batch(
            input_dir,
            output_dir,
            engine="python",
            force_oir_reproject=True,
        )

    process_timed.assert_called_once()
    assert result.cache_hits == []
    assert result.reprojected == ["sample.tif"]


def test_run_oir_zmax_batch_fiji_skips_macro_when_all_cached(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    logs_dir = tmp_path / "logs"
    input_dir.mkdir()
    output_dir.mkdir()
    oir_path = input_dir / "sample.oir"
    oir_path.write_bytes(b"oir")
    (output_dir / "sample.tif").write_bytes(b"projected")

    fiji_exe = tmp_path / "ImageJ-win64.exe"
    fiji_exe.write_text("stub", encoding="utf-8")

    with patch("bioimage_pipeline.fiji_runner.run_fiji_macro") as run_macro:
        result = run_oir_zmax_batch(
            input_dir,
            output_dir,
            engine="fiji",
            logs_dir=logs_dir,
            fiji_executable=fiji_exe,
        )

    run_macro.assert_not_called()
    assert result.cache_hits == ["sample.tif"]
    assert result.reprojected == []
    assert result.processed == ["sample.tif"]


def test_build_oir_zmax_macro_includes_cache_skip(tmp_path: Path) -> None:
    macro_text = build_oir_zmax_macro(tmp_path / "input", tmp_path / "output")
    assert "cache hit, skipping" in macro_text
    assert "lastModified()" in macro_text


def test_resolve_workflow_output_dir_relative_path(tmp_path: Path, monkeypatch) -> None:
    from bioimage_pipeline.analysis import resolve_workflow_output_dir

    monkeypatch.chdir(tmp_path)
    resolved = resolve_workflow_output_dir("output")

    assert resolved == (tmp_path / "output").resolve()
    assert resolved.is_dir()


def test_build_pre_cache_snapshot_lists_existing_tifs(tmp_path: Path) -> None:
    from bioimage_pipeline.oir_zmax_batch import OirFilePair, build_pre_cache_snapshot

    projection_dir = tmp_path / "oir_projection"
    projection_dir.mkdir()
    existing = projection_dir / "existing.tif"
    existing.write_bytes(b"x" * 100)
    oir_path = tmp_path / "input" / "sample.oir"
    oir_path.parent.mkdir()
    oir_path.write_bytes(b"oir")

    snapshot = build_pre_cache_snapshot(
        projection_dir,
        [
            OirFilePair(
                input_oir=oir_path.resolve(),
                output_tif=(projection_dir / "sample.tif").resolve(),
            )
        ],
    )

    assert snapshot["projection_output_dir"] == str(projection_dir.resolve())
    assert snapshot["projection_output_dir_exists"] is True
    assert len(snapshot["existing_tifs"]) == 1
    assert snapshot["existing_tifs"][0]["path"] == str(existing.resolve())
    assert snapshot["existing_tifs"][0]["size_bytes"] == 100
    assert snapshot["expected_pairs"] == [
        {
            "input_oir": str(oir_path.resolve()),
            "expected_output_tif": str((projection_dir / "sample.tif").resolve()),
        }
    ]


def test_log_projection_cache_decisions_writes_pre_cache_snapshot(tmp_path: Path) -> None:
    import json

    from bioimage_pipeline.oir_zmax_batch import (
        OirFilePair,
        build_pre_cache_snapshot,
        log_projection_cache_decisions,
    )

    projection_dir = tmp_path / "oir_projection"
    projection_dir.mkdir()
    oir_path = tmp_path / "sample.oir"
    oir_path.write_bytes(b"oir")
    pair = OirFilePair(
        input_oir=oir_path.resolve(),
        output_tif=(projection_dir / "sample.tif").resolve(),
    )
    snapshot = build_pre_cache_snapshot(projection_dir, [pair])
    logs_dir = tmp_path / "logs"

    log_projection_cache_decisions(
        [pair],
        engine="python",
        force_reproject=False,
        cached_pairs=[],
        to_project=[pair],
        skip_fiji=False,
        logs_dir=logs_dir,
        pre_cache_snapshot=snapshot,
    )

    debug_json = json.loads(
        (logs_dir / "oir_projection_cache_debug.json").read_text(encoding="utf-8")
    )
    debug_text = (logs_dir / "oir_projection_cache_debug.txt").read_text(encoding="utf-8")

    assert debug_json["pre_cache_snapshot"]["projection_output_dir"] == str(
        projection_dir.resolve()
    )
    assert debug_json["pre_cache_snapshot"]["expected_pairs"][0]["input_oir"] == str(
        oir_path.resolve()
    )
    assert "Pre-cache snapshot:" in debug_text
    assert "expected_output_tif:" in debug_text
