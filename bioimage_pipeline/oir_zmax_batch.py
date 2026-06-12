"""Batch Z-max projection for Olympus .oir files (Fiji macro parity).

Output-folder projection cache (mtime + 2s tolerance) and debug logs are
documented in docs/cellprofiler_workflow.md#oir-projection-cache-output-folder.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from bioimage_pipeline.z_projection import (
    OirPythonReadError,
    PYTHON_OIR_MISSING_DEPS_MESSAGE,
    format_oir_read_dependency_error,
    iter_oir_files,
    oir_output_path,
    process_oir_file_python,
    process_oir_file_python_timed,
    python_oir_dependencies_available,
)

DEFAULT_MACRO_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "fiji_macros"
    / "stacking_zmax.ijm"
)
GENERATED_MACRO_NAME = "stacking_zmax_generated.ijm"
FIJI_OIR_STDOUT_LOG = "fiji_oir_projection_stdout.log"
FIJI_OIR_STDERR_LOG = "fiji_oir_projection_stderr.log"
FIJI_OIR_COMMAND_LOG = "fiji_oir_projection_command.txt"
OIR_PROJECTION_CACHE_DEBUG_LOG = "oir_projection_cache_debug.txt"
# Windows/FAT32 and float rounding can make projected TIFFs look slightly older
# than source .oir files even when nothing changed.
CACHE_MTIME_TOLERANCE_SECONDS = 2.0

logger = logging.getLogger(__name__)

OirZmaxEngine = Literal["fiji", "python", "auto"]


@dataclass
class OirFilePair:
    """One input ``.oir`` file and its projected output TIFF path."""

    input_oir: Path
    output_tif: Path

    @property
    def bioformats_import_command(self) -> str:
        """The exact Fiji macro command used to import this file."""
        options = (
            f"open=[{_ijm_path(self.input_oir)}] autoscale "
            "view=Hyperstack stack_format=Default"
        )
        return f'run("Bio-Formats Windowless Importer", "{options}");'


@dataclass
class OirZmaxBatchResult:
    """Summary of a completed OIR Z-max batch run."""

    input_dir: Path
    output_dir: Path
    engine: str
    fiji_executable: Path | None = None
    fiji_headless: bool | None = None
    manual_macro_path: Path | None = None
    generated_macro_path: Path | None = None
    fiji_log_files: dict[str, Path] = field(default_factory=dict)
    fiji_returncode: int | None = None
    files_created: list[str] = field(default_factory=list)
    remapped_outputs: list[dict[str, str]] = field(default_factory=list)
    file_pairs: list[OirFilePair] = field(default_factory=list)
    processed: list[str] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    file_profiles: list[Any] = field(default_factory=list)
    cache_hits: list[str] = field(default_factory=list)
    reprojected: list[str] = field(default_factory=list)
    force_oir_reproject: bool = False


def _normalize_input_dir(input_dir: str | Path) -> Path:
    path = Path(input_dir)
    if not path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {path}")
    return path.resolve()


def _build_file_pairs(oir_files: list[Path], output_dir: Path) -> list[OirFilePair]:
    return [
        OirFilePair(
            input_oir=oir_path.resolve(),
            output_tif=oir_output_path(oir_path, output_dir).resolve(),
        )
        for oir_path in oir_files
    ]


def _ijm_path(path: Path) -> str:
    """Return an absolute path string safe for ImageJ macro string literals."""
    return str(path.resolve()).replace("\\", "/").replace('"', '\\"')


    force_oir_reproject: bool = False


@dataclass
class ProjectionCacheDecision:
    """Debug record for one OIR projection cache evaluation."""

    input_oir: str
    expected_output_tif: str
    output_tif_exists: bool
    output_tif_size: int | None
    input_mtime: float | None
    output_mtime: float | None
    input_mtime_ns: int | None
    output_mtime_ns: int | None
    cache_is_fresh: bool
    force_oir_reproject: bool
    cache_reason: str


def evaluate_projection_cache(
    pair: OirFilePair,
    *,
    force_reproject: bool = False,
) -> ProjectionCacheDecision:
    """Evaluate cache freshness and return a debug-friendly decision record."""
    input_oir = pair.input_oir.resolve()
    output_tif = pair.output_tif.resolve()
    base = {
        "input_oir": str(input_oir),
        "expected_output_tif": str(output_tif),
        "output_tif_exists": output_tif.is_file(),
        "output_tif_size": None,
        "input_mtime": None,
        "output_mtime": None,
        "input_mtime_ns": None,
        "output_mtime_ns": None,
        "force_oir_reproject": force_reproject,
    }

    if force_reproject:
        return ProjectionCacheDecision(
            **base,
            cache_is_fresh=False,
            cache_reason="force_oir_reproject is True",
        )

    if not input_oir.is_file():
        return ProjectionCacheDecision(
            **base,
            cache_is_fresh=False,
            cache_reason="input .oir file is missing",
        )

    if not output_tif.is_file():
        return ProjectionCacheDecision(
            **base,
            cache_is_fresh=False,
            cache_reason="expected projected TIFF is missing",
        )

    try:
        input_stat = input_oir.stat()
        output_stat = output_tif.stat()
    except OSError as exc:
        return ProjectionCacheDecision(
            **base,
            cache_is_fresh=False,
            cache_reason=f"stat failed: {exc}",
        )

    base.update(
        {
            "output_tif_size": output_stat.st_size,
            "input_mtime": input_stat.st_mtime,
            "output_mtime": output_stat.st_mtime,
            "input_mtime_ns": input_stat.st_mtime_ns,
            "output_mtime_ns": output_stat.st_mtime_ns,
        }
    )

    if output_stat.st_size <= 0:
        return ProjectionCacheDecision(
            **base,
            cache_is_fresh=False,
            cache_reason="expected projected TIFF is zero bytes",
        )

    if output_stat.st_mtime_ns >= input_stat.st_mtime_ns:
        return ProjectionCacheDecision(
            **base,
            cache_is_fresh=True,
            cache_reason="output mtime_ns >= input mtime_ns",
        )

    mtime_delta = input_stat.st_mtime - output_stat.st_mtime
    if mtime_delta <= CACHE_MTIME_TOLERANCE_SECONDS:
        return ProjectionCacheDecision(
            **base,
            cache_is_fresh=True,
            cache_reason=(
                f"output within {CACHE_MTIME_TOLERANCE_SECONDS:.1f}s of input mtime "
                f"(delta={mtime_delta:.3f}s)"
            ),
        )

    return ProjectionCacheDecision(
        **base,
        cache_is_fresh=False,
        cache_reason=(
            f"input is {mtime_delta:.3f}s newer than output "
            f"(>{CACHE_MTIME_TOLERANCE_SECONDS:.1f}s tolerance)"
        ),
    )


def projection_cache_is_fresh(
    pair: OirFilePair,
    *,
    force_reproject: bool = False,
) -> bool:
    """Return whether a projected TIFF in the output folder is up to date."""
    return evaluate_projection_cache(
        pair,
        force_reproject=force_reproject,
    ).cache_is_fresh


def format_projection_cache_debug_report(
    decisions: list[ProjectionCacheDecision],
    *,
    engine: str,
    cached_count: int,
    to_project_count: int,
    skip_fiji: bool,
) -> str:
    """Format cache decisions for logs and on-disk debugging."""
    lines = [
        "OIR projection cache debug:",
        f"  engine: {engine}",
        f"  force_oir_reproject: {decisions[0].force_oir_reproject if decisions else False}",
        f"  cache_hits: {cached_count}",
        f"  to_project: {to_project_count}",
        f"  skip_fiji_launch: {skip_fiji}",
        "",
        "Per-pair decisions:",
    ]
    for decision in decisions:
        lines.extend(
            [
                f"  - input_oir: {decision.input_oir}",
                f"    expected_output_tif: {decision.expected_output_tif}",
                f"    output_tif_exists: {decision.output_tif_exists}",
                f"    output_tif_size: {decision.output_tif_size}",
                f"    input_mtime: {decision.input_mtime}",
                f"    output_mtime: {decision.output_mtime}",
                f"    input_mtime_ns: {decision.input_mtime_ns}",
                f"    output_mtime_ns: {decision.output_mtime_ns}",
                f"    cache_is_fresh: {decision.cache_is_fresh}",
                f"    force_oir_reproject: {decision.force_oir_reproject}",
                f"    cache_reason: {decision.cache_reason}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def log_projection_cache_decisions(
    file_pairs: list[OirFilePair],
    *,
    engine: str,
    force_reproject: bool,
    cached_pairs: list[OirFilePair],
    to_project: list[OirFilePair],
    skip_fiji: bool,
    logs_dir: Path | None = None,
) -> list[ProjectionCacheDecision]:
    """Log per-pair cache decisions and optionally persist them under logs_dir."""
    decisions = [
        evaluate_projection_cache(pair, force_reproject=force_reproject)
        for pair in file_pairs
    ]
    report = format_projection_cache_debug_report(
        decisions,
        engine=engine,
        cached_count=len(cached_pairs),
        to_project_count=len(to_project),
        skip_fiji=skip_fiji,
    )
    logger.info("\n%s", report.rstrip())
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        debug_path = logs_dir / OIR_PROJECTION_CACHE_DEBUG_LOG
        debug_path.write_text(report, encoding="utf-8")
        summary_path = logs_dir / "oir_projection_cache_debug.json"
        summary_path.write_text(
            json.dumps(
                {
                    "engine": engine,
                    "force_oir_reproject": force_reproject,
                    "cache_hits": len(cached_pairs),
                    "to_project": len(to_project),
                    "skip_fiji_launch": skip_fiji,
                    "decisions": [asdict(decision) for decision in decisions],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return decisions


def partition_projection_pairs(
    file_pairs: list[OirFilePair],
    *,
    force_reproject: bool = False,
) -> tuple[list[OirFilePair], list[OirFilePair]]:
    """Split pairs into ``(cached, to_project)`` using output-folder mtime cache."""
    if force_reproject:
        return [], list(file_pairs)
    cached: list[OirFilePair] = []
    to_project: list[OirFilePair] = []
    for pair in file_pairs:
        if projection_cache_is_fresh(pair, force_reproject=force_reproject):
            cached.append(pair)
        else:
            to_project.append(pair)
    return cached, to_project


def resolve_oir_projection_engine(
    engine: OirZmaxEngine | None,
    *,
    fiji_executable: str | Path | None = None,
) -> str:
    """Resolve ``auto``/``None`` to ``python`` or ``fiji`` with clear failures."""
    from bioimage_pipeline.fiji_runner import find_fiji_executable

    if engine == "python":
        return "python"
    if engine == "fiji":
        return "fiji"

    if python_oir_dependencies_available():
        return "python"
    if find_fiji_executable(fiji_executable) is not None:
        return "fiji"

    raise RuntimeError(
        "No OIR projection engine is available on this machine. "
        f"{PYTHON_OIR_MISSING_DEPS_MESSAGE} "
        "Alternatively, configure a Fiji executable in the Run workflow panel "
        "or set FIJI_EXECUTABLE and choose OIR projection engine Fiji."
    )


def build_oir_zmax_macro(input_dir: Path, output_dir: Path) -> str:
    """Build the OIR Z-max macro with embedded paths and diagnostic logging.

    Paths are embedded directly in the macro (same approach as the working
    manual Fiji macro) so filenames with spaces or ``+`` are not split by
    ``getArgument()`` or ``|``-joined CLI macro arguments.
    """
    input_text = _ijm_path(input_dir)
    output_text = _ijm_path(output_dir)
    return f"""// Auto-generated OIR Z-max macro (Stacking+Drectly.ijm parity).
// Embedded input/output paths — do not rely on getArgument() for paths.
// Bio-Formats Windowless Importer → Z Project Max Intensity → saveAs TIFF.

input = ensureTrailingSeparator("{input_text}");
output = ensureTrailingSeparator("{output_text}");

print("[OIR] macro input folder: " + input);
print("[OIR] macro output folder: " + output);

setBatchMode(true);
if (!File.exists(output)) {{
    File.makeDirectory(output);
}}

function ensureTrailingSeparator(path) {{
    if (endsWith(path, "/") || endsWith(path, "\\\\")) {{
        return path;
    }}
    return path + "/";
}}

function processFolder(dir) {{
    dir = ensureTrailingSeparator(dir);
    list = getFileList(dir);
    for (i = 0; i < list.length; i++) {{
        name = list[i];
        lower = toLowerCase(name);
        if (endsWith(lower, ".oir")) {{
            processFile(dir, output, name);
        }} else if (endsWith(name, "/")) {{
            processFolder(dir + name);
        }} else {{
            print("[OIR] skipping non-image file: " + dir + name);
        }}
    }}
}}

function processFile(inputFolder, outputFolder, file) {{
    inputFolder = ensureTrailingSeparator(inputFolder);
    outputFolder = ensureTrailingSeparator(outputFolder);
    inputPath = inputFolder + file;
    print("[OIR] input path: " + inputPath);
    print("[OIR] output folder: " + outputFolder);
    if (!File.exists(inputPath)) {{
        print("[OIR] file not found: " + inputPath);
        return;
    }}
    if (!File.exists(outputFolder)) {{
        File.makeDirectory(outputFolder);
    }}

    saveName = replace(file, ".oir", ".tif");
    savePath = outputFolder + saveName;
    if (File.exists(savePath)) {{
        oirMod = (new java.io.File(inputPath)).lastModified();
        tifMod = (new java.io.File(savePath)).lastModified();
        if (tifMod >= oirMod) {{
            print("[OIR] cache hit, skipping: " + inputPath);
            return;
        }}
    }}

    importOptions = "open=[" + inputPath + "] autoscale view=Hyperstack stack_format=Default";
    print("[OIR] Bio-Formats import options: " + importOptions);
    tReadStart = getTime();
    run("Bio-Formats Windowless Importer", importOptions);
    tReadEnd = getTime();
    print("[OIR] timing read_seconds: " + ((tReadEnd - tReadStart) / 1000.0));
    print("[OIR] nImages after Bio-Formats import: " + nImages);
    if (nImages == 0) {{
        print("[OIR] import failed: " + inputPath);
        return;
    }}
    selectImage(1);
    print("[OIR] current image title: " + getTitle());
    if (is("hyperstack")) {{
        print("[OIR] stack slices: " + nSlices);
    }} else {{
        print("[OIR] stack slices: n/a (not a hyperstack)");
    }}
    tConvStart = getTime();
    run("Z Project...", "projection=[Max Intensity]");
    tConvEnd = getTime();
    print("[OIR] timing conversion_seconds: " + ((tConvEnd - tConvStart) / 1000.0));
    print("[OIR] Z Project completed");
    print("[OIR] saveAs target: " + savePath);
    tWriteStart = getTime();
    saveAs("Tiff", savePath);
    tWriteEnd = getTime();
    print("[OIR] timing write_seconds: " + ((tWriteEnd - tWriteStart) / 1000.0));
    print("[OIR] saveAs called");
    if (File.exists(savePath)) {{
        print("[OIR] saved file verified on disk: " + savePath);
    }} else {{
        print("[OIR] WARNING: saved file not found on disk: " + savePath);
    }}
    close("*");
}}

processFolder(input);
setBatchMode(false);
"""


def build_manual_oir_zmax_macro(input_dir: Path, output_dir: Path) -> str:
    """Build a Fiji macro that users run manually from the Fiji GUI."""
    return build_oir_zmax_macro(input_dir, output_dir)


def write_oir_zmax_generated_macro(
    logs_dir: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    """Write the generated OIR Z-max macro into the workflow logs folder."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    macro_path = logs_dir / GENERATED_MACRO_NAME
    macro_path.write_text(
        build_oir_zmax_macro(input_dir, output_dir),
        encoding="utf-8",
    )
    return macro_path.resolve()


def write_manual_oir_zmax_macro(input_dir: Path, output_dir: Path) -> Path:
    """Write the manual-run Fiji macro into the output directory."""
    macro_path = output_dir / "run_oir_zmax_manual.ijm"
    macro_path.write_text(
        build_manual_oir_zmax_macro(input_dir, output_dir),
        encoding="utf-8",
    )
    return macro_path.resolve()


def _cache_hit_record(pair: OirFilePair, *, engine: str) -> Any:
    from bioimage_pipeline.prepare_input_profile import (
        PrepareInputFileRecord,
        detect_file_type,
    )

    input_path = pair.input_oir.resolve()
    output_path = pair.output_tif.resolve()
    output_bytes = output_path.stat().st_size if output_path.is_file() else None
    return PrepareInputFileRecord(
        input_path=str(input_path),
        detected_type=detect_file_type(input_path),
        output_path=str(output_path),
        input_bytes=input_path.stat().st_size if input_path.is_file() else 0,
        output_bytes=output_bytes,
        skipped=True,
        skip_reason="projection_cache_hit",
        output_existed_before_run=True,
        notes=[f"Reused projected TIFF from output folder ({engine} cache)."],
    )


def _build_cache_only_result(
    *,
    input_path: Path,
    output_path: Path,
    engine: str,
    file_pairs: list[OirFilePair],
    cached_pairs: list[OirFilePair],
    force_oir_reproject: bool,
) -> OirZmaxBatchResult:
    cache_hits = [pair.output_tif.name for pair in cached_pairs]
    file_profiles = [_cache_hit_record(pair, engine=engine) for pair in cached_pairs]
    return OirZmaxBatchResult(
        input_dir=input_path,
        output_dir=output_path.resolve(),
        engine=engine,
        file_pairs=file_pairs,
        processed=cache_hits,
        failed=[],
        file_profiles=file_profiles,
        cache_hits=cache_hits,
        reprojected=[],
        force_oir_reproject=force_oir_reproject,
    )


def _prepare_input_record_for_pair(
    pair: OirFilePair,
    *,
    engine: str,
    output_existed_before_run: bool = False,
    notes: list[str] | None = None,
) -> Any:
    from bioimage_pipeline.prepare_input_profile import (
        PrepareInputFileRecord,
        detect_file_type,
    )

    input_path = pair.input_oir.resolve()
    input_bytes = input_path.stat().st_size if input_path.is_file() else 0
    record = PrepareInputFileRecord(
        input_path=str(input_path),
        detected_type=detect_file_type(input_path),
        output_path=str(pair.output_tif.resolve()),
        input_bytes=input_bytes,
        output_existed_before_run=output_existed_before_run,
        notes=list(notes or []),
    )
    if output_existed_before_run:
        record.notes.append(
            "Projected output already existed before this run but was processed again."
        )
    record.notes.append(f"Processed via {engine} OIR projection engine.")
    return record


def _output_existed_map(file_pairs: list[OirFilePair]) -> dict[str, bool]:
    return {
        str(pair.input_oir.resolve()): pair.output_tif.is_file()
        for pair in file_pairs
    }


def _failure_entry(
    pair: OirFilePair,
    *,
    error: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "file": pair.input_oir.name,
        "input_oir": str(pair.input_oir),
        "output_tif": str(pair.output_tif),
        "import_command": pair.bioformats_import_command,
        "error": error,
    }
    if extra:
        entry.update(extra)
    return entry


def _list_output_tiffs(output_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )


def _reconcile_fiji_outputs(
    output_dir: Path,
    file_pairs: list[OirFilePair],
) -> tuple[list[str], list[dict[str, Any]], list[str], list[dict[str, str]]]:
    """Match created TIFFs to expected outputs and rename when needed."""
    actual_files = _list_output_tiffs(output_dir)
    files_created = [str(path) for path in actual_files]
    remapped: list[dict[str, str]] = []
    processed: list[str] = []
    failed: list[dict[str, Any]] = []
    used: set[Path] = set()
    unmatched_pairs: list[OirFilePair] = []

    for pair in file_pairs:
        if pair.output_tif.is_file():
            processed.append(pair.output_tif.name)
            used.add(pair.output_tif.resolve())
            continue

        expected_stem = pair.input_oir.stem
        candidates = [
            path
            for path in actual_files
            if path.resolve() not in used and path.stem == expected_stem
        ]
        if not candidates:
            candidates = [
                path
                for path in actual_files
                if path.resolve() not in used
                and path.stem.lower() == expected_stem.lower()
            ]

        if len(candidates) == 1:
            source = candidates[0]
            if source.resolve() != pair.output_tif.resolve():
                pair.output_tif.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(pair.output_tif))
                remapped.append({"from": str(source), "to": str(pair.output_tif)})
            processed.append(pair.output_tif.name)
            used.add(pair.output_tif.resolve())
            continue

        if len(candidates) > 1:
            failed.append(
                _failure_entry(
                    pair,
                    error=(
                        "Multiple TIFF files match the expected output stem "
                        f"{expected_stem!r}."
                    ),
                    extra={"candidate_tifs": [str(path) for path in candidates]},
                )
            )
            continue

        unmatched_pairs.append(pair)

    unused_files = [
        path for path in actual_files if path.resolve() not in used
    ]
    if unmatched_pairs and unused_files:
        for pair, source in zip(unmatched_pairs, unused_files, strict=False):
            if source.resolve() != pair.output_tif.resolve():
                pair.output_tif.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(pair.output_tif))
                remapped.append({"from": str(source), "to": str(pair.output_tif)})
            processed.append(pair.output_tif.name)
            used.add(pair.output_tif.resolve())

        for pair in unmatched_pairs[len(unused_files) :]:
            failed.append(
                _failure_entry(
                    pair,
                    error="Expected output file was not created.",
                    extra={"files_created_in_output_dir": files_created},
                )
            )
    else:
        for pair in unmatched_pairs:
            failed.append(
                _failure_entry(
                    pair,
                    error="Expected output file was not created.",
                    extra={"files_created_in_output_dir": files_created},
                )
            )

    return processed, failed, files_created, remapped


def _resolve_fiji_launch_executable(executable: Path) -> Path:
    """Prefer the ImageJ1 launcher for Bio-Formats OIR import when available."""
    from bioimage_pipeline.fiji_runner import suggest_imagej1_executable

    suggested = suggest_imagej1_executable(executable)
    if suggested is not None:
        return suggested
    return executable


def _default_oir_fiji_headless(explicit: bool | None) -> bool:
    """Bio-Formats ``.oir`` import is unreliable with ``--headless``; default off."""
    if explicit is not None:
        return explicit
    return False


def _write_fiji_oir_projection_logs(
    logs_dir: Path,
    *,
    command: list[str],
    stdout: str,
    stderr: str,
    returncode: int,
) -> dict[str, Path]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_files = {
        "stdout": logs_dir / FIJI_OIR_STDOUT_LOG,
        "stderr": logs_dir / FIJI_OIR_STDERR_LOG,
        "command": logs_dir / FIJI_OIR_COMMAND_LOG,
    }
    log_files["stdout"].write_text(stdout, encoding="utf-8")
    log_files["stderr"].write_text(stderr, encoding="utf-8")
    log_files["command"].write_text(" ".join(command), encoding="utf-8")
    (logs_dir / "fiji_oir_projection_exit_code.txt").write_text(
        str(returncode),
        encoding="utf-8",
    )
    return log_files


def _run_fiji_oir_zmax_batch(
    input_path: Path,
    output_path: Path,
    file_pairs: list[OirFilePair],
    *,
    logs_dir: Path | None = None,
    fiji_executable: str | Path | None = None,
    fiji_headless: bool | None = None,
    fiji_timeout: float | None = None,
    force_oir_reproject: bool = False,
) -> OirZmaxBatchResult:
    from bioimage_pipeline.fiji_runner import (
        find_fiji_executable,
        fiji_not_found_message,
        format_fiji_error_summary,
        run_fiji_macro,
    )

    cached_pairs, to_project = partition_projection_pairs(
        file_pairs,
        force_reproject=force_oir_reproject,
    )
    cache_hits = [pair.output_tif.name for pair in cached_pairs]
    file_profiles = [_cache_hit_record(pair, engine="fiji") for pair in cached_pairs]
    skip_fiji = not to_project

    log_projection_cache_decisions(
        file_pairs,
        engine="fiji",
        force_reproject=force_oir_reproject,
        cached_pairs=cached_pairs,
        to_project=to_project,
        skip_fiji=skip_fiji,
        logs_dir=logs_dir,
    )

    if skip_fiji:
        logger.info(
            "Skipping Fiji OIR projection: all %d file(s) satisfied output-folder cache.",
            len(cached_pairs),
        )
        return _build_cache_only_result(
            input_path=input_path,
            output_path=output_path,
            engine="fiji",
            file_pairs=file_pairs,
            cached_pairs=cached_pairs,
            force_oir_reproject=force_oir_reproject,
        )

    executable = find_fiji_executable(fiji_executable)
    if executable is None:
        raise FileNotFoundError(fiji_not_found_message())

    launch_executable = _resolve_fiji_launch_executable(executable)
    use_headless = _default_oir_fiji_headless(fiji_headless)
    output_existed_before = _output_existed_map(to_project)

    if logs_dir is None:
        logs_dir = output_path.parent / "logs"
    generated_macro_path = write_oir_zmax_generated_macro(
        logs_dir,
        input_path,
        output_path,
    )

    run_result = run_fiji_macro(
        generated_macro_path,
        fiji_executable=launch_executable,
        headless=use_headless,
        timeout=fiji_timeout,
    )
    fiji_log_files = _write_fiji_oir_projection_logs(
        logs_dir,
        command=run_result.command,
        stdout=run_result.stdout,
        stderr=run_result.stderr,
        returncode=run_result.returncode,
    )

    processed, failed, files_created, remapped = _reconcile_fiji_outputs(
        output_path,
        to_project,
    )
    from bioimage_pipeline.prepare_input_profile import parse_fiji_oir_file_records

    reprojected = list(processed)
    projected_profiles = parse_fiji_oir_file_records(run_result.stdout)
    parsed_by_input = {record.input_path: record for record in projected_profiles}
    for pair in to_project:
        key = str(pair.input_oir.resolve())
        existed_before = output_existed_before.get(key, False)
        record = parsed_by_input.get(key)
        if record is None:
            projected_profiles.append(
                _prepare_input_record_for_pair(
                    pair,
                    engine="fiji",
                    output_existed_before_run=existed_before,
                    notes=["No macro timing lines found."],
                )
            )
            continue
        record.output_path = record.output_path or str(pair.output_tif.resolve())
        record.output_existed_before_run = existed_before
        if existed_before:
            record.notes.append(
                "Projected output already existed before this run but was processed again."
            )
        if pair.output_tif.is_file() and record.output_bytes is None:
            record.output_bytes = pair.output_tif.stat().st_size

    file_profiles.extend(projected_profiles)
    all_processed = cache_hits + reprojected
    fiji_error = (
        format_fiji_error_summary(run_result)
        if not run_result.succeeded or failed
        else None
    )
    if fiji_error:
        for entry in failed:
            if entry.get("error") == "Expected output file was not created.":
                entry["error"] = (
                    "Expected output file was not created.\n"
                    f"Fiji output:\n{fiji_error}"
                )

    return OirZmaxBatchResult(
        input_dir=input_path,
        output_dir=output_path.resolve(),
        engine="fiji",
        fiji_executable=launch_executable,
        fiji_headless=use_headless,
        manual_macro_path=None,
        generated_macro_path=generated_macro_path,
        fiji_log_files=fiji_log_files,
        fiji_returncode=run_result.returncode,
        files_created=files_created,
        remapped_outputs=remapped,
        file_pairs=file_pairs,
        processed=all_processed,
        failed=failed,
        file_profiles=file_profiles,
        cache_hits=cache_hits,
        reprojected=reprojected,
        force_oir_reproject=force_oir_reproject,
    )


def run_oir_zmax_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    engine: OirZmaxEngine = "auto",
    logs_dir: str | Path | None = None,
    fiji_executable: str | Path | None = None,
    fiji_headless: bool | None = None,
    fiji_timeout: float | None = None,
    force_oir_reproject: bool = False,
) -> OirZmaxBatchResult:
    """Z-max project every ``.oir`` file under *input_dir* into *output_dir*.

    When *engine* is ``fiji``, writes a generated macro with embedded paths to
    ``logs_dir/stacking_zmax_generated.ijm`` and runs it via Fiji/ImageJ.
    When *engine* is ``python``, reads ``.oir`` files with aicsimageio/bfio.
    ``auto`` prefers Python when aicsimageio is available, otherwise Fiji.
    """
    input_path = _normalize_input_dir(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logs_path = Path(logs_dir) if logs_dir is not None else None

    resolved_engine = resolve_oir_projection_engine(
        engine,
        fiji_executable=fiji_executable,
    )
    oir_files = list(iter_oir_files(input_path))
    file_pairs = _build_file_pairs(oir_files, output_path)
    cached_pairs, to_project = partition_projection_pairs(
        file_pairs,
        force_reproject=force_oir_reproject,
    )

    if resolved_engine == "fiji":
        return _run_fiji_oir_zmax_batch(
            input_path,
            output_path,
            file_pairs,
            logs_dir=logs_path,
            fiji_executable=fiji_executable,
            fiji_headless=fiji_headless,
            fiji_timeout=fiji_timeout,
            force_oir_reproject=force_oir_reproject,
        )

    log_projection_cache_decisions(
        file_pairs,
        engine="python",
        force_reproject=force_oir_reproject,
        cached_pairs=cached_pairs,
        to_project=to_project,
        skip_fiji=False,
        logs_dir=logs_path,
    )

    cache_hits = [pair.output_tif.name for pair in cached_pairs]
    file_profiles = [_cache_hit_record(pair, engine="python") for pair in cached_pairs]

    if not to_project:
        return _build_cache_only_result(
            input_path=input_path,
            output_path=output_path,
            engine="python",
            file_pairs=file_pairs,
            cached_pairs=cached_pairs,
            force_oir_reproject=force_oir_reproject,
        )

    if not python_oir_dependencies_available():
        raise RuntimeError(PYTHON_OIR_MISSING_DEPS_MESSAGE)

    processed: list[str] = list(cache_hits)
    failed: list[dict[str, Any]] = []
    reprojected: list[str] = []
    for pair in to_project:
        try:
            _output_path, record = process_oir_file_python_timed(
                pair.input_oir,
                output_path,
            )
            file_profiles.append(record)
            if pair.output_tif.is_file():
                processed.append(pair.output_tif.name)
                reprojected.append(pair.output_tif.name)
            else:
                failed.append(
                    _failure_entry(
                        pair,
                        error="Expected output file was not created.",
                    )
                )
        except OirPythonReadError as exc:
            failure = _failure_entry(pair, error=str(exc))
            failed.append(failure)
            file_profiles.append(
                _prepare_input_record_for_pair(
                    pair,
                    engine="python",
                    output_existed_before_run=pair.output_tif.is_file(),
                    notes=[str(exc)],
                )
            )
        except Exception as exc:
            message = format_oir_read_dependency_error(exc)
            failed.append(_failure_entry(pair, error=message))
            file_profiles.append(
                _prepare_input_record_for_pair(
                    pair,
                    engine="python",
                    output_existed_before_run=pair.output_tif.is_file(),
                    notes=[message],
                )
            )

    return OirZmaxBatchResult(
        input_dir=input_path,
        output_dir=output_path.resolve(),
        engine="python",
        fiji_executable=None,
        fiji_headless=None,
        manual_macro_path=None,
        generated_macro_path=None,
        file_pairs=file_pairs,
        processed=processed,
        failed=failed,
        file_profiles=file_profiles,
        cache_hits=cache_hits,
        reprojected=reprojected,
        force_oir_reproject=force_oir_reproject,
    )


def oir_output_name(oir_path: str | Path) -> str:
    """Output basename for an input OIR path (macro naming rules)."""
    from bioimage_pipeline.z_projection import oir_output_filename

    return oir_output_filename(Path(oir_path).name)
