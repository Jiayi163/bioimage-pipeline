"""Profiling helpers for the prepare_input workflow stage."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TIFF_GLOB_PATTERNS = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")


@dataclass
class PrepareInputFileRecord:
    """Per-file timing for one prepare_input candidate."""

    input_path: str
    detected_type: str
    output_path: str | None = None
    input_bytes: int = 0
    output_bytes: int | None = None
    read_seconds: float = 0.0
    conversion_seconds: float = 0.0
    write_seconds: float = 0.0
    total_seconds: float = 0.0
    skipped: bool = False
    skip_reason: str | None = None
    output_existed_before_run: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class PrepareInputScanResult:
    """Folder scan statistics collected before prepare_input work."""

    input_dir: str
    directories_scanned: int
    oir_files: list[Path]
    tiff_files: list[Path]
    scan_seconds: float

    @property
    def oir_count(self) -> int:
        return len(self.oir_files)

    @property
    def tiff_count(self) -> int:
        return len(self.tiff_files)


@dataclass
class PrepareInputProfile:
    """Full prepare_input profiling report."""

    input_dir: str
    action: str
    scan: PrepareInputScanResult
    engine: str | None = None
    projection_output_dir: str | None = None
    projection_seconds: float = 0.0
    file_records: list[PrepareInputFileRecord] = field(default_factory=list)
    investigation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scan"] = {
            **asdict(self.scan),
            "oir_files": [str(path) for path in self.scan.oir_files],
            "tiff_files": [str(path) for path in self.scan.tiff_files],
        }
        return payload


def _file_size(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    return path.stat().st_size


def scan_prepare_input_folder(root: str | Path) -> PrepareInputScanResult:
    """Scan *root* for ``.oir`` (recursive) and TIFF files (top level only)."""
    root_path = Path(root).resolve()
    started = time.perf_counter()
    directories_scanned = 0
    oir_files: list[Path] = []

    def scan_oir_recursive(directory: Path) -> None:
        nonlocal directories_scanned
        directories_scanned += 1
        for entry in sorted(directory.iterdir()):
            if entry.is_file() and entry.suffix.lower() == ".oir":
                oir_files.append(entry.resolve())
            elif entry.is_dir():
                scan_oir_recursive(entry)

    tiff_seen: set[Path] = set()
    tiff_files: list[Path] = []
    for pattern in _TIFF_GLOB_PATTERNS:
        for path in sorted(root_path.glob(pattern)):
            if path.is_file():
                resolved = path.resolve()
                if resolved not in tiff_seen:
                    tiff_seen.add(resolved)
                    tiff_files.append(resolved)

    scan_oir_recursive(root_path)
    return PrepareInputScanResult(
        input_dir=str(root_path),
        directories_scanned=directories_scanned,
        oir_files=sorted(oir_files),
        tiff_files=sorted(tiff_files),
        scan_seconds=time.perf_counter() - started,
    )


def detect_file_type(path: Path) -> str:
    """Return a short type label for profiling output."""
    suffix = path.suffix.lower()
    if suffix == ".oir":
        return "oir"
    if suffix in {".tif", ".tiff"}:
        return "tiff"
    return suffix.lstrip(".") or "unknown"


def build_investigation_notes(profile: PrepareInputProfile) -> list[str]:
    """Summarize likely causes for prepare_input overhead."""
    notes: list[str] = []
    scan = profile.scan

    if scan.oir_count == 0:
        notes.append(
            "No .oir files found — prepare_input should passthrough the source folder "
            "without reading, copying, or converting images."
        )
        if profile.projection_seconds > 0.1:
            notes.append(
                "Unexpected prepare_input duration on TIFF-only input — check folder "
                "scan cost or downstream logging overhead."
            )
        return notes

    notes.append(
        "OIR projection is active — any .oir file under the input folder triggers "
        "Z-max projection even when normal TIFF images are also present."
    )

    if scan.tiff_count > 0:
        notes.append(
            f"Folder contains {scan.tiff_count} top-level TIFF(s) and {scan.oir_count} "
            ".oir file(s); TIFFs are not copied by prepare_input, but OIR batch still runs."
        )

    if scan.directories_scanned > 5:
        notes.append(
            f"Recursive scan visited {scan.directories_scanned} directories — nested "
            "subfolders (including old outputs) increase discovery time and may pick "
            "up extra .oir files."
        )

    cache_hits = [
        record
        for record in profile.file_records
        if record.skipped and record.skip_reason == "projection_cache_hit"
    ]
    if cache_hits:
        notes.append(
            f"{len(cache_hits)} file(s) reused from oir_projection/ (mtime cache)."
        )

    duplicate_outputs = [
        record
        for record in profile.file_records
        if record.output_existed_before_run and not record.skipped
    ]
    if duplicate_outputs:
        notes.append(
            f"{len(duplicate_outputs)} file(s) already had projected TIFF outputs "
            "but were processed again (no reuse/skip logic)."
        )

    large_reads = [
        record
        for record in profile.file_records
        if record.input_bytes >= 10 * 1024 * 1024 and record.read_seconds >= 1.0
    ]
    if large_reads:
        notes.append(
            f"{len(large_reads)} large input file(s) (>10 MB) with slow read times — "
            "check whether full stacks are loaded into memory via aicsimageio/Bio-Formats."
        )

    slow_writes = [
        record
        for record in profile.file_records
        if record.write_seconds >= 1.0
    ]
    if slow_writes:
        notes.append(
            f"{len(slow_writes)} file(s) spent >=1s writing projected TIFFs "
            "(default tifffile.imwrite, no compression tuning)."
        )

    if profile.action == "oir_projection_cache_hit":
        notes.append(
            "All projected TIFFs were reused from the output folder — no Bio-Formats "
            "or aicsimageio projection ran."
        )

    if profile.engine == "fiji":
        notes.append(
            "Fiji engine runs one batch macro for all .oir files — per-file read/convert/"
            "write timings come from macro log output."
        )
    elif profile.engine == "python":
        notes.append(
            "Python engine reads each .oir with aicsimageio, projects in NumPy, and "
            "writes via tifffile.imwrite (default settings)."
        )

    multi_pass = [
        record
        for record in profile.file_records
        if not record.skipped and record.total_seconds >= 5.0
    ]
    if len(multi_pass) >= 2:
        notes.append(
            f"{len(multi_pass)} file(s) each took >=5s — total prepare_input time is "
            "dominated by per-file OIR projection, not CellProfiler."
        )

    return notes


def format_prepare_input_report(profile: PrepareInputProfile) -> str:
    """Format a human-readable prepare_input profiling report."""
    lines = [
        "Prepare input profiling report:",
        f"  input_dir: {profile.input_dir}",
        f"  action: {profile.action}",
        f"  scan_seconds: {profile.scan.scan_seconds:.2f}s",
        f"  directories_scanned: {profile.scan.directories_scanned}",
        f"  oir_files_found: {profile.scan.oir_count}",
        f"  tiff_files_found: {profile.scan.tiff_count}",
    ]
    if profile.engine:
        lines.append(f"  engine: {profile.engine}")
    if profile.projection_output_dir:
        lines.append(f"  projection_output_dir: {profile.projection_output_dir}")
    lines.append(f"  projection_seconds: {profile.projection_seconds:.2f}s")
    lines.append("")
    lines.append("Per-file timings:")
    if not profile.file_records:
        lines.append("  (no files processed)")
    for record in profile.file_records:
        lines.extend(
            [
                f"  - input: {record.input_path}",
                f"    detected_type: {record.detected_type}",
                f"    output: {record.output_path or '(none)'}",
                f"    input_bytes: {record.input_bytes}",
                f"    output_bytes: {record.output_bytes if record.output_bytes is not None else '(none)'}",
                f"    read_seconds: {record.read_seconds:.3f}",
                f"    conversion_seconds: {record.conversion_seconds:.3f}",
                f"    write_seconds: {record.write_seconds:.3f}",
                f"    total_seconds: {record.total_seconds:.3f}",
            ]
        )
        if record.skipped:
            lines.append(f"    skipped: yes ({record.skip_reason})")
        if record.output_existed_before_run:
            lines.append("    output_existed_before_run: yes")
        for note in record.notes:
            lines.append(f"    note: {note}")
        lines.append("")
    if profile.investigation_notes:
        lines.append("Investigation notes:")
        for note in profile.investigation_notes:
            lines.append(f"  - {note}")
    return "\n".join(lines).rstrip() + "\n"


def write_prepare_input_profile(
    logs_dir: Path,
    profile: PrepareInputProfile,
) -> tuple[Path, Path]:
    """Persist prepare_input profiling artifacts and log the report."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_path = logs_dir / "prepare_input_profile.json"
    text_path = logs_dir / "prepare_input_profile.txt"
    json_path.write_text(
        json.dumps(profile.to_dict(), indent=2),
        encoding="utf-8",
    )
    report = format_prepare_input_report(profile)
    text_path.write_text(report, encoding="utf-8")
    logger.info("\n%s", report.rstrip())
    return json_path, text_path


def parse_fiji_oir_file_records(stdout: str) -> list[PrepareInputFileRecord]:
    """Parse per-file timing lines emitted by the generated OIR Fiji macro."""
    records: list[PrepareInputFileRecord] = []
    current: PrepareInputFileRecord | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            current.total_seconds = (
                current.read_seconds
                + current.conversion_seconds
                + current.write_seconds
            )
            records.append(current)
            current = None

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("[OIR] input path:"):
            flush()
            input_path = line.split(":", 1)[1].strip()
            path = Path(input_path)
            current = PrepareInputFileRecord(
                input_path=str(path.resolve()) if path.exists() else input_path,
                detected_type="oir",
                input_bytes=_file_size(path) or 0,
            )
            continue
        if current is None:
            continue
        if line.startswith("[OIR] saveAs target:"):
            output_text = line.split(":", 1)[1].strip()
            current.output_path = output_text
            continue
        if line.startswith("[OIR] timing read_seconds:"):
            current.read_seconds = float(line.split(":", 1)[1].strip())
            continue
        if line.startswith("[OIR] timing conversion_seconds:"):
            current.conversion_seconds = float(line.split(":", 1)[1].strip())
            continue
        if line.startswith("[OIR] timing write_seconds:"):
            current.write_seconds = float(line.split(":", 1)[1].strip())
            continue
        if line.startswith("[OIR] saved file verified on disk:"):
            save_path = Path(line.split(":", 1)[1].strip())
            current.output_path = str(save_path)
            current.output_bytes = _file_size(save_path)

    flush()
    return records
