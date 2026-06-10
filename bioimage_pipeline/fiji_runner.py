"""Fiji/ImageJ macro runners for batch export workflows."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

FijiExportEngine = Literal["fiji"]
FijiExportMode = Literal["batch"]
DEFAULT_FIJI_EXPORT_MACRO = (
    Path(__file__).resolve().parents[1] / "examples" / "fiji_macros" / "export_folder.ijm"
)

_FIJI_ERROR_MARKERS = (
    "VerifyError",
    "File not found:",
    "Import failed:",
    "Open failed:",
    "Unsupported format",
    "Exception in",
    "java.lang.",
)


@dataclass
class FijiRunResult:
    """Result of a headless Fiji macro invocation."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    macro_path: Path | None
    executable: Path

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}\n{self.stderr}".strip()

    @property
    def error_lines(self) -> list[str]:
        return extract_fiji_errors(self.stdout, self.stderr)

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.error_lines


@dataclass
class FijiExportResult:
    """Result of one Fiji batch export invocation."""

    input_dir: Path
    masks_dir: Path
    labels_dir: Path
    macro_path: Path
    executable: Path
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    log_files: dict[str, Path]
    mask_exports: list[Path]
    label_exports: list[Path]
    export_engine: FijiExportEngine = "fiji"
    export_mode: FijiExportMode = "batch"

    @property
    def combined_output(self) -> str:
        return f"{self.stdout}\n{self.stderr}".strip()

    @property
    def error_lines(self) -> list[str]:
        return extract_fiji_errors(self.stdout, self.stderr)

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.error_lines

    def to_dict(self) -> dict[str, object]:
        """Serialize the export result for workflow summaries."""
        return {
            "export_engine": self.export_engine,
            "export_mode": self.export_mode,
            "input_dir": str(self.input_dir),
            "masks_dir": str(self.masks_dir),
            "labels_dir": str(self.labels_dir),
            "macro_path": str(self.macro_path),
            "executable": str(self.executable),
            "command": list(self.command),
            "returncode": self.returncode,
            "mask_exports": [str(path) for path in self.mask_exports],
            "label_exports": [str(path) for path in self.label_exports],
            "log_files": {key: str(path) for key, path in self.log_files.items()},
            "errors": self.error_lines,
        }


def extract_fiji_errors(stdout: str, stderr: str) -> list[str]:
    """Return notable Fiji/ImageJ error lines from process output."""
    errors: list[str] = []
    for line in f"{stdout}\n{stderr}".splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(marker in stripped for marker in _FIJI_ERROR_MARKERS):
            errors.append(stripped)
    return errors


def format_fiji_error_summary(
    result: FijiRunResult,
    *,
    max_lines: int = 8,
) -> str:
    """Build a short user-facing summary of process output errors."""
    lines = result.error_lines[-max_lines:]
    if lines:
        return "\n".join(lines)
    output = result.combined_output
    if output:
        tail = [line.strip() for line in output.splitlines() if line.strip()][-max_lines:]
        return "\n".join(tail)
    return "Fiji produced no captured console output."


def suggest_imagej1_executable(executable: Path) -> Path | None:
    """Suggest the ImageJ1 launcher when a Fiji 2 wrapper executable was used."""
    if executable.name.lower().startswith("fiji-windows"):
        sibling = executable.parent / "ImageJ-win64.exe"
        if sibling.is_file():
            return sibling.resolve()
    return None


def _fiji_launchers_in_dir(directory: Path) -> list[Path]:
    """Return known Fiji/ImageJ launcher executables inside *directory*."""
    if not directory.is_dir():
        return []
    launchers: list[Path] = []
    for name in (
        "fiji-windows-x64.exe",
        "fiji-windows.exe",
        "ImageJ-win64.exe",
        "ImageJ.exe",
    ):
        path = directory / name
        if path.is_file():
            launchers.append(path.resolve())
    return launchers


def _dedupe_paths(candidates: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _windows_fiji_candidates() -> list[Path]:
    candidates: list[Path] = []
    program_files = os.environ.get("ProgramFiles")
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    local_app_data = os.environ.get("LocalAppData")
    home = Path.home()

    fixed_dirs = [
        home / "Fiji.app",
        home / "Fiji",
        home / "Desktop" / "Fiji.app",
        home / "Desktop" / "Fiji",
        home / "Downloads" / "Fiji.app",
        home / "Downloads" / "Fiji",
    ]
    if local_app_data:
        fixed_dirs.extend(
            [
                Path(local_app_data) / "Programs" / "Fiji.app",
                Path(local_app_data) / "Programs" / "Fiji",
            ]
        )
    for base in (program_files, program_files_x86):
        if base:
            fixed_dirs.extend(
                [
                    Path(base) / "Fiji.app",
                    Path(base) / "Fiji",
                ]
            )

    for directory in fixed_dirs:
        candidates.extend(_fiji_launchers_in_dir(directory))

    for program_root in (program_files, program_files_x86):
        if not program_root:
            continue
        root_path = Path(program_root)
        if not root_path.is_dir():
            continue
        try:
            for folder in root_path.glob("Fiji*"):
                if folder.is_dir():
                    candidates.extend(_fiji_launchers_in_dir(folder))
        except OSError:
            continue

    return _dedupe_paths(candidates)


def _unix_fiji_candidates() -> list[Path]:
    home = Path.home()
    return [
        Path("/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"),
        home / "Fiji.app" / "ImageJ-linux64",
        Path("/opt/Fiji.app") / "ImageJ-linux64",
    ]


def find_fiji_executable(explicit: str | Path | None = None) -> Path | None:
    """Resolve a Fiji/ImageJ executable path."""
    if explicit is not None:
        value = str(explicit).strip()
        if not value:
            explicit = None
        else:
            path = Path(value)
            if path.is_file():
                return path.resolve()
            found = shutil.which(value)
            if found:
                return Path(found).resolve()
            return None

    for env_name in ("FIJI_EXECUTABLE", "IMAGEJ_EXECUTABLE", "FIJI_PATH"):
        env_value = os.environ.get(env_name)
        if env_value:
            path = Path(env_value)
            if path.is_file():
                return path.resolve()

    for which_name in (
        "fiji-windows-x64.exe",
        "fiji-windows.exe",
        "ImageJ-win64.exe",
        "ImageJ-linux64",
        "fiji",
    ):
        found = shutil.which(which_name)
        if found:
            return Path(found).resolve()

    candidates = (
        _windows_fiji_candidates()
        if platform.system() == "Windows"
        else _unix_fiji_candidates()
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def fiji_not_found_message() -> str:
    """Return guidance when Fiji/ImageJ cannot be resolved."""
    return (
        "Fiji/ImageJ executable not found. Install Fiji/ImageJ, pass a Fiji "
        "executable path, or set FIJI_EXECUTABLE."
    )


def default_fiji_headless() -> bool:
    """Return the platform default for Fiji macro execution."""
    return True


def _subprocess_kwargs() -> dict:
    return {
        "capture_output": True,
        "text": True,
        "check": False,
        "env": os.environ.copy(),
    }


def _format_macro_argument(macro_args: Sequence[str]) -> list[str]:
    """Format ImageJ macro args as one portable argument string."""
    if not macro_args:
        return []
    return ["|".join(str(arg) for arg in macro_args)]


def _write_fiji_logs(
    log_dir: Path,
    *,
    command: Sequence[str],
    stdout: str,
    stderr: str,
    returncode: int,
) -> dict[str, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_files = {
        "stdout": log_dir / "fiji_stdout.log",
        "stderr": log_dir / "fiji_stderr.log",
        "command": log_dir / "fiji_command.txt",
        "exit_code": log_dir / "fiji_exit_code.txt",
    }
    log_files["stdout"].write_text(stdout, encoding="utf-8")
    log_files["stderr"].write_text(stderr, encoding="utf-8")
    log_files["command"].write_text(" ".join(command), encoding="utf-8")
    log_files["exit_code"].write_text(str(returncode), encoding="utf-8")
    return log_files


def run_fiji_macro(
    macro_path: str | Path,
    *macro_args: str,
    fiji_executable: str | Path | None = None,
    headless: bool | None = None,
    timeout: float | None = None,
) -> FijiRunResult:
    """Run a Fiji macro with positional arguments."""
    macro = Path(macro_path).resolve()
    if not macro.is_file():
        raise FileNotFoundError(f"Fiji macro not found: {macro}")

    executable = find_fiji_executable(fiji_executable)
    if executable is None:
        if fiji_executable is not None and str(fiji_executable).strip():
            raise FileNotFoundError(f"Fiji executable not found: {fiji_executable}")
        raise FileNotFoundError(fiji_not_found_message())

    use_headless = default_fiji_headless() if headless is None else headless
    command = [str(executable)]
    if use_headless:
        command.append("--headless")
    command.extend(["-macro", str(macro), *_format_macro_argument(macro_args)])
    completed = subprocess.run(command, timeout=timeout, **_subprocess_kwargs())
    return FijiRunResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        macro_path=macro,
        executable=executable,
    )


def run_fiji_batch_export(
    input_dir: str | Path,
    masks_dir: str | Path,
    labels_dir: str | Path,
    *,
    macro_path: str | Path | None = None,
    fiji_executable: str | Path | None = None,
    headless: bool | None = None,
    timeout: float | None = None,
    image_pattern: str = "*.tif",
    log_dir: str | Path | None = None,
) -> FijiExportResult:
    """Run one Fiji/ImageJ batch macro to export final mask and label TIFFs.

    The macro receives the raw CellProfiler output folder plus destination
    ``masks`` and ``labels`` folders. It is expected to loop over the folder in
    Fiji, so this function performs exactly one Fiji subprocess invocation.
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Fiji export input directory not found: {input_path}")

    masks_path = Path(masks_dir)
    labels_path = Path(labels_dir)
    masks_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)

    macro = Path(macro_path) if macro_path is not None else DEFAULT_FIJI_EXPORT_MACRO
    if not macro.is_file():
        raise FileNotFoundError(f"Fiji batch export macro not found: {macro}")

    run_result = run_fiji_macro(
        macro,
        str(input_path.resolve()),
        str(masks_path.resolve()),
        str(labels_path.resolve()),
        image_pattern,
        fiji_executable=fiji_executable,
        headless=headless,
        timeout=timeout,
    )
    log_files: dict[str, Path] = {}
    if log_dir is not None:
        log_files = _write_fiji_logs(
            Path(log_dir),
            command=run_result.command,
            stdout=run_result.stdout,
            stderr=run_result.stderr,
            returncode=run_result.returncode,
        )

    return FijiExportResult(
        input_dir=input_path.resolve(),
        masks_dir=masks_path.resolve(),
        labels_dir=labels_path.resolve(),
        macro_path=macro.resolve(),
        executable=run_result.executable,
        command=run_result.command,
        returncode=run_result.returncode,
        stdout=run_result.stdout,
        stderr=run_result.stderr,
        log_files=log_files,
        mask_exports=sorted(masks_path.glob("*.tif")),
        label_exports=sorted(labels_path.glob("*.tif")),
    )
