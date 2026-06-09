"""Minimal Fiji/ImageJ macro runner."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

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


def _windows_fiji_candidates() -> list[Path]:
    candidates: list[Path] = []
    program_files = os.environ.get("ProgramFiles")
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    for base in (program_files, program_files_x86, Path.home()):
        if not base:
            continue
        fiji_dir = Path(base) / "Fiji.app"
        candidates.append(fiji_dir / "fiji-windows-x64.exe")
        candidates.append(fiji_dir / "ImageJ-win64.exe")
    return candidates


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
        path = Path(explicit)
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"Fiji executable not found: {path}")

    for env_name in ("FIJI_EXECUTABLE", "IMAGEJ_EXECUTABLE", "FIJI_PATH"):
        env_value = os.environ.get(env_name)
        if env_value:
            path = Path(env_value)
            if path.is_file():
                return path.resolve()

    for which_name in ("fiji-windows-x64.exe", "ImageJ-win64.exe", "ImageJ-linux64"):
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
        "Fiji/ImageJ executable not found. This workflow requires Fiji for .oir files.\n"
        "Pass --fiji D:\\path\\to\\Fiji\\fiji-windows-x64.exe or set FIJI_EXECUTABLE."
    )


def default_fiji_headless() -> bool:
    """Return the platform default for Fiji macro execution."""
    return platform.system() != "Windows"


def _subprocess_kwargs() -> dict:
    return {
        "capture_output": True,
        "text": True,
        "check": False,
        "env": os.environ.copy(),
    }


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
        raise FileNotFoundError(fiji_not_found_message())

    use_headless = default_fiji_headless() if headless is None else headless
    command = [str(executable)]
    if use_headless:
        command.append("--headless")
    command.extend(["-macro", str(macro), *macro_args])
    completed = subprocess.run(command, timeout=timeout, **_subprocess_kwargs())
    return FijiRunResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        macro_path=macro,
        executable=executable,
    )
