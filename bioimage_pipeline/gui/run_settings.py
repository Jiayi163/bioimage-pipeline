"""Persisted GUI run settings and cached executable discovery."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bioimage_pipeline.cellprofiler_runner import (
    cellprofiler_not_found_message,
    find_cellprofiler_gui_executable,
)
from bioimage_pipeline.fiji_runner import find_fiji_executable, fiji_not_found_message

DEFAULT_SETTINGS_DIR = Path.home() / ".bioimage-pipeline"
DEFAULT_SETTINGS_FILE = DEFAULT_SETTINGS_DIR / "gui_run_settings.json"

CELLPROFILER_SETTINGS_KEY = "cellprofiler_executable"
FIJI_SETTINGS_KEY = "fiji_executable"
OIR_PROJECTION_METHOD_KEY = "oir_projection_method"


@dataclass(frozen=True)
class ResolvedExecutable:
    """One resolved external-engine executable for the GUI run panel."""

    display_value: str
    resolved_path: Path | None
    source: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CachedRunExecutables:
    """Startup-cached executable resolution for the GUI run panel."""

    cellprofiler: ResolvedExecutable
    fiji: ResolvedExecutable


def default_gui_run_settings_path() -> Path:
    """Return the default on-disk GUI run-settings file path."""
    return DEFAULT_SETTINGS_FILE


def load_gui_run_settings(
    settings_path: str | Path | None = None,
) -> dict[str, str]:
    """Load persisted GUI run settings."""
    path = Path(settings_path) if settings_path is not None else DEFAULT_SETTINGS_FILE
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): str(value).strip()
        for key, value in payload.items()
        if value and str(value).strip()
    }


def save_gui_run_settings(
    values: dict[str, Any],
    *,
    settings_path: str | Path | None = None,
) -> Path:
    """Persist GUI run settings such as executable paths."""
    path = Path(settings_path) if settings_path is not None else DEFAULT_SETTINGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = {
        str(key): str(value).strip()
        for key, value in values.items()
        if value and str(value).strip()
    }
    path.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    return path.resolve()


def _resolve_saved_executable(
    saved_value: str | None,
    *,
    label: str,
) -> tuple[Path | None, list[str]]:
    if not saved_value or not saved_value.strip():
        return None, []
    resolved = _resolve_executable_candidate(saved_value.strip())
    if resolved is not None:
        return resolved, []
    return None, [
        f"Ignoring invalid saved {label} executable path: {saved_value.strip()}"
    ]


def _resolve_executable_candidate(value: str) -> Path | None:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    found = shutil.which(value)
    if found:
        return Path(found).resolve()
    return None


def _cellprofiler_env_path() -> Path | None:
    env_value = os.environ.get("CELLPROFILER_EXECUTABLE")
    if not env_value or not env_value.strip():
        return None
    return _resolve_executable_candidate(env_value.strip())


def _fiji_env_path() -> Path | None:
    for env_name in ("FIJI_EXECUTABLE", "IMAGEJ_EXECUTABLE", "FIJI_PATH"):
        env_value = os.environ.get(env_name)
        if not env_value or not env_value.strip():
            continue
        resolved = _resolve_executable_candidate(env_value.strip())
        if resolved is not None:
            return resolved
    return None


def _warn_when_saved_differs_from_preferred(
    warnings: list[str],
    *,
    saved_path: Path | None,
    preferred_path: Path,
    preferred_label: str,
    tool_label: str,
) -> None:
    if saved_path is not None and saved_path != preferred_path:
        warnings.append(
            f"Using {preferred_label} ({preferred_path}) instead of saved "
            f"{tool_label} path ({saved_path})"
        )


def resolve_cellprofiler_executable(
    saved_setting: str | None = None,
    *,
    explicit_override: str | None = None,
) -> ResolvedExecutable:
    """Resolve CellProfiler for the GUI run panel.

    Priority: explicit GUI input > environment variable > saved settings >
    auto-discovery.
    """
    warnings: list[str] = []
    saved_path, saved_warnings = _resolve_saved_executable(
        saved_setting,
        label="CellProfiler",
    )
    warnings.extend(saved_warnings)

    if explicit_override and explicit_override.strip():
        explicit_path = _resolve_executable_candidate(explicit_override.strip())
        if explicit_path is not None:
            return ResolvedExecutable(
                display_value=str(explicit_path),
                resolved_path=explicit_path,
                source="explicit",
                warnings=tuple(warnings),
            )
        warnings.append(
            f"Ignoring invalid explicit CellProfiler executable path: "
            f"{explicit_override.strip()}"
        )

    env_path = _cellprofiler_env_path()
    if env_path is not None:
        _warn_when_saved_differs_from_preferred(
            warnings,
            saved_path=saved_path,
            preferred_path=env_path,
            preferred_label="CELLPROFILER_EXECUTABLE",
            tool_label="CellProfiler",
        )
        return ResolvedExecutable(
            display_value=str(env_path),
            resolved_path=env_path,
            source="environment",
            warnings=tuple(warnings),
        )

    if saved_path is not None:
        return ResolvedExecutable(
            display_value=str(saved_path),
            resolved_path=saved_path,
            source="saved",
            warnings=tuple(warnings),
        )

    discovered = find_cellprofiler_gui_executable()
    if discovered is not None:
        return ResolvedExecutable(
            display_value=str(discovered),
            resolved_path=discovered,
            source="discovered",
            warnings=tuple(warnings),
        )

    warnings.append(cellprofiler_not_found_message())
    return ResolvedExecutable(
        display_value="cellprofiler",
        resolved_path=None,
        source="default",
        warnings=tuple(warnings),
    )


def resolve_fiji_executable(
    saved_setting: str | None = None,
    *,
    explicit_override: str | None = None,
) -> ResolvedExecutable:
    """Resolve Fiji for the GUI run panel.

    Priority: explicit GUI input > environment variables (FIJI_EXECUTABLE,
    IMAGEJ_EXECUTABLE, FIJI_PATH) > saved settings > auto-discovery.
    """
    warnings: list[str] = []
    saved_path, saved_warnings = _resolve_saved_executable(
        saved_setting,
        label="Fiji",
    )
    warnings.extend(saved_warnings)

    if explicit_override and explicit_override.strip():
        explicit_path = _resolve_executable_candidate(explicit_override.strip())
        if explicit_path is not None:
            return ResolvedExecutable(
                display_value=str(explicit_path),
                resolved_path=explicit_path,
                source="explicit",
                warnings=tuple(warnings),
            )
        warnings.append(
            f"Ignoring invalid explicit Fiji executable path: {explicit_override.strip()}"
        )

    env_path = _fiji_env_path()
    if env_path is not None:
        _warn_when_saved_differs_from_preferred(
            warnings,
            saved_path=saved_path,
            preferred_path=env_path,
            preferred_label="FIJI/IMAGEJ environment variable",
            tool_label="Fiji",
        )
        return ResolvedExecutable(
            display_value=str(env_path),
            resolved_path=env_path,
            source="environment",
            warnings=tuple(warnings),
        )

    if saved_path is not None:
        return ResolvedExecutable(
            display_value=str(saved_path),
            resolved_path=saved_path,
            source="saved",
            warnings=tuple(warnings),
        )

    discovered = find_fiji_executable()
    if discovered is not None:
        return ResolvedExecutable(
            display_value=str(discovered),
            resolved_path=discovered,
            source="discovered",
            warnings=tuple(warnings),
        )

    warnings.append(fiji_not_found_message())
    return ResolvedExecutable(
        display_value="",
        resolved_path=None,
        source="default",
        warnings=tuple(warnings),
    )


def _should_persist_discovered_executable(
    *,
    env_path: Path | None,
    saved_setting: str | None,
) -> bool:
    """Return True when a discovered path may be written to GUI settings."""
    if env_path is not None:
        return False
    saved_path, _ = _resolve_saved_executable(saved_setting, label="executable")
    return saved_path is None


def sync_discovered_executables_to_settings(
    cached: CachedRunExecutables,
    *,
    settings_path: str | Path | None = None,
) -> dict[str, str]:
    """Persist resolved executables without clobbering env or valid saved paths."""
    loaded = load_gui_run_settings(settings_path)
    updates: dict[str, str] = {}

    if cached.cellprofiler.resolved_path is not None:
        if cached.cellprofiler.source == "environment":
            updates[CELLPROFILER_SETTINGS_KEY] = cached.cellprofiler.display_value
        elif cached.cellprofiler.source == "discovered" and _should_persist_discovered_executable(
            env_path=_cellprofiler_env_path(),
            saved_setting=loaded.get(CELLPROFILER_SETTINGS_KEY),
        ):
            updates[CELLPROFILER_SETTINGS_KEY] = cached.cellprofiler.display_value

    if cached.fiji.resolved_path is not None:
        if cached.fiji.source == "environment":
            updates[FIJI_SETTINGS_KEY] = cached.fiji.display_value
        elif cached.fiji.source == "discovered" and _should_persist_discovered_executable(
            env_path=_fiji_env_path(),
            saved_setting=loaded.get(FIJI_SETTINGS_KEY),
        ):
            updates[FIJI_SETTINGS_KEY] = cached.fiji.display_value

    if not updates:
        return loaded

    merged = {**loaded, **updates}
    save_gui_run_settings(merged, settings_path=settings_path)
    return merged


def build_cached_run_executables(
    settings: dict[str, str] | None = None,
    *,
    settings_path: str | Path | None = None,
) -> CachedRunExecutables:
    """Resolve external executables once for GUI startup."""
    loaded = settings if settings is not None else load_gui_run_settings(settings_path)
    return CachedRunExecutables(
        cellprofiler=resolve_cellprofiler_executable(
            loaded.get(CELLPROFILER_SETTINGS_KEY),
        ),
        fiji=resolve_fiji_executable(loaded.get(FIJI_SETTINGS_KEY)),
    )


def collect_run_settings_from_values(
    *,
    cellprofiler_executable: str,
    fiji_executable: str,
    oir_projection_method: str | None = None,
) -> dict[str, str]:
    """Build the persisted settings payload from current GUI field values."""
    payload: dict[str, str] = {}
    cp_value = cellprofiler_executable.strip()
    fiji_value = fiji_executable.strip()
    if cp_value:
        payload[CELLPROFILER_SETTINGS_KEY] = cp_value
    if fiji_value:
        payload[FIJI_SETTINGS_KEY] = fiji_value
    if oir_projection_method:
        payload[OIR_PROJECTION_METHOD_KEY] = oir_projection_method.strip()
    return payload
