"""Persisted GUI run settings and cached executable discovery."""

from __future__ import annotations

import json
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
CPPIPE_PATH_KEY = "cppipe_path"
INPUT_DIR_KEY = "input_dir"
OUTPUT_DIR_KEY = "output_dir"
FIJI_MACRO_PATH_KEY = "fiji_macro_path"
OIR_PROJECTION_ENGINE_KEY = "oir_projection_engine"
EXPORT_FIJI_TIFFS_KEY = "export_fiji_tiffs"
GENERATE_QC_KEY = "generate_qc"
FORCE_OIR_REPROJECT_KEY = "force_oir_reproject"


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


def resolve_cellprofiler_executable(
    saved_setting: str | None = None,
) -> ResolvedExecutable:
    """Resolve CellProfiler using saved setting, env, PATH, then common paths."""
    warnings: list[str] = []
    saved_path, saved_warnings = _resolve_saved_executable(
        saved_setting,
        label="CellProfiler",
    )
    warnings.extend(saved_warnings)
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


def resolve_fiji_executable(saved_setting: str | None = None) -> ResolvedExecutable:
    """Resolve Fiji using saved setting, then standard Fiji discovery."""
    warnings: list[str] = []
    saved_path, saved_warnings = _resolve_saved_executable(
        saved_setting,
        label="Fiji",
    )
    warnings.extend(saved_warnings)
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


def sync_discovered_executables_to_settings(
    cached: CachedRunExecutables,
    *,
    settings_path: str | Path | None = None,
) -> dict[str, str]:
    """Persist auto-discovered executables when no valid saved path exists."""
    loaded = load_gui_run_settings(settings_path)
    updates: dict[str, str] = {}

    if cached.cellprofiler.source == "discovered":
        updates[CELLPROFILER_SETTINGS_KEY] = cached.cellprofiler.display_value
    if cached.fiji.source == "discovered":
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
    cppipe_path: str | None = None,
    input_dir: str | None = None,
    output_dir: str | None = None,
    fiji_macro_path: str | None = None,
    oir_projection_engine: str | None = None,
    export_fiji_tiffs: bool | None = None,
    generate_qc: bool | None = None,
    force_oir_reproject: bool | None = None,
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
    if cppipe_path and cppipe_path.strip():
        payload[CPPIPE_PATH_KEY] = cppipe_path.strip()
    if input_dir and input_dir.strip():
        payload[INPUT_DIR_KEY] = input_dir.strip()
    if output_dir and output_dir.strip():
        payload[OUTPUT_DIR_KEY] = output_dir.strip()
    if fiji_macro_path and fiji_macro_path.strip():
        payload[FIJI_MACRO_PATH_KEY] = fiji_macro_path.strip()
    if oir_projection_engine and oir_projection_engine.strip():
        payload[OIR_PROJECTION_ENGINE_KEY] = oir_projection_engine.strip()
    if export_fiji_tiffs is not None:
        payload[EXPORT_FIJI_TIFFS_KEY] = "true" if export_fiji_tiffs else "false"
    if generate_qc is not None:
        payload[GENERATE_QC_KEY] = "true" if generate_qc else "false"
    if force_oir_reproject is not None:
        payload[FORCE_OIR_REPROJECT_KEY] = "true" if force_oir_reproject else "false"
    return payload


def parse_bool_setting(value: str | None, *, default: bool) -> bool:
    """Parse persisted boolean GUI settings."""
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def extract_recent_workflow_paths(settings: dict[str, str]) -> dict[str, str]:
    """Return saved experiment paths for explicit 'Load recent paths' actions."""
    return {
        INPUT_DIR_KEY: settings.get(INPUT_DIR_KEY, "").strip(),
        OUTPUT_DIR_KEY: settings.get(OUTPUT_DIR_KEY, "").strip(),
        CPPIPE_PATH_KEY: settings.get(CPPIPE_PATH_KEY, "").strip(),
    }
