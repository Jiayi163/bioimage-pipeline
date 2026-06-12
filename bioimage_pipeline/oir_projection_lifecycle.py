"""Lifecycle snapshots and audit logging for ``oir_projection/`` cache debugging."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OIR_PROJECTION_DIR_NAME = "oir_projection"
LIFECYCLE_JSON = "oir_projection_lifecycle.json"
AUDIT_LOG = "oir_projection_audit.log"


def is_oir_projection_path(path: str | Path) -> bool:
    """Return whether *path* is a file directly under an ``oir_projection/`` folder."""
    resolved = Path(path).resolve()
    return resolved.parent.name == OIR_PROJECTION_DIR_NAME


def list_projection_tiff_entries(projection_output_dir: Path) -> list[dict[str, Any]]:
    """Return metadata for TIFF files present in the projection output folder."""
    resolved_dir = projection_output_dir.expanduser().resolve()
    if not resolved_dir.is_dir():
        return []

    entries: list[dict[str, Any]] = []
    for path in sorted(resolved_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        stat = path.stat()
        entries.append(
            {
                "path": str(path.resolve()),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return entries


def snapshot_oir_projection_dir(projection_dir: Path) -> dict[str, Any]:
    """Capture directory existence and TIFF listing for one projection folder."""
    resolved_dir = projection_dir.expanduser().resolve()
    return {
        "projection_output_dir": str(resolved_dir),
        "projection_output_dir_exists": resolved_dir.is_dir(),
        "existing_tifs": list_projection_tiff_entries(resolved_dir),
    }


def _tiff_paths(snapshot: dict[str, Any]) -> set[str]:
    return {entry["path"] for entry in snapshot.get("existing_tifs") or []}


def compare_cross_run_snapshots(
    previous_end: dict[str, Any] | None,
    current_start: dict[str, Any],
) -> list[str]:
    """Compare previous run ``workflow_end`` to this run ``workflow_start``."""
    if previous_end is None:
        return []

    previous_paths = _tiff_paths(previous_end)
    current_paths = _tiff_paths(current_start)
    if not previous_paths:
        return []

    warnings: list[str] = []
    missing = sorted(previous_paths - current_paths)
    if missing:
        warnings.append(
            f"{len(missing)} TIFF(s) present at previous workflow_end but missing at "
            f"workflow_start — not removed by pipeline code."
        )
        for path in missing:
            warnings.append(f"  missing: {path}")

    previous_dir = previous_end.get("projection_output_dir")
    current_dir = current_start.get("projection_output_dir")
    if previous_dir and current_dir and previous_dir != current_dir:
        warnings.append(
            "projection_output_dir changed between runs: "
            f"previous={previous_dir} current={current_dir}"
        )
    return warnings


def load_previous_workflow_end(lifecycle_path: Path) -> dict[str, Any] | None:
    """Return the ``workflow_end`` snapshot from an on-disk lifecycle file."""
    if not lifecycle_path.is_file():
        return None
    try:
        payload = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    stages = payload.get("stages") or []
    for stage in reversed(stages):
        if stage.get("stage") == "workflow_end":
            return stage.get("snapshot")
    return None


def log_oir_projection_audit(
    logs_dir: str | Path | None,
    event: str,
    details: dict[str, Any],
) -> None:
    """Append one audit event to ``logs/oir_projection_audit.log``."""
    if logs_dir is None:
        return
    log_path = Path(logs_dir) / AUDIT_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"[{timestamp}] {event} {json.dumps(details, sort_keys=True)}\n"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


@dataclass
class OirProjectionLifecycleRecorder:
    """Accumulates lifecycle stage snapshots for one workflow run."""

    logs_dir: Path
    results_dir: Path
    projection_dir: Path
    stages: list[dict[str, Any]] = field(default_factory=list)
    cross_run_warnings: list[str] = field(default_factory=list)

    @property
    def lifecycle_path(self) -> Path:
        return self.logs_dir / LIFECYCLE_JSON

    def record_workflow_start(self) -> dict[str, Any]:
        """Record workflow_start and compare against the previous run if present."""
        previous_end = load_previous_workflow_end(self.lifecycle_path)
        snapshot = snapshot_oir_projection_dir(self.projection_dir)
        self.cross_run_warnings = compare_cross_run_snapshots(previous_end, snapshot)
        return self.record_stage(
            "workflow_start",
            snapshot=snapshot,
            extra={"cross_run_warnings": list(self.cross_run_warnings)},
        )

    def record_stage(
        self,
        stage: str,
        *,
        snapshot: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one lifecycle stage and persist ``oir_projection_lifecycle.json``."""
        if snapshot is None:
            snapshot = snapshot_oir_projection_dir(self.projection_dir)
        record: dict[str, Any] = {
            "stage": stage,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "results_dir": str(self.results_dir.resolve()),
            "snapshot": snapshot,
        }
        if extra:
            record["extra"] = extra
        self.stages.append(record)
        self._write()
        return record

    def _write(self) -> None:
        payload = {
            "results_dir": str(self.results_dir.resolve()),
            "projection_output_dir": str(self.projection_dir.resolve()),
            "cross_run_warnings": list(self.cross_run_warnings),
            "stages": self.stages,
        }
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.lifecycle_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
