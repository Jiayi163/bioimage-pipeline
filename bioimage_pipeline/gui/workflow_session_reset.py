"""Helpers for clearing experiment-specific GUI state without touching disk files."""

from __future__ import annotations

from bioimage_pipeline.gui.run_settings import (
    CPPIPE_PATH_KEY,
    INPUT_DIR_KEY,
    OUTPUT_DIR_KEY,
)

EXPERIMENT_PATH_SETTINGS_KEYS = (
    CPPIPE_PATH_KEY,
    INPUT_DIR_KEY,
    OUTPUT_DIR_KEY,
)


def clear_experiment_paths_from_settings(settings: dict[str, str]) -> dict[str, str]:
    """Return persisted settings with experiment path keys removed."""
    return {
        key: value
        for key, value in settings.items()
        if key not in EXPERIMENT_PATH_SETTINGS_KEYS
    }


def workflow_has_experiment_fields(
    *,
    cppipe_path: str,
    input_dir: str,
    output_dir: str,
    pipeline_loaded: bool,
) -> bool:
    """Return whether any experiment-specific path fields are populated."""
    return bool(
        cppipe_path.strip()
        or input_dir.strip()
        or output_dir.strip()
        or pipeline_loaded
    )


def needs_reset_confirmation(
    *,
    cppipe_path: str,
    input_dir: str,
    output_dir: str,
    pipeline_loaded: bool,
    has_result_display: bool,
    run_in_progress: bool,
) -> bool:
    """Return whether the GUI should confirm before clearing session state."""
    if run_in_progress:
        return True
    if has_result_display:
        return True
    return workflow_has_experiment_fields(
        cppipe_path=cppipe_path,
        input_dir=input_dir,
        output_dir=output_dir,
        pipeline_loaded=pipeline_loaded,
    )


def workflow_run_actions_enabled(
    *,
    cppipe_path: str,
    input_dir: str,
    output_dir: str,
    running: bool,
) -> bool:
    """Return whether primary workflow run actions should be enabled."""
    if running:
        return False
    return bool(
        cppipe_path.strip()
        and input_dir.strip()
        and output_dir.strip()
    )
