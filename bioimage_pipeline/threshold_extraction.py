"""Read-only extraction of threshold-related settings from imported pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bioimage_pipeline.cppipe_io import (
    CppipeModule,
    CppipePipeline,
    load_and_validate_imported_pipeline,
    load_cppipe,
)

# IdentifyPrimaryObjects settings relevant to thresholding, spot size, and declumping.
IDENTIFY_PRIMARY_OBJECTS_THRESHOLD_SETTING_KEYS: frozenset[str] = frozenset(
    {
        "Threshold strategy",
        "Thresholding method",
        "Threshold smoothing scale",
        "Threshold correction factor",
        "Lower and upper bounds on threshold",
        "Manual threshold",
        "Select the measurement to threshold with",
        "Two-class or three-class thresholding?",
        "Assign pixels in the middle intensity class to the foreground or the background?",
        "Size of adaptive window",
        "Lower outlier fraction",
        "Upper outlier fraction",
        "Averaging method",
        "Variance method",
        "# of deviations",
        "Typical diameter of objects, in pixel units (Min,Max)",
        "Discard objects outside the diameter range?",
        "Discard objects touching the border of the image?",
        "Method to distinguish clumped objects",
        "Method to draw dividing lines between clumped objects",
        "Size of smoothing filter",
        "Suppress local maxima that are closer than this minimum allowed distance",
        "Automatically calculate size of smoothing filter for declumping?",
        "Automatically calculate minimum allowed distance between local maxima?",
        "Handling of objects if excessive number of objects identified",
        "Maximum number of objects",
    }
)

@dataclass(frozen=True)
class IdentifyPrimaryObjectsThresholdProfile:
    """Threshold-related settings read from one IdentifyPrimaryObjects module."""

    module_index: int
    module_num: int
    input_image: str | None
    object_name: str | None
    threshold_strategy: str | None
    thresholding_method: str | None
    threshold_smoothing_scale: str | None
    threshold_correction_factor: str | None
    threshold_bounds: str | None
    typical_diameter: str | None
    discard_outside_diameter: str | None
    declumping_method: str | None
    dividing_lines_method: str | None
    declumping_smoothing_size: str | None
    local_maxima_min_distance: str | None
    threshold_settings: dict[str, str] = field(default_factory=dict)
    all_module_settings: dict[str, str] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        """CellProfiler-style module label."""
        label = self.object_name or "IdentifyPrimaryObjects"
        return f"{self.module_num}. IdentifyPrimaryObjects ({label})"


def _module_settings_dict(module: CppipeModule) -> dict[str, str]:
    return {setting.key: setting.value for setting in module.settings}


def _pick(settings: dict[str, str], key: str) -> str | None:
    value = settings.get(key)
    if value is None or value == "":
        return None
    return value


def _threshold_settings_subset(settings: dict[str, str]) -> dict[str, str]:
    return {
        key: settings[key]
        for key in IDENTIFY_PRIMARY_OBJECTS_THRESHOLD_SETTING_KEYS
        if key in settings
    }


def extract_identify_primary_objects_threshold_profiles(
    pipeline: CppipePipeline,
) -> list[IdentifyPrimaryObjectsThresholdProfile]:
    """Return threshold profiles for every IdentifyPrimaryObjects module.

    This function is read-only: it does not modify ``pipeline`` or any file on
    disk.
    """
    profiles: list[IdentifyPrimaryObjectsThresholdProfile] = []
    for module_index, module in enumerate(pipeline.modules):
        if module.name != "IdentifyPrimaryObjects":
            continue
        settings = _module_settings_dict(module)
        threshold_settings = _threshold_settings_subset(settings)
        profiles.append(
            IdentifyPrimaryObjectsThresholdProfile(
                module_index=module_index,
                module_num=module.module_num,
                input_image=_pick(settings, "Select the input image"),
                object_name=_pick(settings, "Name the primary objects to be identified"),
                threshold_strategy=_pick(settings, "Threshold strategy"),
                thresholding_method=_pick(settings, "Thresholding method"),
                threshold_smoothing_scale=_pick(settings, "Threshold smoothing scale"),
                threshold_correction_factor=_pick(
                    settings, "Threshold correction factor"
                ),
                threshold_bounds=_pick(settings, "Lower and upper bounds on threshold"),
                typical_diameter=_pick(
                    settings,
                    "Typical diameter of objects, in pixel units (Min,Max)",
                ),
                discard_outside_diameter=_pick(
                    settings, "Discard objects outside the diameter range?"
                ),
                declumping_method=_pick(
                    settings, "Method to distinguish clumped objects"
                ),
                dividing_lines_method=_pick(
                    settings, "Method to draw dividing lines between clumped objects"
                ),
                declumping_smoothing_size=_pick(settings, "Size of smoothing filter"),
                local_maxima_min_distance=_pick(
                    settings,
                    "Suppress local maxima that are closer than this minimum allowed distance",
                ),
                threshold_settings=threshold_settings,
                all_module_settings=dict(settings),
            )
        )
    return profiles


def load_identify_primary_objects_threshold_profiles(
    cppipe_path: str | Path,
    *,
    validate: bool = True,
) -> list[IdentifyPrimaryObjectsThresholdProfile]:
    """Load a ``.cppipe`` file and extract IdentifyPrimaryObjects threshold profiles."""
    if validate:
        pipeline = load_and_validate_imported_pipeline(cppipe_path)
    else:
        pipeline = load_cppipe(cppipe_path)
    return extract_identify_primary_objects_threshold_profiles(pipeline)
