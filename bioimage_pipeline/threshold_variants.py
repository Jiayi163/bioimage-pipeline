"""Generate temporary ``.cppipe`` variants with modified IPO threshold settings."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from bioimage_pipeline.cppipe_io import (
    CppipePipeline,
    load_cppipe,
    save_cppipe,
    update_module_setting,
)
from bioimage_pipeline.threshold_extraction import (
    IDENTIFY_PRIMARY_OBJECTS_THRESHOLD_SETTING_KEYS,
    IdentifyPrimaryObjectsThresholdProfile,
)

_THRESHOLD_SETTING_APPLY_KEYS: tuple[tuple[str, str], ...] = (
    ("threshold_strategy", "Threshold strategy"),
    ("thresholding_method", "Thresholding method"),
    ("threshold_correction_factor", "Threshold correction factor"),
    ("threshold_smoothing_scale", "Threshold smoothing scale"),
    ("threshold_bounds", "Lower and upper bounds on threshold"),
    ("adaptive_window_size", "Size of adaptive window"),
)

OPTIMISTIC_CORRECTION_FACTOR = "0.9"
_DIAMETER_TO_SMOOTHING_DIVISOR = 6.25
_MIN_SMOOTHING_SCALE = 0.5
_MIN_ADAPTIVE_WINDOW = 7
_MAX_ADAPTIVE_WINDOW = 101

ROBUST_BACKGROUND_METHOD = "Robust Background"
MINIMUM_CROSS_ENTROPY_METHOD = "Minimum Cross-Entropy"
OTSU_METHOD = "Otsu"
GLOBAL_STRATEGY = "Global"
ADAPTIVE_STRATEGY = "Adaptive"


class AmbiguousIdentifyPrimaryObjectsError(ValueError):
    """Raised when multiple IPO modules match and none was selected explicitly."""


@dataclass(frozen=True)
class ThresholdVariantSpec:
    """Description of one candidate threshold configuration for an IPO module."""

    variant_id: str
    display_name: str
    target_module_index: int
    target_module_num: int | None = None
    thresholding_method: str | None = None
    threshold_strategy: str | None = None
    threshold_correction_factor: str | None = None
    threshold_smoothing_scale: str | None = None
    threshold_bounds: str | None = None
    adaptive_window_size: str | None = None
    notes: str | None = None
    is_baseline: bool = False


@dataclass(frozen=True)
class ThresholdVariantArtifact:
    """One generated variant pipeline written under the output directory."""

    spec: ThresholdVariantSpec
    variant_dir: Path
    pipeline_path: Path


def select_ipo_threshold_profile(
    profiles: list[IdentifyPrimaryObjectsThresholdProfile],
    *,
    module_index: int | None = None,
    module_num: int | None = None,
    object_name: str | None = None,
) -> IdentifyPrimaryObjectsThresholdProfile:
    """Select one IPO profile when a pipeline contains multiple IPO modules."""
    if not profiles:
        raise ValueError("No IdentifyPrimaryObjects modules found in the pipeline.")

    selectors = [
        selector
        for selector, value in (
            ("module_index", module_index),
            ("module_num", module_num),
            ("object_name", object_name),
        )
        if value is not None
    ]
    if len(selectors) > 1:
        raise ValueError(
            "Specify only one IPO selector: module_index, module_num, or object_name."
        )

    if module_index is not None:
        matches = [profile for profile in profiles if profile.module_index == module_index]
        if not matches:
            raise ValueError(
                f"No IdentifyPrimaryObjects module found at index {module_index}."
            )
        return matches[0]

    if module_num is not None:
        matches = [profile for profile in profiles if profile.module_num == module_num]
        if not matches:
            raise ValueError(
                f"No IdentifyPrimaryObjects module found with module_num {module_num}."
            )
        if len(matches) > 1:
            raise AmbiguousIdentifyPrimaryObjectsError(
                f"Multiple IdentifyPrimaryObjects modules share module_num {module_num}."
            )
        return matches[0]

    if object_name is not None:
        matches = [
            profile for profile in profiles if profile.object_name == object_name
        ]
        if not matches:
            raise ValueError(
                f"No IdentifyPrimaryObjects module found for object name {object_name!r}."
            )
        if len(matches) > 1:
            raise AmbiguousIdentifyPrimaryObjectsError(
                f"Multiple IdentifyPrimaryObjects modules use object name {object_name!r}."
            )
        return matches[0]

    if len(profiles) == 1:
        return profiles[0]

    object_names = ", ".join(
        profile.object_name or f"index {profile.module_index}" for profile in profiles
    )
    raise AmbiguousIdentifyPrimaryObjectsError(
        "Pipeline contains multiple IdentifyPrimaryObjects modules "
        f"({object_names}). Select one with module_index, module_num, or object_name."
    )


def parse_typical_diameter_pixels(
    typical_diameter: str | None,
) -> tuple[float | None, float | None]:
    """Parse IPO ``Typical diameter ... (Min,Max)`` into pixel bounds."""
    if typical_diameter is None:
        return None, None
    parts = [part.strip() for part in typical_diameter.split(",")]
    if len(parts) != 2:
        return None, None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None, None


def _odd_clamped(value: int, minimum: int, maximum: int) -> int:
    clamped = max(minimum, min(maximum, value))
    if clamped % 2 == 0:
        clamped += 1 if clamped < maximum else -1
    return max(minimum, min(maximum, clamped))


def estimate_threshold_smoothing_scale_from_diameter(
    typical_diameter: str | None,
) -> str | None:
    """Estimate IPO threshold smoothing scale from expected object diameter."""
    min_diameter, max_diameter = parse_typical_diameter_pixels(typical_diameter)
    if min_diameter is None or max_diameter is None:
        return None
    reference_diameter = (min_diameter + max_diameter) / 2.0
    smoothing = max(_MIN_SMOOTHING_SCALE, reference_diameter / _DIAMETER_TO_SMOOTHING_DIVISOR)
    return f"{smoothing:.4f}".rstrip("0").rstrip(".")


def estimate_adaptive_window_size_from_diameter(
    typical_diameter: str | None,
) -> str | None:
    """Estimate IPO adaptive window size from expected object diameter."""
    _, max_diameter = parse_typical_diameter_pixels(typical_diameter)
    if max_diameter is None:
        return None
    window = _odd_clamped(
        int(round(max_diameter * 3.0)),
        _MIN_ADAPTIVE_WINDOW,
        _MAX_ADAPTIVE_WINDOW,
    )
    return str(window)


def profile_has_adaptive_window_setting(
    profile: IdentifyPrimaryObjectsThresholdProfile,
) -> bool:
    """Return whether the imported IPO module exposes adaptive window size."""
    return "Size of adaptive window" in profile.all_module_settings


def generate_optimistic_threshold_variant_spec(
    profile: IdentifyPrimaryObjectsThresholdProfile,
) -> ThresholdVariantSpec:
    """Build one fast optimistic Otsu adaptive candidate for subset trial."""
    smoothing_scale = estimate_threshold_smoothing_scale_from_diameter(
        profile.typical_diameter
    )
    adaptive_window_size = None
    if profile_has_adaptive_window_setting(profile):
        adaptive_window_size = estimate_adaptive_window_size_from_diameter(
            profile.typical_diameter
        )

    notes = [
        "Fast optimistic candidate: Otsu adaptive with correction factor "
        f"{OPTIMISTIC_CORRECTION_FACTOR}.",
    ]
    if smoothing_scale is not None:
        notes.append(
            f"Threshold smoothing scale estimated from typical diameter "
            f"{profile.typical_diameter!r} -> {smoothing_scale}."
        )
    if adaptive_window_size is not None:
        notes.append(
            f"Adaptive window size estimated from typical diameter "
            f"{profile.typical_diameter!r} -> {adaptive_window_size}."
        )

    return ThresholdVariantSpec(
        variant_id="001_optimistic_otsu_adaptive",
        display_name="Optimistic Otsu Adaptive",
        target_module_index=profile.module_index,
        target_module_num=profile.module_num,
        thresholding_method=OTSU_METHOD,
        threshold_strategy=ADAPTIVE_STRATEGY,
        threshold_correction_factor=OPTIMISTIC_CORRECTION_FACTOR,
        threshold_smoothing_scale=smoothing_scale,
        adaptive_window_size=adaptive_window_size,
        notes=" ".join(notes),
    )


def profile_supports_robust_background(
    profile: IdentifyPrimaryObjectsThresholdProfile,
) -> bool:
    """Return whether Robust Background is supported for this imported IPO module."""
    if profile.thresholding_method == ROBUST_BACKGROUND_METHOD:
        return True

    version_text = profile.all_module_settings.get("Threshold setting version")
    if version_text is None:
        return False
    try:
        return int(version_text) >= 11
    except ValueError:
        return False


def generate_basic_threshold_variant_specs(
    profile: IdentifyPrimaryObjectsThresholdProfile,
) -> list[ThresholdVariantSpec]:
    """Build a small default set of threshold candidate specs for one IPO module."""
    specs: list[ThresholdVariantSpec] = []
    next_id = 1

    def append_spec(
        slug: str,
        display_name: str,
        *,
        is_baseline: bool = False,
        thresholding_method: str | None = None,
        threshold_strategy: str | None = None,
        threshold_correction_factor: str | None = None,
        threshold_smoothing_scale: str | None = None,
        threshold_bounds: str | None = None,
        notes: str | None = None,
    ) -> None:
        nonlocal next_id
        variant_id = f"{next_id:03d}_{slug}"
        next_id += 1
        specs.append(
            ThresholdVariantSpec(
                variant_id=variant_id,
                display_name=display_name,
                target_module_index=profile.module_index,
                target_module_num=profile.module_num,
                thresholding_method=thresholding_method,
                threshold_strategy=threshold_strategy,
                threshold_correction_factor=threshold_correction_factor,
                threshold_smoothing_scale=threshold_smoothing_scale,
                threshold_bounds=threshold_bounds,
                notes=notes,
                is_baseline=is_baseline,
            )
        )

    append_spec("baseline", "Baseline (original)", is_baseline=True)

    append_spec(
        "otsu_global",
        "Otsu Global",
        thresholding_method=OTSU_METHOD,
        threshold_strategy=GLOBAL_STRATEGY,
    )

    for correction_factor in ("0.9", "1.0", "1.1"):
        slug = f"otsu_adaptive_cf_{correction_factor.replace('.', '_')}"
        append_spec(
            slug,
            f"Otsu Adaptive (CF {correction_factor})",
            thresholding_method=OTSU_METHOD,
            threshold_strategy=ADAPTIVE_STRATEGY,
            threshold_correction_factor=correction_factor,
        )

    append_spec(
        "mce_global",
        "Minimum Cross-Entropy Global",
        thresholding_method=MINIMUM_CROSS_ENTROPY_METHOD,
        threshold_strategy=GLOBAL_STRATEGY,
    )

    for correction_factor in ("0.9", "1.0", "1.1"):
        slug = f"mce_adaptive_cf_{correction_factor.replace('.', '_')}"
        append_spec(
            slug,
            f"Minimum Cross-Entropy Adaptive (CF {correction_factor})",
            thresholding_method=MINIMUM_CROSS_ENTROPY_METHOD,
            threshold_strategy=ADAPTIVE_STRATEGY,
            threshold_correction_factor=correction_factor,
        )

    if profile_supports_robust_background(profile):
        append_spec(
            "robust_background_global",
            "Robust Background Global",
            thresholding_method=ROBUST_BACKGROUND_METHOD,
            threshold_strategy=GLOBAL_STRATEGY,
            notes="Included because the imported IPO module supports Robust Background.",
        )

    return specs


def _resolve_under_directory(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to write outside the variants directory: {resolved}"
        ) from exc
    return resolved


def _variant_directory(output_variants_dir: Path, spec: ThresholdVariantSpec) -> Path:
    return output_variants_dir / f"variant_{spec.variant_id}"


def _validate_specs_against_pipeline(
    pipeline: CppipePipeline,
    specs: Iterable[ThresholdVariantSpec],
) -> None:
    for spec in specs:
        if spec.target_module_index < 0 or spec.target_module_index >= len(
            pipeline.modules
        ):
            raise ValueError(
                f"Variant {spec.variant_id} targets invalid module index "
                f"{spec.target_module_index}."
            )
        module = pipeline.modules[spec.target_module_index]
        if module.name != "IdentifyPrimaryObjects":
            raise ValueError(
                f"Variant {spec.variant_id} targets module index "
                f"{spec.target_module_index}, which is {module.name!r}, not "
                "IdentifyPrimaryObjects."
            )
        if (
            spec.target_module_num is not None
            and module.module_num != spec.target_module_num
        ):
            raise ValueError(
                f"Variant {spec.variant_id} expected module_num "
                f"{spec.target_module_num}, found {module.module_num}."
            )


def apply_threshold_variant_spec(
    pipeline: CppipePipeline,
    spec: ThresholdVariantSpec,
) -> CppipePipeline:
    """Return a copy of ``pipeline`` with threshold settings applied for one spec."""
    if spec.is_baseline:
        return pipeline

    updated = pipeline
    for field_name, setting_key in _THRESHOLD_SETTING_APPLY_KEYS:
        value = getattr(spec, field_name)
        if value is not None:
            updated = update_module_setting(
                updated,
                spec.target_module_index,
                setting_key,
                value,
            )
    return updated


def write_threshold_pipeline_variants(
    imported_cppipe_path: str | Path,
    output_variants_dir: str | Path,
    specs: Iterable[ThresholdVariantSpec],
) -> list[ThresholdVariantArtifact]:
    """Write candidate pipeline variants without modifying the imported ``.cppipe``."""
    imported_path = Path(imported_cppipe_path).resolve()
    if not imported_path.is_file():
        raise FileNotFoundError(
            f"CellProfiler pipeline file not found: {imported_path}"
        )

    variants_root = Path(output_variants_dir).resolve()
    variants_root.mkdir(parents=True, exist_ok=True)

    spec_list = list(specs)
    if not spec_list:
        return []

    pipeline = load_cppipe(imported_path)
    _validate_specs_against_pipeline(pipeline, spec_list)

    artifacts: list[ThresholdVariantArtifact] = []
    for spec in spec_list:
        variant_dir = _resolve_under_directory(
            _variant_directory(variants_root, spec),
            variants_root,
        )
        variant_dir.mkdir(parents=True, exist_ok=True)
        pipeline_path = _resolve_under_directory(
            variant_dir / "pipeline.cppipe",
            variants_root,
        )

        if pipeline_path.resolve() == imported_path:
            raise ValueError(
                "Refusing to overwrite the imported pipeline file with a variant."
            )

        if spec.is_baseline:
            shutil.copy2(imported_path, pipeline_path)
        else:
            variant_pipeline = apply_threshold_variant_spec(pipeline, spec)
            save_cppipe(variant_pipeline, pipeline_path)

        artifacts.append(
            ThresholdVariantArtifact(
                spec=spec,
                variant_dir=variant_dir,
                pipeline_path=pipeline_path,
            )
        )

    return artifacts
