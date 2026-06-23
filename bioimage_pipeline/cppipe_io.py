"""Load, inspect, and save CellProfiler ``.cppipe`` pipeline files.

CellProfiler pipeline files are text files with version-specific details. This
module uses a conservative text-preserving parser: it identifies module blocks
and common ``setting:value`` lines without trying to reimplement CellProfiler's
full schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from bioimage_pipeline.pipeline_catalog import ModuleDefinition, get_module_definition

_MODULE_HEADER_RE = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9_]*):\[(?P<meta>.*)\]$")
_MODULE_NUM_RE = re.compile(r"module_num:(?P<num>\d+)")
DEFAULT_CPPIPE_PREAMBLE = [
    "CellProfiler Pipeline: http://www.cellprofiler.org",
    "Version:5",
    "DateRevision:400",
    "GitHash:",
    "ModuleCount:0",
    "HasImagePlaneDetails:False",
    "",
]
REQUIRED_SETUP_MODULES = (
    "Images",
    "Metadata",
    "NamesAndTypes",
    "Groups",
)
MINIMAL_GUI_PIPELINE_MODULES = REQUIRED_SETUP_MODULES
GUI_MANAGED_CPPIPE_SETTINGS = frozenset({"Input folder path"})
INPUT_MODULES_TO_RESET_FOR_CELLPROFILER = frozenset(
    {"Metadata", "NamesAndTypes", "Groups"},
)
ANALYSIS_MODULES_REQUIRING_EXPORT = frozenset({"IdentifyPrimaryObjects"})


@dataclass
class CppipeSetting:
    """A display/editable setting parsed from a module block."""

    key: str
    value: str
    line_index: int


@dataclass
class CppipeModule:
    """A module block inside a CellProfiler pipeline."""

    name: str
    module_num: int
    lines: list[str]
    start_line: int = 0
    settings: list[CppipeSetting] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return f"{self.module_num}. {self.name}"


@dataclass
class CppipePipeline:
    """A text-preserving representation of a CellProfiler pipeline."""

    preamble: list[str]
    modules: list[CppipeModule]
    trailing: list[str] = field(default_factory=list)
    source_path: Path | None = None

    def to_text(self) -> str:
        """Serialize the pipeline back to ``.cppipe`` text."""
        renumbered = renumber_modules(self)
        preamble = _with_module_count(renumbered.preamble, len(renumbered.modules))
        lines = [
            *preamble,
            *[line for module in renumbered.modules for line in module.lines],
            *renumbered.trailing,
        ]
        return "\n".join(lines).rstrip() + "\n"


def _parse_module_num(header: str, fallback: int) -> int:
    match = _MODULE_NUM_RE.search(header)
    if match:
        return int(match.group("num"))
    return fallback


def _parse_settings(lines: list[str]) -> list[CppipeSetting]:
    settings: list[CppipeSetting] = []
    for index, line in enumerate(lines[1:], start=1):
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if not key or key.startswith("#"):
            continue
        settings.append(CppipeSetting(key=key, value=value.strip(), line_index=index))
    return settings


def parse_cppipe_text(text: str, *, source_path: str | Path | None = None) -> CppipePipeline:
    """Parse CellProfiler pipeline text into module blocks."""
    raw_lines = text.splitlines()
    preamble: list[str] = []
    modules: list[CppipeModule] = []
    current: list[str] | None = None
    current_start = 0

    for line_number, line in enumerate(raw_lines):
        if _MODULE_HEADER_RE.match(line):
            if current is None:
                current = [line]
                current_start = line_number
            else:
                modules.append(_module_from_lines(current, current_start, len(modules) + 1))
                current = [line]
                current_start = line_number
            continue

        if current is None:
            preamble.append(line)
        else:
            current.append(line)

    if current is not None:
        modules.append(_module_from_lines(current, current_start, len(modules) + 1))

    return CppipePipeline(
        preamble=preamble,
        modules=modules,
        source_path=Path(source_path) if source_path is not None else None,
    )


def _module_from_lines(lines: list[str], start_line: int, fallback_num: int) -> CppipeModule:
    header = lines[0]
    match = _MODULE_HEADER_RE.match(header)
    if match is None:
        raise ValueError(f"Invalid CellProfiler module header: {header}")
    module = CppipeModule(
        name=match.group("name"),
        module_num=_parse_module_num(header, fallback_num),
        lines=list(lines),
        start_line=start_line,
    )
    module.settings = _parse_settings(module.lines)
    return module


def load_cppipe(path: str | Path) -> CppipePipeline:
    """Load a CellProfiler ``.cppipe`` file."""
    cppipe_path = Path(path)
    if not cppipe_path.is_file():
        raise FileNotFoundError(f"CellProfiler pipeline file not found: {cppipe_path}")
    return parse_cppipe_text(cppipe_path.read_text(encoding="utf-8"), source_path=cppipe_path)


def save_cppipe(pipeline: CppipePipeline, path: str | Path) -> Path:
    """Save a pipeline to a ``.cppipe`` file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(pipeline.to_text(), encoding="utf-8")
    return output_path.resolve()


def validate_cppipe(pipeline: CppipePipeline) -> list[str]:
    """Return validation messages for GUI-edited pipelines."""
    errors: list[str] = []
    if not pipeline.modules:
        errors.append("Pipeline has no CellProfiler modules.")

    seen_numbers: set[int] = set()
    for module in pipeline.modules:
        if module.module_num in seen_numbers:
            errors.append(f"Duplicate module number: {module.module_num}")
        seen_numbers.add(module.module_num)
        if not module.name:
            errors.append("Module with empty name found.")
        if not module.lines or not _MODULE_HEADER_RE.match(module.lines[0]):
            errors.append(f"Module {module.display_name} has an invalid header.")

    return errors


def renumber_modules(pipeline: CppipePipeline) -> CppipePipeline:
    """Return a copy with sequential module numbers in headers."""
    modules: list[CppipeModule] = []
    for index, module in enumerate(pipeline.modules, start=1):
        lines = list(module.lines)
        lines[0] = _replace_module_num(lines[0], index)
        new_module = _module_from_lines(lines, module.start_line, index)
        new_module.module_num = index
        modules.append(new_module)
    return CppipePipeline(
        preamble=list(pipeline.preamble),
        modules=modules,
        trailing=list(pipeline.trailing),
        source_path=pipeline.source_path,
    )


def _replace_module_num(header: str, module_num: int) -> str:
    if _MODULE_NUM_RE.search(header):
        return _MODULE_NUM_RE.sub(f"module_num:{module_num}", header, count=1)
    if header.endswith("]"):
        return header[:-1] + f"|module_num:{module_num}]"
    return header


def _with_module_count(preamble: list[str], module_count: int) -> list[str]:
    """Return a CellProfiler preamble with a current ``ModuleCount`` line."""
    lines = list(preamble) if preamble else list(DEFAULT_CPPIPE_PREAMBLE)
    for index, line in enumerate(lines):
        if line.startswith("ModuleCount:"):
            lines[index] = f"ModuleCount:{module_count}"
            return lines

    insert_at = len(lines)
    while insert_at > 0 and lines[insert_at - 1] == "":
        insert_at -= 1
    lines.insert(insert_at, f"ModuleCount:{module_count}")
    return lines


def move_module(pipeline: CppipePipeline, from_index: int, to_index: int) -> CppipePipeline:
    """Move a module by zero-based index and return a renumbered pipeline."""
    modules = list(pipeline.modules)
    module = modules.pop(from_index)
    modules.insert(to_index, module)
    return renumber_modules(
        CppipePipeline(
            preamble=list(pipeline.preamble),
            modules=modules,
            trailing=list(pipeline.trailing),
            source_path=pipeline.source_path,
        )
    )


def remove_module(pipeline: CppipePipeline, index: int) -> CppipePipeline:
    """Remove a module by zero-based index and return a renumbered pipeline."""
    modules = list(pipeline.modules)
    modules.pop(index)
    return renumber_modules(
        CppipePipeline(
            preamble=list(pipeline.preamble),
            modules=modules,
            trailing=list(pipeline.trailing),
            source_path=pipeline.source_path,
        )
    )


def rewrite_groups_module_settings(
    pipeline: CppipePipeline,
    module_index: int,
    *,
    wants_groups: str | None = None,
    metadata_categories: list[str] | None = None,
) -> CppipePipeline:
    """Rewrite a Groups module body to match CellProfiler v2 structure."""
    modules = list(pipeline.modules)
    module = modules[module_index]
    if module.name != "Groups":
        raise ValueError(f"Expected Groups module, got {module.name!r}")

    current_wants = "No"
    current_categories: list[str] = []
    for setting in module.settings:
        if setting.key == "Do you want to group your images?":
            current_wants = setting.value
        elif setting.key == "Metadata category":
            current_categories.append(setting.value)
    if not current_categories:
        current_categories = ["None"]

    resolved_wants = current_wants if wants_groups is None else wants_groups
    resolved_categories = (
        current_categories if metadata_categories is None else metadata_categories
    )
    if not resolved_categories:
        resolved_categories = ["None"]

    lines = [
        module.lines[0],
        f"    Do you want to group your images?:{resolved_wants}",
        f"    grouping metadata count:{len(resolved_categories)}",
        *(
            f"    Metadata category:{category}"
            for category in resolved_categories
        ),
        "",
    ]
    modules[module_index] = _module_from_lines(
        lines, module.start_line, module.module_num,
    )
    return CppipePipeline(
        preamble=list(pipeline.preamble),
        modules=modules,
        trailing=list(pipeline.trailing),
        source_path=pipeline.source_path,
    )


def _append_module_setting_line(lines: list[str], setting_key: str, value: str) -> list[str]:
    """Append an indented setting line before the module's trailing blank line."""
    trimmed = list(lines)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    trimmed.append(f"    {setting_key}:{value}")
    trimmed.append("")
    return trimmed


def _sync_module_settings_from_catalog(module: CppipeModule) -> CppipeModule:
    """Add catalog default settings that are visible but missing from a module block."""
    try:
        definition = get_module_definition(module.name)
    except KeyError:
        return module

    settings_dict = _module_settings_dict(module)
    lines = list(module.lines)
    for parameter in definition.parameters:
        if parameter.label in settings_dict:
            continue
        if parameter.label in GUI_MANAGED_CPPIPE_SETTINGS:
            continue
        if not (
            parameter.internal
            or parameter.visibility.is_visible(settings_dict)
        ):
            continue
        lines = _append_module_setting_line(lines, parameter.label, parameter.default)
        settings_dict[parameter.label] = parameter.default

    return _module_from_lines(lines, module.start_line, module.module_num)


def _setting_controls_visibility(module_name: str, setting_key: str) -> bool:
    """Return whether changing ``setting_key`` can reveal other catalog settings."""
    try:
        definition = get_module_definition(module_name)
    except KeyError:
        return False
    for parameter in definition.parameters:
        visibility = parameter.visibility
        if visibility.mode == "conditional" and visibility.setting_label == setting_key:
            return True
    return False


def update_module_setting(
    pipeline: CppipePipeline,
    module_index: int,
    setting_key: str,
    value: str,
) -> CppipePipeline:
    """Update or append a module setting by display key."""
    modules = list(pipeline.modules)
    module = modules[module_index]
    lines = list(module.lines)
    updated = False
    for setting in module.settings:
        if setting.key == setting_key:
            original = lines[setting.line_index]
            indent = original[: len(original) - len(original.lstrip())]
            lines[setting.line_index] = f"{indent}{setting.key}:{value}"
            updated = True
    if not updated:
        lines = _append_module_setting_line(lines, setting_key, value)

    new_module = _module_from_lines(lines, module.start_line, module.module_num)
    if (
        new_module.name in REQUIRED_SETUP_MODULES
        and _setting_controls_visibility(new_module.name, setting_key)
    ):
        new_module = _sync_module_settings_from_catalog(new_module)
    modules[module_index] = new_module
    return CppipePipeline(
        preamble=list(pipeline.preamble),
        modules=modules,
        trailing=list(pipeline.trailing),
        source_path=pipeline.source_path,
    )


def _module_settings_dict(module: CppipeModule) -> dict[str, str]:
    return {setting.key: setting.value for setting in module.settings}


def _primary_object_name(pipeline: CppipePipeline) -> str:
    reserved = frozenset(
        {"IdentifyPrimaryObjects", "IdentifySecondaryObjects", "SaveImages"},
    )
    for module in pipeline.modules:
        if module.name != "IdentifyPrimaryObjects":
            continue
        for setting in module.settings:
            if setting.key == "Name the primary objects to be identified":
                name = setting.value.strip()
                if name and name not in reserved:
                    return name
    return "Nuclei"


def save_images_needs_normalization(settings: dict[str, str]) -> bool:
    """Return whether SaveImages settings would skip mask/QC export paths."""
    save_type = settings.get("Select the type of image to save", "Image")
    if save_type == "Image":
        return True
    if settings.get("Select method for constructing file names") == "Sequential numbers":
        return True

    prefix = settings.get("Enter file prefix", "").lower()
    suffix = settings.get("Text to append to the image name", "").lower()
    if settings.get("Append a suffix to the image file name?", "No") == "Yes":
        combined = f"{prefix}{suffix}"
    else:
        combined = prefix

    if save_type == "Mask":
        return "mask" not in combined
    if save_type == "Objects":
        return not any(
            keyword in combined for keyword in ("object", "label", "segmented")
        )
    return False


def _rewrite_save_images_module(
    pipeline: CppipePipeline,
    module_index: int,
    *,
    object_name: str,
    mask_prefix: str,
) -> CppipePipeline:
    """Write a headless-safe SaveImages block that names files from DNA filenames."""
    definition = get_module_definition("SaveImages")
    modules = list(pipeline.modules)
    modules[module_index] = module_template(
        definition,
        module_num=modules[module_index].module_num,
        include_hidden=False,
    )
    updated = CppipePipeline(
        preamble=list(pipeline.preamble),
        modules=modules,
        trailing=list(pipeline.trailing),
        source_path=pipeline.source_path,
    )
    updated = update_module_setting(
        updated, module_index, "Select the type of image to save", "Mask",
    )
    updated = update_module_setting(
        updated, module_index, "Select the image to save", object_name,
    )
    updated = update_module_setting(
        updated,
        module_index,
        "Select method for constructing file names",
        "From image filename",
    )
    updated = update_module_setting(
        updated, module_index, "Select image name for file prefix", "DNA",
    )
    updated = update_module_setting(updated, module_index, "Enter file prefix", mask_prefix)
    updated = update_module_setting(
        updated, module_index, "Append a suffix to the image file name?", "No",
    )
    updated = update_module_setting(updated, module_index, "Image bit depth", "8-bit integer")
    return updated


def normalize_save_images_for_cellprofiler(pipeline: CppipePipeline) -> CppipePipeline:
    """Replace SaveImages with a catalog template that exports object masks."""
    object_name = _primary_object_name(pipeline)
    mask_prefix = f"{object_name}_mask"
    for index, module in enumerate(pipeline.modules):
        if module.name != "SaveImages":
            continue
        return _rewrite_save_images_module(
            pipeline,
            index,
            object_name=object_name,
            mask_prefix=mask_prefix,
        )
    return pipeline


def normalize_save_images_in_pipeline(pipeline: CppipePipeline) -> CppipePipeline:
    """Rewrite misconfigured SaveImages modules to export segmentation masks."""
    updated = pipeline
    object_name = _primary_object_name(pipeline)
    mask_prefix = f"{object_name}_mask"
    for index, module in enumerate(pipeline.modules):
        if module.name != "SaveImages":
            continue
        settings = _module_settings_dict(module)
        if not save_images_needs_normalization(settings):
            if (
                settings.get("Select the type of image to save") == "Mask"
                and settings.get("Select the image to save") != object_name
            ):
                updated = update_module_setting(
                    updated, index, "Select the image to save", object_name,
                )
            continue
        updated = _rewrite_save_images_module(
            updated,
            index,
            object_name=object_name,
            mask_prefix=mask_prefix,
        )
    return updated


def _clean_module_lines_for_cellprofiler(module_name: str, lines: list[str]) -> list[str]:
    """Drop GUI-managed settings and malformed lines CellProfiler cannot load."""
    header = lines[0]
    cleaned: list[str] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        if not line.startswith("    "):
            continue
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key = stripped.split(":", 1)[0]
        if not key or key in GUI_MANAGED_CPPIPE_SETTINGS:
            continue
        cleaned.append(line)
    return [header, *cleaned, ""]


def normalize_identify_primary_objects_for_cellprofiler(
    pipeline: CppipePipeline,
) -> CppipePipeline:
    """Reset IdentifyPrimaryObjects to a catalog template CellProfiler can load."""
    modules = list(pipeline.modules)
    for index, module in enumerate(modules):
        if module.name != "IdentifyPrimaryObjects":
            continue
        definition = get_module_definition("IdentifyPrimaryObjects")
        modules[index] = module_template(
            definition, module_num=module.module_num, include_hidden=True,
        )
        updated = CppipePipeline(
            preamble=list(pipeline.preamble),
            modules=modules,
            trailing=list(pipeline.trailing),
            source_path=pipeline.source_path,
        )
        return update_module_setting(
            updated,
            index,
            "Name the primary objects to be identified",
            _primary_object_name(pipeline),
        )
    return pipeline


def normalize_input_modules_for_cellprofiler(
    pipeline: CppipePipeline,
) -> CppipePipeline:
    """Reset input modules to catalog templates CellProfiler headless can load."""
    modules = list(pipeline.modules)
    for index, module in enumerate(modules):
        if module.name not in INPUT_MODULES_TO_RESET_FOR_CELLPROFILER:
            continue
        definition = get_module_definition(module.name)
        modules[index] = module_template(
            definition, module_num=module.module_num, include_hidden=True,
        )
    return CppipePipeline(
        preamble=list(pipeline.preamble),
        modules=modules,
        trailing=list(pipeline.trailing),
        source_path=pipeline.source_path,
    )


def load_and_validate_imported_pipeline(cppipe_path: str | Path) -> CppipePipeline:
    """Load a user-authored ``.cppipe`` file and validate structure only."""
    path = Path(cppipe_path)
    if not path.is_file():
        raise FileNotFoundError(f"CellProfiler pipeline file not found: {path}")
    pipeline = load_cppipe(path)
    errors = validate_cppipe(pipeline)
    if errors:
        raise ValueError("\n".join(errors))
    return pipeline


def advise_pipeline_for_run(pipeline: CppipePipeline) -> list[str]:
    """Return advisory warnings before running an imported pipeline headlessly."""
    advisories: list[str] = []
    module_names = {module.name for module in pipeline.modules}
    if not module_names:
        advisories.append("Pipeline has no CellProfiler modules.")
        return advisories
    if "ExportToSpreadsheet" not in module_names:
        advisories.append(
            "No ExportToSpreadsheet module found — measurement CSVs may be empty."
        )
    if "SaveImages" not in module_names:
        advisories.append(
            "No SaveImages module found — mask/label TIFF exports and QC overlays "
            "may be empty."
        )
    for module in pipeline.modules:
        if module.name != "Images":
            continue
        for setting in module.settings:
            if setting.key != "Input folder path":
                continue
            folder = setting.value.strip()
            if not folder:
                continue
            folder_path = Path(folder)
            if not folder_path.is_dir():
                advisories.append(
                    "Images module references an input folder path that does not "
                    f"exist on this machine: {folder_path}"
                )
    return advisories


def ensure_export_to_spreadsheet(pipeline: CppipePipeline) -> CppipePipeline:
    """Append ExportToSpreadsheet when analysis modules need measurement CSVs."""
    module_names = {module.name for module in pipeline.modules}
    if "ExportToSpreadsheet" in module_names:
        return pipeline
    if not module_names.intersection(ANALYSIS_MODULES_REQUIRING_EXPORT):
        return pipeline
    return append_module(pipeline, "ExportToSpreadsheet")


def prepare_pipeline_for_cellprofiler(
    pipeline: CppipePipeline,
    *,
    apply_legacy_rewrites: bool = False,
) -> CppipePipeline:
    """Return a copy of a pipeline for headless CellProfiler.

    By default (import-only mode) the pipeline is returned unchanged. Set
    ``apply_legacy_rewrites=True`` to apply deprecated GUI-builder normalization
    (SaveImages rewrite, module template resets, auto ExportToSpreadsheet).
    """
    if not apply_legacy_rewrites:
        return CppipePipeline(
            preamble=list(pipeline.preamble),
            modules=list(pipeline.modules),
            trailing=list(pipeline.trailing),
            source_path=pipeline.source_path,
        )

    import warnings

    warnings.warn(
        "prepare_pipeline_for_cellprofiler(apply_legacy_rewrites=True) is deprecated "
        "and will be removed. Author pipelines in CellProfiler and run them import-only.",
        DeprecationWarning,
        stacklevel=2,
    )
    prepared = normalize_input_modules_for_cellprofiler(pipeline)
    prepared = normalize_identify_primary_objects_for_cellprofiler(prepared)
    prepared = normalize_save_images_for_cellprofiler(prepared)
    prepared = ensure_export_to_spreadsheet(prepared)
    prepared = renumber_modules(prepared)
    modules = [
        _module_from_lines(
            _clean_module_lines_for_cellprofiler(module.name, list(module.lines)),
            module.start_line,
            module.module_num,
        )
        for module in prepared.modules
    ]
    return CppipePipeline(
        preamble=list(prepared.preamble),
        modules=modules,
        trailing=list(prepared.trailing),
        source_path=prepared.source_path,
    )


def append_module(
    pipeline: CppipePipeline,
    module: ModuleDefinition | str,
) -> CppipePipeline:
    """Append a catalog module template and return a renumbered pipeline."""
    definition = get_module_definition(module) if isinstance(module, str) else module
    modules = list(pipeline.modules)
    modules.append(module_template(definition, module_num=len(modules) + 1))
    return renumber_modules(
        CppipePipeline(
            preamble=list(pipeline.preamble),
            modules=modules,
            trailing=list(pipeline.trailing),
            source_path=pipeline.source_path,
        )
    )


def create_pipeline_from_catalog(
    module_names: Iterable[str] = MINIMAL_GUI_PIPELINE_MODULES,
) -> CppipePipeline:
    """Create a new ``.cppipe`` pipeline from catalog module templates."""
    pipeline = CppipePipeline(
        preamble=list(DEFAULT_CPPIPE_PREAMBLE),
        modules=[],
    )
    for module_name in module_names:
        pipeline = append_module(pipeline, module_name)
    return pipeline


def module_template(
    module: ModuleDefinition,
    *,
    module_num: int = 1,
    include_hidden: bool = False,
    setting_overrides: dict[str, str] | None = None,
) -> CppipeModule:
    """Create a conservative ``.cppipe`` module block for a catalog module."""
    lines = [
        (
            f"{module.name}:[module_num:{module_num}|svn_version:'Unknown'|"
            f"variable_revision_number:{module.variable_revision_number}|"
            "show_window:False|notes:[]|batch_state:array([], dtype=uint8)|"
            "enabled:True|wants_pause:False]"
        )
    ]
    default_settings = {
        parameter.label: parameter.default for parameter in module.parameters
    }
    if setting_overrides:
        default_settings.update(setting_overrides)
    for parameter in module.parameters:
        if parameter.label in GUI_MANAGED_CPPIPE_SETTINGS:
            continue
        if parameter.internal and not include_hidden:
            if parameter.visibility.mode != "always":
                continue
        if parameter.internal and parameter.default in ("[]", "{}", "None|None"):
            if parameter.visibility.mode != "always":
                continue
        if not parameter.visibility.is_visible(default_settings):
            continue
        lines.append(f"    {parameter.label}:{default_settings[parameter.label]}")
    lines.append("")
    return _module_from_lines(lines, start_line=0, fallback_num=module_num)


def summarize_modules(modules: Iterable[CppipeModule]) -> list[str]:
    """Return compact module display strings for GUI list boxes."""
    return [module.display_name for module in modules]
