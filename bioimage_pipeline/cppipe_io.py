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
MINIMAL_GUI_PIPELINE_MODULES = (
    "Images",
    "Metadata",
    "NamesAndTypes",
    "IdentifyPrimaryObjects",
    "SaveImages",
    "ExportToSpreadsheet",
)


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
    for setting in module.settings:
        if setting.key == setting_key:
            lines[setting.line_index] = f"{setting.key}:{value}"
            break
    else:
        lines.append(f"{setting_key}:{value}")

    modules[module_index] = _module_from_lines(lines, module.start_line, module.module_num)
    return CppipePipeline(
        preamble=list(pipeline.preamble),
        modules=modules,
        trailing=list(pipeline.trailing),
        source_path=pipeline.source_path,
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


def module_template(module: ModuleDefinition, *, module_num: int = 1) -> CppipeModule:
    """Create a conservative ``.cppipe`` module block for a catalog module."""
    lines = [
        (
            f"{module.name}:[module_num:{module_num}|svn_version:'Unknown'|"
            f"variable_revision_number:{module.variable_revision_number}|"
            "show_window:False|notes:[]|batch_state:array([], dtype=uint8)|"
            "enabled:True|wants_pause:False]"
        )
    ]
    lines.extend(f"    {parameter.label}:{parameter.default}" for parameter in module.parameters)
    lines.append("")
    return _module_from_lines(lines, start_line=0, fallback_num=module_num)


def summarize_modules(modules: Iterable[CppipeModule]) -> list[str]:
    """Return compact module display strings for GUI list boxes."""
    return [module.display_name for module in modules]
