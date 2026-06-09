"""Batch Z-max projection for Olympus .oir files (Fiji macro parity)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from bioimage_pipeline.z_projection import (
    OirPythonReadError,
    format_oir_read_dependency_error,
    iter_oir_files,
    oir_output_path,
    process_oir_file_python,
)

DEFAULT_MACRO_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "fiji_macros"
    / "stacking_zmax.ijm"
)

OirZmaxEngine = Literal["fiji", "python", "auto"]


@dataclass
class OirFilePair:
    """One input ``.oir`` file and its projected output TIFF path."""

    input_oir: Path
    output_tif: Path

    @property
    def bioformats_import_command(self) -> str:
        """The exact Fiji macro command used to import this file."""
        options = (
            f"open=[{_ijm_path(self.input_oir)}] autoscale "
            "view=Hyperstack stack_format=Default"
        )
        return f'run("Bio-Formats Windowless Importer", "{options}");'


@dataclass
class OirZmaxBatchResult:
    """Summary of a completed OIR Z-max batch run."""

    input_dir: Path
    output_dir: Path
    engine: str
    fiji_executable: Path | None = None
    fiji_headless: bool | None = None
    manual_macro_path: Path | None = None
    file_pairs: list[OirFilePair] = field(default_factory=list)
    processed: list[str] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)


def _normalize_input_dir(input_dir: str | Path) -> Path:
    path = Path(input_dir)
    if not path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {path}")
    return path.resolve()


def _build_file_pairs(oir_files: list[Path], output_dir: Path) -> list[OirFilePair]:
    return [
        OirFilePair(
            input_oir=oir_path.resolve(),
            output_tif=oir_output_path(oir_path, output_dir).resolve(),
        )
        for oir_path in oir_files
    ]


def _ijm_path(path: Path) -> str:
    """Return an absolute path string safe for ImageJ macro literals."""
    return str(path.resolve()).replace("\\", "/").replace('"', '\\"')


def _verify_existing_outputs(file_pairs: list[OirFilePair]) -> list[str]:
    """Return output names that already exist on disk."""
    processed: list[str] = []
    for pair in file_pairs:
        if pair.output_tif.is_file():
            processed.append(pair.output_tif.name)
    return processed


def build_manual_oir_zmax_macro(input_dir: Path, output_dir: Path) -> str:
    """Build a Fiji macro that users run manually from the Fiji GUI."""
    input_text = _ijm_path(input_dir)
    output_text = _ijm_path(output_dir)
    return f"""// Auto-generated OIR Z-max macro.
// Run this from Fiji GUI: File > Open... or Plugins > Macros > Run...
// It uses Bio-Formats Windowless Importer, not open() and not BFConvert.

input = ensureTrailingSeparator("{input_text}");
output = ensureTrailingSeparator("{output_text}");

setBatchMode(true);
if (!File.exists(output)) {{
    File.makeDirectory(output);
}}

function ensureTrailingSeparator(path) {{
    if (endsWith(path, "/") || endsWith(path, "\\\\")) {{
        return path;
    }}
    return path + "/";
}}

function processFolder(dir) {{
    dir = ensureTrailingSeparator(dir);
    list = getFileList(dir);
    for (i = 0; i < list.length; i++) {{
        name = list[i];
        lower = toLowerCase(name);
        if (endsWith(lower, ".oir")) {{
            processFile(dir, output, name);
        }} else if (endsWith(name, "/")) {{
            processFolder(dir + name);
        }} else {{
            print("Skipping non-image file: " + dir + name);
        }}
    }}
}}

function processFile(inputFolder, outputFolder, file) {{
    inputFolder = ensureTrailingSeparator(inputFolder);
    outputFolder = ensureTrailingSeparator(outputFolder);
    inputPath = inputFolder + file;
    if (!File.exists(inputPath)) {{
        print("File not found: " + inputPath);
        return;
    }}
    if (!File.exists(outputFolder)) {{
        File.makeDirectory(outputFolder);
    }}

    importOptions = "open=[" + inputPath + "] autoscale view=Hyperstack stack_format=Default";
    print("Bio-Formats macro command: run(\\"Bio-Formats Windowless Importer\\", \\"" + importOptions + "\\");");
    run("Bio-Formats Windowless Importer", importOptions);
    if (nImages == 0) {{
        print("Import failed: " + inputPath);
        return;
    }}
    selectImage(1);
    run("Z Project...", "projection=[Max Intensity]");
    saveName = replace(file, ".oir", ".tif");
    saveAs("Tiff", outputFolder + saveName);
    close("*");
}}

processFolder(input);
setBatchMode(false);
"""


def write_manual_oir_zmax_macro(input_dir: Path, output_dir: Path) -> Path:
    """Write the manual-run Fiji macro into the output directory."""
    macro_path = output_dir / "run_oir_zmax_manual.ijm"
    macro_path.write_text(
        build_manual_oir_zmax_macro(input_dir, output_dir),
        encoding="utf-8",
    )
    return macro_path.resolve()


def _resolve_engine(engine: OirZmaxEngine) -> str:
    if engine == "python":
        return "python"
    return "fiji"


def run_oir_zmax_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    engine: OirZmaxEngine = "fiji",
    fiji_executable: str | Path | None = None,
    fiji_headless: bool | None = None,
    fiji_timeout: float | None = None,
) -> OirZmaxBatchResult:
    """Z-max project every ``.oir`` file under *input_dir* into *output_dir*.

    The default Fiji path now writes a manual-run macro instead of invoking Fiji
    with command-line ``-macro``. This avoids the Bio-Formats ``MainDialog``
    VerifyError seen when importing ``.oir`` from command-line macro mode.
    """
    input_path = _normalize_input_dir(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ``fiji_executable``, ``fiji_headless``, and ``fiji_timeout`` are accepted
    # for CLI/API compatibility, but the Fiji OIR path intentionally does not
    # launch Fiji. It writes a manual-run macro to avoid command-line import
    # failures in Bio-Formats.
    _ = (fiji_executable, fiji_headless, fiji_timeout)

    resolved_engine = _resolve_engine(engine)
    oir_files = list(iter_oir_files(input_path))
    file_pairs = _build_file_pairs(oir_files, output_path)

    if resolved_engine == "fiji":
        manual_macro_path = write_manual_oir_zmax_macro(input_path, output_path)
        processed = _verify_existing_outputs(file_pairs)

        return OirZmaxBatchResult(
            input_dir=input_path,
            output_dir=output_path.resolve(),
            engine="fiji-manual-macro",
            fiji_executable=None,
            fiji_headless=None,
            manual_macro_path=manual_macro_path,
            file_pairs=file_pairs,
            processed=processed,
            failed=[],
        )

    processed: list[str] = []
    failed: list[dict[str, Any]] = []
    for pair in file_pairs:
        try:
            process_oir_file_python(pair.input_oir, output_path)
            if pair.output_tif.is_file():
                processed.append(pair.output_tif.name)
            else:
                failed.append(
                    {
                        "file": pair.input_oir.name,
                        "input_oir": str(pair.input_oir),
                        "output_tif": str(pair.output_tif),
                        "import_command": pair.bioformats_import_command,
                        "error": "Expected output file was not created.",
                    }
                )
        except OirPythonReadError as exc:
            failed.append(
                {
                    "file": pair.input_oir.name,
                    "input_oir": str(pair.input_oir),
                    "output_tif": str(pair.output_tif),
                    "import_command": pair.bioformats_import_command,
                    "error": str(exc),
                }
            )
        except Exception as exc:
            failed.append(
                {
                    "file": pair.input_oir.name,
                    "input_oir": str(pair.input_oir),
                    "output_tif": str(pair.output_tif),
                    "import_command": pair.bioformats_import_command,
                    "error": format_oir_read_dependency_error(exc),
                }
            )

    return OirZmaxBatchResult(
        input_dir=input_path,
        output_dir=output_path.resolve(),
        engine="python",
        fiji_executable=None,
        fiji_headless=None,
        manual_macro_path=None,
        file_pairs=file_pairs,
        processed=processed,
        failed=failed,
    )


def oir_output_name(oir_path: str | Path) -> str:
    """Output basename for an input OIR path (macro naming rules)."""
    from bioimage_pipeline.z_projection import oir_output_filename

    return oir_output_filename(Path(oir_path).name)
