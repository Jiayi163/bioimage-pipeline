"""TrackMate LoG spot detection via Fiji (one subprocess per image or batch)."""

from __future__ import annotations

import csv
import textwrap
from dataclasses import dataclass
from pathlib import Path

from bioimage_pipeline.fiji_runner import find_fiji_executable, run_fiji_macro
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.types import ImagePeakTable, PeakCandidate

DEFAULT_TRACKMATE_MACRO = (
    Path(__file__).resolve().parents[2] / "examples" / "fiji_macros" / "trackmate_log_detect.groovy"
)


@dataclass
class TrackMateDetectResult:
    csv_path: Path
    stdout: str
    stderr: str
    returncode: int


def trackmate_settings(config: PunctaDeclumpConfig) -> dict[str, float]:
    return {
        "radius": config.expected_single_spot_diameter / 2.0,
        "threshold": config.peak_min_relative_height,
    }


def parse_trackmate_csv(csv_path: Path) -> list[PeakCandidate]:
    peaks: list[PeakCandidate] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x_key = "POSITION_X" if "POSITION_X" in row else "x"
            y_key = "POSITION_Y" if "POSITION_Y" in row else "y"
            if x_key not in row or y_key not in row:
                continue
            col = float(row[x_key])
            row_coord = float(row[y_key])
            quality = float(row.get("QUALITY", row.get("quality", 0.0)) or 0.0)
            peaks.append(
                PeakCandidate(
                    row=row_coord,
                    col=col,
                    intensity=quality,
                )
            )
    return peaks


def run_trackmate_on_image(
    image_path: Path,
    output_csv: Path,
    *,
    config: PunctaDeclumpConfig,
    fiji_executable: str | Path | None = None,
    headless: bool = True,
    timeout: float | None = 600.0,
) -> TrackMateDetectResult:
    """Run TrackMate LoG detection once on one image; write spots CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    settings = trackmate_settings(config)
    macro_path = _write_generated_trackmate_macro(
        image_path=image_path.resolve(),
        output_csv=output_csv.resolve(),
        radius=settings["radius"],
        threshold=settings["threshold"],
    )
    result = run_fiji_macro(
        macro_path,
        fiji_executable=fiji_executable,
        headless=headless,
        timeout=timeout,
    )
    if not output_csv.is_file():
        raise RuntimeError(
            f"TrackMate did not produce CSV at {output_csv}. "
            f"stderr={result.stderr[:500]}"
        )
    return TrackMateDetectResult(
        csv_path=output_csv,
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
    )


def run_trackmate_batch(
    image_paths: list[Path],
    output_dir: Path,
    *,
    config: PunctaDeclumpConfig,
    fiji_executable: str | Path | None = None,
    headless: bool = True,
    timeout: float | None = 3600.0,
) -> dict[Path, Path]:
    """Run TrackMate once for multiple images inside one Fiji subprocess."""
    if not image_paths:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    settings = trackmate_settings(config)
    csv_map: dict[Path, Path] = {
        path: output_dir / f"{path.stem}_trackmate.csv" for path in image_paths
    }
    macro_path = _write_generated_trackmate_batch_macro(
        image_paths=[p.resolve() for p in image_paths],
        csv_paths=[p.resolve() for p in csv_map.values()],
        radius=settings["radius"],
        threshold=settings["threshold"],
    )
    result = run_fiji_macro(
        macro_path,
        fiji_executable=fiji_executable,
        headless=headless,
        timeout=timeout,
    )
    missing = [path for path in csv_map.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"TrackMate batch missing CSV outputs: {missing[:3]}. "
            f"stderr={result.stderr[:500]}"
        )
    return csv_map


def peaks_from_trackmate_csv(csv_path: Path) -> ImagePeakTable:
    return ImagePeakTable(
        peaks=parse_trackmate_csv(csv_path),
        detector_name="trackmate",
        method="trackmate_log",
        cache_hit=False,
    )


def _write_generated_trackmate_macro(
    *,
    image_path: Path,
    output_csv: Path,
    radius: float,
    threshold: float,
) -> Path:
    macro_dir = output_csv.parent / ".fiji_macros"
    macro_dir.mkdir(parents=True, exist_ok=True)
    macro_path = macro_dir / "trackmate_log_detect_generated.groovy"
    image_str = str(image_path).replace("\\", "/")
    csv_str = str(output_csv).replace("\\", "/")
    content = textwrap.dedent(
        f"""
        import fiji.plugin.trackmate.Model
        import fiji.plugin.trackmate.Settings
        import fiji.plugin.trackmate.TrackMate
        import fiji.plugin.trackmate.detection.LogDetectorFactory
        import fiji.plugin.trackmate.tracking.jaqaman.SparseLAPTrackerFactory
        import fiji.plugin.trackmate.io.CSVExporter

        imp = IJ.openImage("{image_str}")
        if (imp == null) throw new RuntimeException("Failed to open {image_str}")

        settings = new Settings(imp)
        settings.detectorFactory = new LogDetectorFactory()
        settings.detectorSettings = settings.detectorFactory.getDefaultSettings()
        settings.detectorSettings['RADIUS'] = {radius}
        settings.detectorSettings['THRESHOLD'] = {threshold}
        settings.detectorSettings['DO_SUBPIXEL_LOCALIZATION'] = true

        settings.trackerFactory = new SparseLAPTrackerFactory()
        settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
        settings.trackerSettings['LINKING_MAX_DISTANCE'] = 0.0
        settings.trackerSettings['MAX_FRAME_GAP'] = 0

        model = new Model()
        trackmate = new TrackMate(model, settings)
        if (!trackmate.checkInput() || !trackmate.process()) {{
            throw new RuntimeException(trackmate.getErrorMessage())
        }}

        outFile = new File("{csv_str}")
        CSVExporter.exportSpots(outFile, model, false)
        imp.close()
        """
    ).strip()
    macro_path.write_text(content, encoding="utf-8")
    if find_fiji_executable() is None:
        pass
    return macro_path


def _write_generated_trackmate_batch_macro(
    *,
    image_paths: list[Path],
    csv_paths: list[Path],
    radius: float,
    threshold: float,
) -> Path:
    if len(image_paths) != len(csv_paths):
        raise ValueError("image_paths and csv_paths length mismatch")
    macro_dir = csv_paths[0].parent / ".fiji_macros"
    macro_dir.mkdir(parents=True, exist_ok=True)
    macro_path = macro_dir / "trackmate_log_batch_generated.groovy"

    blocks: list[str] = []
    for image_path, csv_path in zip(image_paths, csv_paths, strict=True):
        image_str = str(image_path).replace("\\", "/")
        csv_str = str(csv_path).replace("\\", "/")
        blocks.append(
            textwrap.dedent(
                f"""
                {{
                    def imp = IJ.openImage("{image_str}")
                    if (imp == null) throw new RuntimeException("Failed to open {image_str}")
                    def settings = new Settings(imp)
                    settings.detectorFactory = new LogDetectorFactory()
                    settings.detectorSettings = settings.detectorFactory.getDefaultSettings()
                    settings.detectorSettings['RADIUS'] = {radius}
                    settings.detectorSettings['THRESHOLD'] = {threshold}
                    settings.detectorSettings['DO_SUBPIXEL_LOCALIZATION'] = true
                    settings.trackerFactory = new SparseLAPTrackerFactory()
                    settings.trackerSettings = settings.trackerFactory.getDefaultSettings()
                    settings.trackerSettings['LINKING_MAX_DISTANCE'] = 0.0
                    settings.trackerSettings['MAX_FRAME_GAP'] = 0
                    def model = new Model()
                    def trackmate = new TrackMate(model, settings)
                    if (!trackmate.checkInput() || !trackmate.process()) {{
                        throw new RuntimeException(trackmate.getErrorMessage())
                    }}
                    CSVExporter.exportSpots(new File("{csv_str}"), model, false)
                    imp.close()
                }}
                """
            ).strip()
        )

    content = textwrap.dedent(
        """
        import fiji.plugin.trackmate.Model
        import fiji.plugin.trackmate.Settings
        import fiji.plugin.trackmate.TrackMate
        import fiji.plugin.trackmate.detection.LogDetectorFactory
        import fiji.plugin.trackmate.tracking.jaqaman.SparseLAPTrackerFactory
        import fiji.plugin.trackmate.io.CSVExporter
        """
    ).strip()
    content += "\n\n" + "\n".join(blocks)
    macro_path.write_text(content, encoding="utf-8")
    return macro_path
