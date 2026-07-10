"""Batch puncta declumping over folders of TIFF images."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bioimage_pipeline.io import read_tiff
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.export import ResultExporter
from bioimage_pipeline.puncta.pipeline import run_puncta_declump
from bioimage_pipeline.puncta.types import DeclumpResult
from bioimage_pipeline.puncta.ui import load_grayscale_plane, load_mask_plane


@dataclass
class PunctaBatchResult:
    processed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    results: dict[str, DeclumpResult] = field(default_factory=dict)


def _collect_tiff_paths(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    return sorted(set(paths))


def run_puncta_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    config: PunctaDeclumpConfig | None = None,
    *,
    frame_index: int = 0,
    mask_dir: str | Path | None = None,
) -> PunctaBatchResult:
    """Process every TIFF in input_dir; write outputs under output_dir/<stem>/."""
    cfg = config or PunctaDeclumpConfig()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    mask_root = Path(mask_dir) if mask_dir is not None else None

    batch_result = PunctaBatchResult()
    image_paths = _collect_tiff_paths(input_path)

    for image_file in image_paths:
        stem = image_file.stem
        image_out = output_path / stem
        image_out.mkdir(parents=True, exist_ok=True)
        try:
            raw_image = read_tiff(image_file)
            image, _ = load_grayscale_plane(
                raw_image,
                frame_index=frame_index,
                source=str(image_file),
            )

            external_mask = None
            if mask_root is not None:
                for suffix in (".tif", ".tiff"):
                    mask_candidate = mask_root / f"{stem}{suffix}"
                    if mask_candidate.is_file():
                        raw_mask = read_tiff(mask_candidate)
                        external_mask, _ = load_mask_plane(
                            raw_mask,
                            frame_index=frame_index,
                            source=str(mask_candidate),
                        )
                        break

            diagnostics_dir: str | None = None
            if cfg.diagnostic_mode not in ("off", "summary"):
                diagnostics_dir = str(image_out / "diagnostics")

            result = run_puncta_declump(
                image,
                cfg,
                external_mask=external_mask,
                diagnostics_dir=diagnostics_dir,
                source_path=str(image_file),
                output_dir=str(image_out),
                stem=stem,
            )

            exporter = ResultExporter()
            exporter.export_all(
                image_out,
                result,
                stem=stem,
                image_shape=image.shape,
                image=image,
                config=cfg,
            )
            batch_result.processed.append(stem)
            batch_result.results[stem] = result
        except Exception as exc:
            batch_result.failed.append((stem, str(exc)))

    return batch_result
