"""Preview path helpers for threshold parameter assistant comparison views."""

from __future__ import annotations

from pathlib import Path


def find_variant_qc_previews(
    variant_dir: str | Path,
    *,
    image_stem: str | None = None,
) -> list[Path]:
    """Return QC overlay PNG paths for one variant directory."""
    qc_dir = Path(variant_dir) / "qc"
    if not qc_dir.is_dir():
        return []

    patterns = ("*_qc_mask_overlay.png", "*_qc_label_overlay.png")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(qc_dir.glob(pattern)))

    if image_stem is None:
        return paths

    filtered = [
        path
        for path in paths
        if path.name.startswith(f"{image_stem}_") or image_stem in path.stem
    ]
    return filtered or paths


def resolve_compare_preview_path(
    variant_dir: str | Path,
    image_name: str,
    *,
    prefer: str = "mask",
) -> Path | None:
    """Pick one QC overlay for a variant and subset image name."""
    image_stem = Path(image_name).stem
    previews = find_variant_qc_previews(variant_dir, image_stem=image_stem)
    if not previews:
        return None

    if prefer == "label":
        for path in previews:
            if "label_overlay" in path.name:
                return path
    for path in previews:
        if "mask_overlay" in path.name:
            return path
    return previews[0]
