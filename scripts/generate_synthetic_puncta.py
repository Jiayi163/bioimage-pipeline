#!/usr/bin/env python3
"""Generate synthetic puncta images with known ground truth for pipeline validation.

Standalone utility — does not import from bioimage_pipeline.puncta.

Usage (from project root):
    python scripts/generate_synthetic_puncta.py
    python scripts/generate_synthetic_puncta.py --case case3_overlapping
    python scripts/generate_synthetic_puncta.py --batch --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tifffile
from scipy import ndimage

GENERATOR_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpotSpec:
    """One synthetic punctum. x = column, y = row (matches pipeline CSV)."""

    x: float
    y: float
    amplitude: float
    sigma_x: float
    sigma_y: float | None = None

    @property
    def effective_sigma_y(self) -> float:
        return self.sigma_x if self.sigma_y is None else self.sigma_y


@dataclass
class BackgroundGradient:
    """Linear background gradient added on top of constant background."""

    row_slope: float = 0.0
    col_slope: float = 0.0


@dataclass
class GenerationConfig:
    """Parameters controlling image synthesis."""

    height: int = 128
    width: int = 128
    background: float = 40.0
    gradient: BackgroundGradient | None = None
    poisson_noise: bool = True
    read_noise_sigma: float = 5.0
    extra_blur_sigma: float = 0.0
    random_seed: int = 42
    mask_threshold_fraction: float = 0.12
    mask_closing_radius: int = 2
    merge_mask_to_single_object: bool = False


@dataclass
class CaseSpec:
    """One validation case definition."""

    name: str
    spots: list[SpotSpec]
    config: GenerationConfig = field(default_factory=GenerationConfig)
    expected_count: int | None = None
    notes: str = ""
    merge_mask: bool = False


NoiseLevel = Literal["low", "medium", "high"]

NOISE_PRESETS: dict[NoiseLevel, dict[str, float]] = {
    "low": {"read_noise_sigma": 2.0, "extra_blur_sigma": 0.0},
    "medium": {"read_noise_sigma": 5.0, "extra_blur_sigma": 0.0},
    "high": {"read_noise_sigma": 10.0, "extra_blur_sigma": 0.5},
}

BRIGHTNESS_RATIOS: list[tuple[float, float]] = [
    (1.0, 1.0),
    (1.0, 1.5),
    (1.0, 2.0),
]

# ---------------------------------------------------------------------------
# Built-in validation cases
# ---------------------------------------------------------------------------

BASIC_CASES: dict[str, CaseSpec] = {
    "case1_isolated": CaseSpec(
        name="case1_isolated",
        spots=[SpotSpec(x=64.0, y=64.0, amplitude=1800.0, sigma_x=2.2, sigma_y=2.2)],
        expected_count=1,
        notes="Single isolated punctum; predicted count should be 1.",
    ),
    "case2_separated": CaseSpec(
        name="case2_separated",
        spots=[
            SpotSpec(x=61.0, y=64.0, amplitude=1800.0, sigma_x=2.2, sigma_y=2.2),
            SpotSpec(x=67.0, y=64.0, amplitude=1800.0, sigma_x=2.2, sigma_y=2.2),
        ],
        config=GenerationConfig(mask_threshold_fraction=0.18, mask_closing_radius=1),
        expected_count=2,
        notes="Two well-separated puncta (6 px center separation); predicted count should be 2.",
    ),
    "case3_overlapping": CaseSpec(
        name="case3_overlapping",
        spots=[
            SpotSpec(x=63.0, y=64.0, amplitude=1800.0, sigma_x=2.2, sigma_y=2.2),
            SpotSpec(x=65.0, y=64.0, amplitude=1800.0, sigma_x=2.2, sigma_y=2.2),
        ],
        expected_count=2,
        merge_mask=True,
        notes=(
            "Two overlapping puncta (2 px separation) in one merged mask object; "
            "tests GMM declumping. Predicted count should be 2."
        ),
    ),
}

# ---------------------------------------------------------------------------
# Batch grid (scaffold for systematic benchmarking)
# ---------------------------------------------------------------------------

BATCH_GRID: dict[str, list[Any]] = {
    "spot_count": [1, 2, 3],
    "separation_px": [1, 2, 3, 4, 5, 6],
    "sigma": [1.5, 2.0, 2.5, 3.0],
    "brightness_ratio": ["1:1", "1:1.5", "1:2"],
    "noise_level": ["low", "medium", "high"],
    "seeds_per_condition": [101, 202, 303],
}

BATCH_SUBSET_LIMIT = 5


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def build_background(config: GenerationConfig) -> np.ndarray:
    """Return constant (+ optional gradient) background as float64."""
    rows = np.arange(config.height, dtype=np.float64)[:, None]
    cols = np.arange(config.width, dtype=np.float64)[None, :]
    background = np.full((config.height, config.width), config.background, dtype=np.float64)
    if config.gradient is not None:
        background += config.gradient.row_slope * rows
        background += config.gradient.col_slope * cols
    return background


def gaussian_spot(
    shape: tuple[int, int],
    spot: SpotSpec,
) -> np.ndarray:
    """Evaluate one 2D Gaussian punctum on a grid (no background)."""
    rows = np.arange(shape[0], dtype=np.float64)[:, None]
    cols = np.arange(shape[1], dtype=np.float64)[None, :]
    sigma_y = spot.effective_sigma_y
    exponent = (
        (cols - spot.x) ** 2 / (2.0 * spot.sigma_x**2)
        + (rows - spot.y) ** 2 / (2.0 * sigma_y**2)
    )
    return spot.amplitude * np.exp(-exponent)


def render_clean_image(
    spots: list[SpotSpec],
    config: GenerationConfig,
) -> np.ndarray:
    """Noise-free image = background + sum of 2D Gaussian puncta."""
    shape = (config.height, config.width)
    image = build_background(config)
    for spot in spots:
        image += gaussian_spot(shape, spot)
    if config.extra_blur_sigma > 0:
        image = ndimage.gaussian_filter(image, sigma=config.extra_blur_sigma)
    return image.astype(np.float64)


def apply_noise(clean: np.ndarray, config: GenerationConfig) -> np.ndarray:
    """Apply Poisson photon noise and Gaussian read noise."""
    rng = np.random.default_rng(config.random_seed)
    photon = np.clip(clean, 0.0, None)
    if config.poisson_noise:
        noisy = rng.poisson(photon).astype(np.float64)
    else:
        noisy = photon.copy()
    if config.read_noise_sigma > 0:
        noisy += rng.normal(0.0, config.read_noise_sigma, size=noisy.shape)
    return noisy.astype(np.float32)


def _structuring_element(radius: int) -> np.ndarray:
    """Square structuring element for binary morphology."""
    size = max(1, 2 * radius + 1)
    return np.ones((size, size), dtype=bool)


def min_pairwise_separation(spots: list[SpotSpec]) -> float:
    """Minimum center-to-center distance across spot pairs."""
    if len(spots) < 2:
        return float("inf")
    min_sep = float("inf")
    for i, a in enumerate(spots):
        for b in spots[i + 1 :]:
            sep = math.hypot(a.x - b.x, a.y - b.y)
            min_sep = min(min_sep, sep)
    return min_sep


def spot_disk_mask(
    shape: tuple[int, int],
    spot: SpotSpec,
    *,
    radius_factor: float = 2.5,
) -> np.ndarray:
    """Circular mask around a spot center sized from its sigma."""
    rows = np.arange(shape[0], dtype=np.float64)[:, None]
    cols = np.arange(shape[1], dtype=np.float64)[None, :]
    radius = radius_factor * max(spot.sigma_x, spot.effective_sigma_y)
    distance_sq = (rows - spot.y) ** 2 + (cols - spot.x) ** 2
    return distance_sq <= radius**2


def build_mask_from_clean(
    clean: np.ndarray,
    config: GenerationConfig,
    *,
    spots: list[SpotSpec] | None = None,
    merge_to_single: bool = False,
) -> np.ndarray:
    """Build binary mask from clean signal (not noisy image).

    Steps: subtract background, threshold, close, fill holes.
    For multiple well-separated spots without merge_to_single, build the mask as
    the union of per-spot threshold regions so each punctum stays a separate object.
    Optionally merge all foreground into one connected object.
    """
    background = build_background(config)
    signal = np.clip(clean - background, 0.0, None)

    if spots and len(spots) > 1 and not merge_to_single:
        min_sep = min_pairwise_separation(spots)
        max_radius = max(1.5, min_sep / 2.0 - 0.25)
        mask = np.zeros(clean.shape, dtype=bool)
        for spot in spots:
            sigma_radius = 2.5 * max(spot.sigma_x, spot.effective_sigma_y)
            spot_mask = spot_disk_mask(
                clean.shape,
                spot,
                radius_factor=min(2.5, max_radius / max(spot.sigma_x, spot.effective_sigma_y)),
            )
            if config.mask_closing_radius > 0:
                spot_mask = ndimage.binary_closing(
                    spot_mask,
                    structure=_structuring_element(max(1, config.mask_closing_radius - 1)),
                )
            spot_mask = ndimage.binary_fill_holes(spot_mask)
            mask |= spot_mask
        return mask.astype(bool)

    peak = float(signal.max())
    if peak <= 0:
        return np.zeros(clean.shape, dtype=bool)

    threshold = config.mask_threshold_fraction * peak
    mask = signal >= threshold

    if config.mask_closing_radius > 0:
        mask = ndimage.binary_closing(mask, structure=_structuring_element(config.mask_closing_radius))
    mask = ndimage.binary_fill_holes(mask)

    if merge_to_single and mask.any():
        labeled, n_components = ndimage.label(mask)
        if n_components > 1:
            # Dilate each component slightly, then union, to bridge overlapping spots.
            dilated = ndimage.binary_dilation(
                mask,
                structure=_structuring_element(max(2, config.mask_closing_radius)),
            )
            mask = ndimage.binary_fill_holes(dilated)

    return mask.astype(bool)


def count_mask_components(mask: np.ndarray) -> int:
    """Count connected foreground components in a binary mask."""
    if not mask.any():
        return 0
    _, count = ndimage.label(mask)
    return int(count)


def build_true_seeds(
    spots: list[SpotSpec],
    shape: tuple[int, int],
) -> np.ndarray:
    """Label image with one seed pixel per true punctum (uint16)."""
    seeds = np.zeros(shape, dtype=np.uint16)
    for index, spot in enumerate(spots, start=1):
        row = int(round(spot.y))
        col = int(round(spot.x))
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            seeds[row, col] = index
    return seeds


def spot_to_dict(spot: SpotSpec, spot_id: int) -> dict[str, Any]:
    return {
        "id": spot_id,
        "x": spot.x,
        "y": spot.y,
        "amplitude": spot.amplitude,
        "sigma_x": spot.sigma_x,
        "sigma_y": spot.effective_sigma_y,
    }


def build_ground_truth(
    case: CaseSpec,
    clean: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Assemble ground-truth JSON payload."""
    config = case.config
    gradient_payload = None
    if config.gradient is not None:
        gradient_payload = asdict(config.gradient)

    return {
        "case_name": case.name,
        "true_spot_count": len(case.spots),
        "image_shape": [config.height, config.width],
        "coordinate_system": {"x": "column", "y": "row"},
        "background": {
            "constant": config.background,
            "gradient": gradient_payload,
        },
        "noise": {
            "poisson": config.poisson_noise,
            "read_noise_sigma": config.read_noise_sigma,
            "extra_blur_sigma": config.extra_blur_sigma,
        },
        "mask": {
            "threshold_fraction": config.mask_threshold_fraction,
            "closing_radius": config.mask_closing_radius,
            "merged_single_object": case.merge_mask,
            "connected_components": count_mask_components(mask),
        },
        "spots": [spot_to_dict(spot, index) for index, spot in enumerate(case.spots, start=1)],
        "expected": {
            "true_count": case.expected_count if case.expected_count is not None else len(case.spots),
            "notes": case.notes,
        },
        "generation": {
            "random_seed": config.random_seed,
            "generator_version": GENERATOR_VERSION,
        },
        "signal_stats": {
            "clean_min": float(clean.min()),
            "clean_max": float(clean.max()),
            "clean_mean": float(clean.mean()),
        },
    }


# ---------------------------------------------------------------------------
# Output paths and file writing
# ---------------------------------------------------------------------------


def case_output_paths(output_root: Path, case_name: str) -> dict[str, Path]:
    """Return output paths for one case."""
    return {
        "noisy": output_root / "images" / case_name / "synthetic_noisy.tif",
        "clean": output_root / "images" / case_name / "synthetic_clean.tif",
        "mask": output_root / "masks" / case_name / "synthetic_mask.tif",
        "seeds": output_root / "ground_truth" / case_name / "synthetic_true_seeds.tif",
        "ground_truth": output_root / "ground_truth" / case_name / "synthetic_ground_truth.json",
    }


def ensure_output_dirs(output_root: Path) -> None:
    """Create top-level synthetic_test_data directories."""
    for subdir in ("images", "masks", "ground_truth", "results"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)


def write_case_outputs(
    case: CaseSpec,
    output_root: Path,
) -> dict[str, Path]:
    """Generate and write all five files for one case."""
    ensure_output_dirs(output_root)
    paths = case_output_paths(output_root, case.name)

    clean = render_clean_image(case.spots, case.config)
    noisy = apply_noise(clean, case.config)
    mask = build_mask_from_clean(
        clean,
        case.config,
        spots=case.spots,
        merge_to_single=case.merge_mask,
    )
    seeds = build_true_seeds(case.spots, (case.config.height, case.config.width))
    ground_truth = build_ground_truth(case, clean, mask)

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(paths["clean"], clean.astype(np.float32))
    tifffile.imwrite(paths["noisy"], noisy)
    tifffile.imwrite(paths["mask"], (mask.astype(np.uint8) * 255))
    tifffile.imwrite(paths["seeds"], seeds)
    paths["ground_truth"].write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")

    return paths


# ---------------------------------------------------------------------------
# Batch grid helpers
# ---------------------------------------------------------------------------


def _ratio_to_amplitudes(base: float, ratio_label: str) -> tuple[float, float]:
    mapping = {"1:1": (1.0, 1.0), "1:1.5": (1.0, 1.5), "1:2": (1.0, 2.0)}
    r1, r2 = mapping.get(ratio_label, (1.0, 1.0))
    return base * r1, base * r2


def _spots_for_batch_condition(
    spot_count: int,
    separation_px: float,
    sigma: float,
    brightness_ratio: str,
    center_x: float = 64.0,
    center_y: float = 64.0,
    base_amplitude: float = 1800.0,
) -> list[SpotSpec]:
    """Place spots horizontally around image center."""
    if spot_count == 1:
        return [SpotSpec(x=center_x, y=center_y, amplitude=base_amplitude, sigma_x=sigma, sigma_y=sigma)]

    amp_a, amp_b = _ratio_to_amplitudes(base_amplitude, brightness_ratio)
    if spot_count == 2:
        offsets = [-separation_px / 2.0, separation_px / 2.0]
        amps = [amp_a, amp_b]
    else:
        offsets = [-separation_px, 0.0, separation_px]
        amps = [amp_a, (amp_a + amp_b) / 2.0, amp_b]

    return [
        SpotSpec(
            x=center_x + offset,
            y=center_y,
            amplitude=amps[index],
            sigma_x=sigma,
            sigma_y=sigma,
        )
        for index, offset in enumerate(offsets[:spot_count])
    ]


def iter_batch_conditions(limit: int | None = None) -> list[dict[str, Any]]:
    """Expand BATCH_GRID into a list of condition dicts."""
    conditions: list[dict[str, Any]] = []
    for spot_count in BATCH_GRID["spot_count"]:
        separations = BATCH_GRID["separation_px"] if spot_count >= 2 else [0]
        for separation in separations:
            for sigma in BATCH_GRID["sigma"]:
                ratios = BATCH_GRID["brightness_ratio"] if spot_count >= 2 else ["1:1"]
                for ratio in ratios:
                    for noise_level in BATCH_GRID["noise_level"]:
                        for seed in BATCH_GRID["seeds_per_condition"]:
                            case_id = (
                                f"batch_n{spot_count}_sep{separation}_sig{sigma}_"
                                f"br{ratio.replace(':', '-')}_noise{noise_level}_s{seed}"
                            )
                            conditions.append(
                                {
                                    "case_id": case_id,
                                    "spot_count": spot_count,
                                    "separation_px": separation,
                                    "sigma": sigma,
                                    "brightness_ratio": ratio,
                                    "noise_level": noise_level,
                                    "random_seed": seed,
                                }
                            )
                            if limit is not None and len(conditions) >= limit:
                                return conditions
    return conditions


def condition_to_case(condition: dict[str, Any], base_config: GenerationConfig) -> CaseSpec:
    """Convert one batch condition dict into a CaseSpec."""
    noise = NOISE_PRESETS[condition["noise_level"]]  # type: ignore[index]
    config = GenerationConfig(
        height=base_config.height,
        width=base_config.width,
        background=base_config.background,
        poisson_noise=base_config.poisson_noise,
        read_noise_sigma=noise["read_noise_sigma"],
        extra_blur_sigma=noise["extra_blur_sigma"],
        random_seed=condition["random_seed"],
        mask_threshold_fraction=base_config.mask_threshold_fraction,
        mask_closing_radius=base_config.mask_closing_radius,
    )
    spots = _spots_for_batch_condition(
        spot_count=condition["spot_count"],
        separation_px=condition["separation_px"],
        sigma=condition["sigma"],
        brightness_ratio=condition["brightness_ratio"],
    )
    merge = condition["spot_count"] >= 2 and condition["separation_px"] <= 2
    return CaseSpec(
        name=condition["case_id"],
        spots=spots,
        config=config,
        expected_count=condition["spot_count"],
        merge_mask=merge,
        notes=f"Batch condition: {json.dumps(condition)}",
    )


# ---------------------------------------------------------------------------
# Separation benchmark (2-spot merged mask, varying center separation)
# ---------------------------------------------------------------------------

SEPARATION_BENCHMARK_SEPARATIONS = [1, 2, 3, 4, 5, 6]
SEPARATION_BENCHMARK_SIGMA = 2.2
SEPARATION_BENCHMARK_AMPLITUDE = 1800.0

BRIGHTNESS_RATIO_BENCHMARK_RATIOS = ["1:1", "1:1.5", "1:2", "1:3", "1:5"]
BRIGHTNESS_RATIO_BENCHMARK_SEPARATIONS = [2, 3, 4, 5]
SIGMA_BENCHMARK_SIGMAS = [1.5, 2.0, 2.5, 3.0]
SIGMA_BENCHMARK_SEPARATIONS = [2, 3, 4, 5]

FALSE_SPLIT_SIGMAS = [1.5, 2.0, 2.5, 3.0]
FALSE_SPLIT_NOISE_LEVELS: list[NoiseLevel] = ["low", "medium", "high"]
FALSE_SPLIT_AMPLITUDE_LEVELS = ["low", "medium", "high"]
FALSE_SPLIT_AMP_VALUES = {"low": 600.0, "medium": 1200.0, "high": 1800.0}
FALSE_SPLIT_ELLIPTICITY = ["circular", "elongated"]


def generate_seed_list(num_seeds: int, *, base: int = 101, step: int = 101) -> list[int]:
    """Return deterministic seed list: base, base+step, base+2*step, ..."""
    if num_seeds <= 0:
        return []
    return [base + index * step for index in range(num_seeds)]


SEPARATION_BENCHMARK_SEEDS = generate_seed_list(3)


def separation_benchmark_case_name(separation_px: int, seed: int) -> str:
    return f"sep_benchmark_sep{separation_px}_seed{seed}"


def build_separation_benchmark_case(
    separation_px: int,
    seed: int,
    *,
    base_config: GenerationConfig | None = None,
) -> CaseSpec:
    """Two equal-amplitude spots in one merged mask at a given center separation."""
    config = base_config or GenerationConfig(random_seed=seed)
    config.random_seed = seed
    center_x = config.width / 2.0
    center_y = config.height / 2.0
    half = separation_px / 2.0
    spots = [
        SpotSpec(
            x=center_x - half,
            y=center_y,
            amplitude=SEPARATION_BENCHMARK_AMPLITUDE,
            sigma_x=SEPARATION_BENCHMARK_SIGMA,
            sigma_y=SEPARATION_BENCHMARK_SIGMA,
        ),
        SpotSpec(
            x=center_x + half,
            y=center_y,
            amplitude=SEPARATION_BENCHMARK_AMPLITUDE,
            sigma_x=SEPARATION_BENCHMARK_SIGMA,
            sigma_y=SEPARATION_BENCHMARK_SIGMA,
        ),
    ]
    return CaseSpec(
        name=separation_benchmark_case_name(separation_px, seed),
        spots=spots,
        config=config,
        expected_count=2,
        merge_mask=True,
        notes=(
            f"Separation benchmark: 2 spots, separation={separation_px}px, "
            f"sigma={SEPARATION_BENCHMARK_SIGMA}, seed={seed}, merged mask."
        ),
    )


def generate_separation_benchmark(
    output_root: Path,
    *,
    base_config: GenerationConfig | None = None,
    separations: list[int] | None = None,
    seeds: list[int] | None = None,
) -> list[str]:
    """Generate all separation benchmark cases; return case names."""
    separations = separations or SEPARATION_BENCHMARK_SEPARATIONS
    seeds = seeds or SEPARATION_BENCHMARK_SEEDS
    generated: list[str] = []
    for separation in separations:
        for seed in seeds:
            case = build_separation_benchmark_case(
                separation,
                seed,
                base_config=base_config,
            )
            write_case_outputs(case, output_root)
            generated.append(case.name)
    return generated


def false_split_benchmark_case_name(
    *,
    sigma: float,
    noise_level: str,
    amplitude_level: str,
    gradient_on: bool,
    ellipticity: str,
    seed: int,
) -> str:
    grad = "on" if gradient_on else "off"
    sig_tag = str(sigma).replace(".", "p")
    return (
        f"false_split_sig{sig_tag}_noise{noise_level}_amp{amplitude_level}_"
        f"grad{grad}_ellip{ellipticity}_seed{seed}"
    )


def build_false_split_benchmark_case(
    *,
    sigma: float,
    noise_level: NoiseLevel,
    amplitude_level: str,
    gradient_on: bool,
    ellipticity: str,
    seed: int,
    base_config: GenerationConfig | None = None,
) -> CaseSpec:
    """Single punctum in a merged mask large enough to trigger GMM."""
    noise = NOISE_PRESETS[noise_level]
    config = base_config or GenerationConfig(random_seed=seed)
    config.random_seed = seed
    config.read_noise_sigma = noise["read_noise_sigma"]
    config.extra_blur_sigma = noise["extra_blur_sigma"]
    config.gradient = (
        BackgroundGradient(row_slope=0.08, col_slope=0.05) if gradient_on else None
    )
    amp = FALSE_SPLIT_AMP_VALUES[amplitude_level]
    sigma_y = sigma if ellipticity == "circular" else sigma * 0.68
    spot = SpotSpec(
        x=config.width / 2.0,
        y=config.height / 2.0,
        amplitude=amp,
        sigma_x=sigma,
        sigma_y=sigma_y,
    )
    return CaseSpec(
        name=false_split_benchmark_case_name(
            sigma=sigma,
            noise_level=noise_level,
            amplitude_level=amplitude_level,
            gradient_on=gradient_on,
            ellipticity=ellipticity,
            seed=seed,
        ),
        spots=[spot],
        config=config,
        expected_count=1,
        merge_mask=True,
        notes=(
            f"Single-punctum false-split benchmark: sigma={sigma}, noise={noise_level}, "
            f"amplitude={amplitude_level}, gradient={gradient_on}, ellipticity={ellipticity}, seed={seed}."
        ),
    )


def generate_false_split_benchmark(
    output_root: Path,
    *,
    base_config: GenerationConfig | None = None,
    seeds: list[int] | None = None,
) -> list[str]:
    seeds = seeds or SEPARATION_BENCHMARK_SEEDS
    generated: list[str] = []
    for sigma in FALSE_SPLIT_SIGMAS:
        for noise_level in FALSE_SPLIT_NOISE_LEVELS:
            for amplitude_level in FALSE_SPLIT_AMPLITUDE_LEVELS:
                for gradient_on in (False, True):
                    for ellipticity in FALSE_SPLIT_ELLIPTICITY:
                        for seed in seeds:
                            case = build_false_split_benchmark_case(
                                sigma=sigma,
                                noise_level=noise_level,
                                amplitude_level=amplitude_level,
                                gradient_on=gradient_on,
                                ellipticity=ellipticity,
                                seed=seed,
                                base_config=base_config,
                            )
                            write_case_outputs(case, output_root)
                            generated.append(case.name)
    return generated


def brightness_ratio_benchmark_case_name(ratio: str, separation_px: int, seed: int) -> str:
    br = ratio.replace(":", "-")
    return f"ratio_benchmark_br{br}_sep{separation_px}_seed{seed}"


def build_brightness_ratio_benchmark_case(
    ratio: str,
    separation_px: int,
    seed: int,
    *,
    base_config: GenerationConfig | None = None,
) -> CaseSpec:
    config = base_config or GenerationConfig(random_seed=seed)
    config.random_seed = seed
    center_x = config.width / 2.0
    center_y = config.height / 2.0
    half = separation_px / 2.0
    parts = ratio.split(":")
    amp_a = SEPARATION_BENCHMARK_AMPLITUDE
    amp_b = amp_a * float(parts[1]) / float(parts[0])
    spots = [
        SpotSpec(
            x=center_x - half,
            y=center_y,
            amplitude=amp_a,
            sigma_x=SEPARATION_BENCHMARK_SIGMA,
            sigma_y=SEPARATION_BENCHMARK_SIGMA,
        ),
        SpotSpec(
            x=center_x + half,
            y=center_y,
            amplitude=amp_b,
            sigma_x=SEPARATION_BENCHMARK_SIGMA,
            sigma_y=SEPARATION_BENCHMARK_SIGMA,
        ),
    ]
    return CaseSpec(
        name=brightness_ratio_benchmark_case_name(ratio, separation_px, seed),
        spots=spots,
        config=config,
        expected_count=2,
        merge_mask=True,
        notes=f"Brightness ratio benchmark: ratio={ratio}, separation={separation_px}px, seed={seed}.",
    )


def generate_brightness_ratio_benchmark(
    output_root: Path,
    *,
    base_config: GenerationConfig | None = None,
    seeds: list[int] | None = None,
    separations: list[int] | None = None,
) -> list[str]:
    seeds = seeds or SEPARATION_BENCHMARK_SEEDS
    separations = separations or BRIGHTNESS_RATIO_BENCHMARK_SEPARATIONS
    generated: list[str] = []
    for ratio in BRIGHTNESS_RATIO_BENCHMARK_RATIOS:
        for separation in separations:
            for seed in seeds:
                case = build_brightness_ratio_benchmark_case(
                    ratio,
                    separation,
                    seed,
                    base_config=base_config,
                )
                write_case_outputs(case, output_root)
                generated.append(case.name)
    return generated


def sigma_benchmark_case_name(sigma: float, separation_px: int, seed: int) -> str:
    sig_tag = str(sigma).replace(".", "p")
    return f"sigma_benchmark_sig{sig_tag}_sep{separation_px}_seed{seed}"


def build_sigma_benchmark_case(
    sigma: float,
    separation_px: int,
    seed: int,
    *,
    base_config: GenerationConfig | None = None,
) -> CaseSpec:
    config = base_config or GenerationConfig(random_seed=seed)
    config.random_seed = seed
    center_x = config.width / 2.0
    center_y = config.height / 2.0
    half = separation_px / 2.0
    spots = [
        SpotSpec(
            x=center_x - half,
            y=center_y,
            amplitude=SEPARATION_BENCHMARK_AMPLITUDE,
            sigma_x=sigma,
            sigma_y=sigma,
        ),
        SpotSpec(
            x=center_x + half,
            y=center_y,
            amplitude=SEPARATION_BENCHMARK_AMPLITUDE,
            sigma_x=sigma,
            sigma_y=sigma,
        ),
    ]
    return CaseSpec(
        name=sigma_benchmark_case_name(sigma, separation_px, seed),
        spots=spots,
        config=config,
        expected_count=2,
        merge_mask=True,
        notes=f"Sigma benchmark: sigma={sigma}, separation={separation_px}px, seed={seed}.",
    )


def generate_sigma_benchmark(
    output_root: Path,
    *,
    base_config: GenerationConfig | None = None,
    seeds: list[int] | None = None,
    sigmas: list[float] | None = None,
    separations: list[int] | None = None,
) -> list[str]:
    seeds = seeds or SEPARATION_BENCHMARK_SEEDS
    sigmas = sigmas or SIGMA_BENCHMARK_SIGMAS
    separations = separations or SIGMA_BENCHMARK_SEPARATIONS
    generated: list[str] = []
    for sigma in sigmas:
        for separation in separations:
            for seed in seeds:
                case = build_sigma_benchmark_case(
                    sigma,
                    separation,
                    seed,
                    base_config=base_config,
                )
                write_case_outputs(case, output_root)
                generated.append(case.name)
    return generated


def write_batch_manifest(output_root: Path, conditions: list[dict[str, Any]]) -> Path:
    """Write batch_manifest.json listing planned or generated conditions."""
    manifest_path = output_root / "batch_manifest.json"
    payload = {
        "generator_version": GENERATOR_VERSION,
        "grid_definition": BATCH_GRID,
        "total_conditions": len(iter_batch_conditions(limit=None)),
        "listed_conditions": len(conditions),
        "conditions": conditions,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def load_spots_file(path: Path) -> tuple[list[SpotSpec], dict[str, Any]]:
    """Load spots and optional metadata from a JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_spots = payload.get("spots", payload)
    if not isinstance(raw_spots, list):
        raise ValueError("spots file must contain a 'spots' list or be a list")

    spots: list[SpotSpec] = []
    for entry in raw_spots:
        spots.append(
            SpotSpec(
                x=float(entry["x"]),
                y=float(entry["y"]),
                amplitude=float(entry["amplitude"]),
                sigma_x=float(entry["sigma_x"]),
                sigma_y=float(entry.get("sigma_y")) if entry.get("sigma_y") is not None else None,
            )
        )
    meta = {key: value for key, value in payload.items() if key != "spots"}
    return spots, meta


def apply_cli_overrides(config: GenerationConfig, args: argparse.Namespace) -> GenerationConfig:
    """Return a copy of config with CLI overrides applied.

    Preserves case-specific mask parameters from the input config; global CLI
    flags control image size, background, noise, and random seed.
    """
    return GenerationConfig(
        height=args.height,
        width=args.width,
        background=args.background,
        gradient=config.gradient,
        poisson_noise=not args.no_poisson,
        read_noise_sigma=args.read_noise,
        extra_blur_sigma=args.extra_blur,
        random_seed=args.seed,
        mask_threshold_fraction=config.mask_threshold_fraction,
        mask_closing_radius=config.mask_closing_radius,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic puncta images with known ground truth.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("synthetic_test_data"),
        help="Root directory for generated files (default: synthetic_test_data).",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Generate one or more built-in cases by name (default: all basic cases).",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List built-in case names and exit.",
    )
    parser.add_argument(
        "--spots-file",
        type=Path,
        help="Generate a custom case from a JSON spot list.",
    )
    parser.add_argument(
        "--custom-case-name",
        type=str,
        default="custom_case",
        help="Case folder name when using --spots-file.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Generate batch grid cases (subset in v1) or write manifest with --dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --batch: write batch_manifest.json only, do not generate images.",
    )
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=BATCH_SUBSET_LIMIT,
        help=f"Max batch cases to generate when --batch is set (default: {BATCH_SUBSET_LIMIT}).",
    )
    parser.add_argument(
        "--separation-benchmark",
        action="store_true",
        help="Generate 2-spot merged-mask separation benchmark cases.",
    )
    parser.add_argument(
        "--false-split-benchmark",
        action="store_true",
        help="Generate single-punctum false-split benchmark cases.",
    )
    parser.add_argument(
        "--brightness-ratio-benchmark",
        action="store_true",
        help="Generate doublet brightness-ratio benchmark cases.",
    )
    parser.add_argument(
        "--sigma-benchmark",
        action="store_true",
        help="Generate doublet sigma benchmark cases.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=3,
        help="Number of random seeds per benchmark condition (default: 3).",
    )
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--background", type=float, default=40.0)
    parser.add_argument("--read-noise", type=float, default=5.0, dest="read_noise")
    parser.add_argument("--extra-blur", type=float, default=0.0, dest="extra_blur")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-poisson", action="store_true")
    parser.add_argument("--mask-threshold", type=float, default=0.12, dest="mask_threshold")
    parser.add_argument("--mask-closing-radius", type=int, default=2, dest="mask_closing_radius")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_cases:
        print("Built-in cases:")
        for name, case in BASIC_CASES.items():
            print(f"  {name}: {len(case.spots)} spot(s) — {case.notes}")
        total_batch = len(iter_batch_conditions(limit=None))
        print(f"\nBatch grid total conditions: {total_batch}")
        return

    output_root = args.output_root.resolve()
    base_config = apply_cli_overrides(GenerationConfig(), args)
    seed_list = generate_seed_list(args.num_seeds)

    if args.separation_benchmark:
        names = generate_separation_benchmark(
            output_root,
            base_config=base_config,
            seeds=seed_list,
        )
        print(f"Generated {len(names)} separation benchmark case(s) with {len(seed_list)} seed(s) each.")
        for name in names[:5]:
            print(f"  {name}")
        if len(names) > 5:
            print(f"  ... and {len(names) - 5} more")
        return

    if args.false_split_benchmark:
        names = generate_false_split_benchmark(
            output_root,
            base_config=base_config,
            seeds=seed_list,
        )
        print(f"Generated {len(names)} false-split benchmark case(s).")
        return

    if args.brightness_ratio_benchmark:
        names = generate_brightness_ratio_benchmark(
            output_root,
            base_config=base_config,
            seeds=seed_list,
        )
        print(f"Generated {len(names)} brightness-ratio benchmark case(s).")
        return

    if args.sigma_benchmark:
        names = generate_sigma_benchmark(
            output_root,
            base_config=base_config,
            seeds=seed_list,
        )
        print(f"Generated {len(names)} sigma benchmark case(s).")
        return

    if args.batch:
        all_conditions = iter_batch_conditions(limit=None)
        if args.dry_run:
            manifest = write_batch_manifest(output_root, all_conditions)
            print(f"Wrote batch manifest ({len(all_conditions)} conditions): {manifest}")
            return

        subset = iter_batch_conditions(limit=args.batch_limit)
        write_batch_manifest(output_root, subset)
        for condition in subset:
            case = condition_to_case(condition, base_config)
            paths = write_case_outputs(case, output_root)
            print(f"Generated batch case: {case.name}")
            print(f"  noisy: {paths['noisy']}")
        print(f"Generated {len(subset)} batch case(s) (limit={args.batch_limit}).")
        return

    cases_to_run: list[CaseSpec] = []

    if args.spots_file:
        spots, meta = load_spots_file(args.spots_file)
        merge = bool(meta.get("merge_mask", False))
        cases_to_run.append(
            CaseSpec(
                name=args.custom_case_name,
                spots=spots,
                config=base_config,
                expected_count=len(spots),
                merge_mask=merge,
                notes=str(meta.get("notes", "Custom case from --spots-file")),
            )
        )
    else:
        selected = args.cases or list(BASIC_CASES.keys())
        for name in selected:
            if name not in BASIC_CASES:
                raise SystemExit(f"Unknown case: {name}. Use --list-cases.")
            case = BASIC_CASES[name]
            case.config = apply_cli_overrides(case.config, args)
            cases_to_run.append(case)

    for case in cases_to_run:
        paths = write_case_outputs(case, output_root)
        mask_path = paths["mask"]
        mask = tifffile.imread(mask_path) > 0
        n_components = count_mask_components(mask)
        print(f"Generated: {case.name}")
        print(f"  true spots: {len(case.spots)}")
        print(f"  mask components: {n_components}")
        print(f"  noisy:  {paths['noisy']}")
        print(f"  clean:  {paths['clean']}")
        print(f"  mask:   {paths['mask']}")
        print(f"  seeds:  {paths['seeds']}")
        print(f"  truth:  {paths['ground_truth']}")


if __name__ == "__main__":
    main()
