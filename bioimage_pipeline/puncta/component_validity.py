"""Phase D component validity: local mixture support, mask tolerance, saturation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.fit_metrics import compute_r_squared, compute_rmse
from bioimage_pipeline.puncta.gaussian_fitter import _elliptical_component
from bioimage_pipeline.puncta.types import GaussianComponent, MixtureFitResult, ObjectPatch


@dataclass(frozen=True)
class SaturationInfo:
    """Saturation/clipping evidence within one object ROI."""

    present: bool
    near_clip_fraction: float
    clip_value: float


@dataclass(frozen=True)
class ComponentValidityMetrics:
    """Local evidence gathered for one mixture component."""

    local_rmse: float
    local_residual_relative: float
    local_r_squared: float
    local_support_fraction: float
    mask_support_fraction: float
    saturation_fraction: float
    saturation_present: bool
    min_neighbor_separation: float | None
    prominence_ratio: float


def detect_roi_saturation(
    patch: ObjectPatch,
    *,
    near_clip_margin: float,
    near_clip_fraction_threshold: float,
) -> SaturationInfo:
    """Detect clipped or near-clipped pixels inside the object mask."""
    if patch.raw is not None:
        values = np.asarray(patch.raw, dtype=np.float64)
    else:
        values = np.asarray(patch.corrected, dtype=np.float64) + patch.background_level

    masked = values[patch.object_mask]
    if masked.size == 0:
        return SaturationInfo(present=False, near_clip_fraction=0.0, clip_value=0.0)

    clip_value = float(masked.max())
    if clip_value <= 0:
        return SaturationInfo(present=False, near_clip_fraction=0.0, clip_value=clip_value)

    near_threshold = clip_value * (1.0 - near_clip_margin)
    near_clip_fraction = float((masked >= near_threshold).sum()) / float(masked.size)
    present = near_clip_fraction >= near_clip_fraction_threshold
    return SaturationInfo(
        present=present,
        near_clip_fraction=near_clip_fraction,
        clip_value=clip_value,
    )


def _patch_center(component: GaussianComponent, patch: ObjectPatch) -> tuple[float, float]:
    return (
        component.fitted_row - patch.row_offset,
        component.fitted_col - patch.col_offset,
    )


def _window_slices(
    center_row: float,
    center_col: float,
    radius: int,
    shape: tuple[int, int],
) -> tuple[slice, slice]:
    row = int(round(center_row))
    col = int(round(center_col))
    min_row = max(0, row - radius)
    max_row = min(shape[0], row + radius + 1)
    min_col = max(0, col - radius)
    max_col = min(shape[1], col + radius + 1)
    return slice(min_row, max_row), slice(min_col, max_col)


def _render_component_patch(component: GaussianComponent, patch: ObjectPatch) -> np.ndarray:
    rows, cols = np.indices(patch.corrected.shape)
    patch_row, patch_col = _patch_center(component, patch)
    return _elliptical_component(
        rows.astype(np.float64),
        cols.astype(np.float64),
        component.amplitude,
        patch_row,
        patch_col,
        component.sigma_row,
        component.sigma_col,
    )


def _saturation_exclude_mask(
    patch: ObjectPatch,
    saturation: SaturationInfo,
    near_clip_margin: float,
) -> np.ndarray:
    if patch.raw is not None:
        values = np.asarray(patch.raw, dtype=np.float64)
    else:
        values = np.asarray(patch.corrected, dtype=np.float64) + patch.background_level

    near_threshold = saturation.clip_value * (1.0 - near_clip_margin)
    return values >= near_threshold


def compute_component_validity_metrics(
    component: GaussianComponent,
    mixture: MixtureFitResult,
    patch: ObjectPatch,
    object_mask: np.ndarray,
    config: PunctaDeclumpConfig,
) -> ComponentValidityMetrics:
    """Compute local validity metrics for one mixture component."""
    patch_row, patch_col = _patch_center(component, patch)
    local_radius = max(1, int(round(component.sigma * config.component_local_residual_radius_sigma)))
    row_slice, col_slice = _window_slices(patch_row, patch_col, local_radius, patch.corrected.shape)

    saturation = detect_roi_saturation(
        patch,
        near_clip_margin=config.saturation_near_clip_margin,
        near_clip_fraction_threshold=config.saturation_near_clip_fraction,
    )

    mask_region = object_mask[row_slice, col_slice]
    observed_region = patch.corrected[row_slice, col_slice]
    if mixture.predicted_patch is not None:
        predicted_region = np.asarray(mixture.predicted_patch, dtype=np.float64)[row_slice, col_slice]
    else:
        predicted_region = np.zeros_like(observed_region)

    valid = mask_region
    if config.saturation_exclude_from_local_residual and saturation.present:
        exclude = _saturation_exclude_mask(patch, saturation, config.saturation_near_clip_margin)
        valid = valid & ~exclude[row_slice, col_slice]

    observed = observed_region[valid]
    predicted = predicted_region[valid]
    if observed.size == 0 and saturation.present:
        valid = mask_region
        observed = observed_region[valid]
        predicted = predicted_region[valid]

    if observed.size == 0:
        local_rmse = float("inf")
        local_r2 = 0.0
    else:
        local_rmse = compute_rmse(observed, predicted)
        local_r2 = compute_r_squared(observed, predicted)

    component_pred = _render_component_patch(component, patch)[row_slice, col_slice]
    if component_pred.size == 0:
        support_fraction = 0.0
        prominence_ratio = 0.0
    else:
        support_threshold = max(float(component_pred.max()) * 0.2, config.min_amplitude * 0.1)
        support_fraction = float((component_pred > support_threshold).sum()) / float(component_pred.size)
        local_peak = float(observed_region[mask_region].max()) if mask_region.any() else 0.0
        local_median = float(np.median(observed_region[mask_region])) if mask_region.any() else 0.0
        prominence_ratio = local_peak / max(local_median, 1.0)

    mask_radius = max(1, int(round(component.sigma * config.component_mask_support_radius_sigma)))
    mask_row_slice, mask_col_slice = _window_slices(
        patch_row,
        patch_col,
        mask_radius,
        object_mask.shape,
    )
    mask_region_tol = object_mask[mask_row_slice, mask_col_slice]
    mask_support_fraction = (
        float(mask_region_tol.sum()) / float(mask_region_tol.size) if mask_region_tol.size else 0.0
    )

    min_neighbor: float | None = None
    for sibling in mixture.components:
        if sibling.component_id == component.component_id:
            continue
        distance = math.hypot(
            component.fitted_row - sibling.fitted_row,
            component.fitted_col - sibling.fitted_col,
        )
        if min_neighbor is None or distance < min_neighbor:
            min_neighbor = distance

    amplitude = max(component.amplitude, 1.0)
    return ComponentValidityMetrics(
        local_rmse=local_rmse,
        local_residual_relative=local_rmse / amplitude,
        local_r_squared=local_r2,
        local_support_fraction=support_fraction,
        mask_support_fraction=mask_support_fraction,
        saturation_fraction=saturation.near_clip_fraction,
        saturation_present=saturation.present,
        min_neighbor_separation=min_neighbor,
        prominence_ratio=prominence_ratio,
    )


def center_has_mask_support(
    component: GaussianComponent,
    patch: ObjectPatch,
    object_mask: np.ndarray,
    config: PunctaDeclumpConfig,
    *,
    metrics: ComponentValidityMetrics | None = None,
) -> bool:
    """Return True when the fitted center has direct or tolerated mask support."""
    patch_row, patch_col = _patch_center(component, patch)
    row = int(round(patch_row))
    col = int(round(patch_col))
    if 0 <= row < object_mask.shape[0] and 0 <= col < object_mask.shape[1] and object_mask[row, col]:
        return True

    if metrics is None:
        metrics = ComponentValidityMetrics(
            local_rmse=float("inf"),
            local_residual_relative=float("inf"),
            local_r_squared=0.0,
            local_support_fraction=0.0,
            mask_support_fraction=0.0,
            saturation_fraction=0.0,
            saturation_present=False,
            min_neighbor_separation=None,
            prominence_ratio=0.0,
        )
        mask_radius = max(1, int(round(component.sigma * config.component_mask_support_radius_sigma)))
        mask_row_slice, mask_col_slice = _window_slices(
            patch_row,
            patch_col,
            mask_radius,
            object_mask.shape,
        )
        mask_region_tol = object_mask[mask_row_slice, mask_col_slice]
        metrics = ComponentValidityMetrics(
            local_rmse=float("inf"),
            local_residual_relative=float("inf"),
            local_r_squared=0.0,
            local_support_fraction=0.0,
            mask_support_fraction=(
                float(mask_region_tol.sum()) / float(mask_region_tol.size)
                if mask_region_tol.size
                else 0.0
            ),
            saturation_fraction=0.0,
            saturation_present=False,
            min_neighbor_separation=None,
            prominence_ratio=0.0,
        )

    if metrics.mask_support_fraction < config.component_min_mask_support_fraction:
        return False

    local_row_slice, local_col_slice = _window_slices(
        patch_row,
        patch_col,
        max(1, int(round(component.sigma * config.component_local_residual_radius_sigma))),
        patch.corrected.shape,
    )
    local_region = patch.corrected[local_row_slice, local_col_slice]
    local_mask = object_mask[local_row_slice, local_col_slice]
    if not local_mask.any():
        return False
    local_peak = float(local_region[local_mask].max())
    return local_peak >= config.min_amplitude * 0.5


def evaluate_component_validity(
    component: GaussianComponent,
    mixture: MixtureFitResult,
    patch: ObjectPatch,
    object_mask: np.ndarray,
    config: PunctaDeclumpConfig,
) -> tuple[str | None, ComponentValidityMetrics]:
    """Return rejection reason (or None) and local validity metrics for one mixture component."""
    metrics = compute_component_validity_metrics(component, mixture, patch, object_mask, config)

    if metrics.local_residual_relative > config.max_fit_residual_relative:
        if math.isinf(metrics.local_residual_relative) and metrics.saturation_present:
            return "insufficient_local_evidence", metrics
        return "residual_too_high", metrics
    if metrics.local_r_squared < config.component_min_local_r_squared:
        return "r_squared_too_low", metrics
    if metrics.local_support_fraction < config.component_local_min_support_fraction:
        return "insufficient_local_support", metrics
    if not center_has_mask_support(component, patch, object_mask, config, metrics=metrics):
        return "center_outside_object_mask", metrics
    if metrics.min_neighbor_separation is not None:
        min_sep = config.gmm_min_component_separation * 0.85
        if metrics.min_neighbor_separation < min_sep:
            return "not_resolvable", metrics
    return None, metrics
