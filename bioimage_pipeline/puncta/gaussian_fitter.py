"""2D circular Gaussian fitting for punctum localization."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.types import GaussianFitResult, PeakCandidate

FWHM_FACTOR = 2.355


def _gaussian_model(
    coords: np.ndarray,
    background: float,
    amplitude: float,
    row_center: float,
    col_center: float,
    sigma: float,
) -> np.ndarray:
    rows, cols = coords
    distance_sq = (rows - row_center) ** 2 + (cols - col_center) ** 2
    return background + amplitude * np.exp(-distance_sq / (2.0 * sigma**2))


class GaussianFitter2D:
    """Fit a circular 2D Gaussian to raw intensity around a candidate peak."""

    def __init__(self, config: PunctaDeclumpConfig) -> None:
        self.config = config

    def fit(self, image: np.ndarray, peak: PeakCandidate) -> GaussianFitResult:
        """Fit a Gaussian centered near the candidate peak."""
        image_arr = np.asarray(image, dtype=np.float64)
        center_row = int(round(peak.row))
        center_col = int(round(peak.col))
        radius = self._effective_radius(image_arr.shape, center_row, center_col)

        min_row = center_row - radius
        max_row = center_row + radius + 1
        min_col = center_col - radius
        max_col = center_col + radius + 1

        roi = image_arr[min_row:max_row, min_col:max_col]
        rows, cols = np.mgrid[min_row:max_row, min_col:max_col]
        coords = np.vstack((rows.ravel(), cols.ravel()))

        roi_max = float(np.max(roi))
        local_median = float(np.median(roi))
        amplitude_guess = max(float(peak.intensity) - local_median, 1.0)
        sigma_guess = max(self.config.expected_single_spot_diameter / FWHM_FACTOR, self.config.min_sigma)

        initial = (
            local_median,
            amplitude_guess,
            peak.row,
            peak.col,
            sigma_guess,
        )
        lower = (
            0.0,
            0.0,
            peak.row - self.config.max_center_shift,
            peak.col - self.config.max_center_shift,
            self.config.min_sigma,
        )
        upper = (
            roi_max,
            max(roi_max, amplitude_guess * 2.0),
            peak.row + self.config.max_center_shift,
            peak.col + self.config.max_center_shift,
            self.config.max_sigma,
        )

        try:
            params, _ = curve_fit(
                _gaussian_model,
                coords,
                roi.ravel(),
                p0=initial,
                bounds=(lower, upper),
                maxfev=5000,
            )
            background, amplitude, fitted_row, fitted_col, sigma = params
            predicted = _gaussian_model(coords, background, amplitude, fitted_row, fitted_col, sigma)
            residual_rmse = float(np.sqrt(np.mean((roi.ravel() - predicted) ** 2)))

            return GaussianFitResult(
                fitted_row=float(fitted_row),
                fitted_col=float(fitted_col),
                sigma=float(sigma),
                width_fwhm=float(sigma * FWHM_FACTOR),
                amplitude=float(amplitude),
                background=float(background),
                residual_rmse=residual_rmse,
                roi_touches_edge=False,
                fit_succeeded=True,
            )
        except Exception as exc:
            return GaussianFitResult(
                fitted_row=peak.row,
                fitted_col=peak.col,
                sigma=float("nan"),
                width_fwhm=float("nan"),
                amplitude=float("nan"),
                background=float("nan"),
                residual_rmse=float("inf"),
                roi_touches_edge=False,
                fit_succeeded=False,
                fit_error=str(exc),
            )

    def _effective_radius(
        self,
        shape: tuple[int, ...],
        center_row: int,
        center_col: int,
    ) -> int:
        """Use the largest fitting radius that stays inside the image bounds."""
        height, width = shape[:2]
        max_radius = self.config.fit_roi_radius
        for radius in range(max_radius, 0, -1):
            if (
                center_row - radius >= 0
                and center_col - radius >= 0
                and center_row + radius < height
                and center_col + radius < width
            ):
                return radius
        return 1
