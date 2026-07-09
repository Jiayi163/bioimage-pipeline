"""Goodness-of-fit metrics for Gaussian models."""

from __future__ import annotations

import numpy as np


def compute_r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Return coefficient of determination for flattened arrays."""
    obs = np.asarray(observed, dtype=np.float64).ravel()
    pred = np.asarray(predicted, dtype=np.float64).ravel()
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    if ss_tot <= 0:
        return 1.0 if ss_res <= 0 else 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def compute_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    obs = np.asarray(observed, dtype=np.float64).ravel()
    pred = np.asarray(predicted, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def compute_aic_bic(
    rss: float,
    n_obs: int,
    n_params: int,
) -> tuple[float, float]:
    """Return (AIC, BIC) for Gaussian noise model."""
    if n_obs <= 0:
        return float("inf"), float("inf")
    rss = max(rss, 1e-12)
    log_rss = float(np.log(rss / n_obs))
    aic = 2.0 * n_params + n_obs * log_rss
    bic = n_params * float(np.log(n_obs)) + n_obs * log_rss
    return aic, bic
