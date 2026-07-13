"""Statistical helpers for synthetic benchmark reporting."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd


def wilson_score_interval(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float | None, float | None]:
    """Return Wilson score interval for a binomial proportion."""
    if n <= 0:
        return None, None
    if successes < 0 or successes > n:
        raise ValueError("successes must be in [0, n]")
    z = 1.959963984540054 if confidence >= 0.95 else 1.644853626951472
    p_hat = successes / n
    denom = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2.0 * n)) / denom
    margin = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) + z**2 / (4.0 * n)) / n)
        / denom
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize_binary_outcomes(
    outcomes: Sequence[bool],
    *,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    """Summarize pass/fail style outcomes with Wilson CI."""
    n = len(outcomes)
    successes = sum(1 for value in outcomes if value)
    rate = successes / n if n else None
    lo, hi = wilson_score_interval(successes, n, confidence=confidence) if n else (None, None)
    return {
        "completed_runs": n,
        "successes": successes,
        "rate": rate,
        "ci_low": lo,
        "ci_high": hi,
    }


def aggregate_benchmark_group(
    frame: pd.DataFrame,
    group_cols: list[str],
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Aggregate per-run benchmark metrics with Wilson CI for pass rates."""
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys, strict=True))
        exact = summarize_binary_outcomes(group["exact_count_correct"].astype(bool), confidence=confidence)
        passed = summarize_binary_outcomes(group["pass_criterion"].astype(bool), confidence=confidence)
        under = summarize_binary_outcomes(group["under_split"].astype(bool), confidence=confidence)
        over = summarize_binary_outcomes(group["over_split"].astype(bool), confidence=confidence)
        rows.append(
            {
                **key_map,
                "completed_runs": int(len(group)),
                "exact_count_pass_rate": exact["rate"],
                "exact_count_ci_low": exact["ci_low"],
                "exact_count_ci_high": exact["ci_high"],
                "full_pass_rate": passed["rate"],
                "full_pass_ci_low": passed["ci_low"],
                "full_pass_ci_high": passed["ci_high"],
                "mean_predicted_count": float(group["predicted_accepted_count"].mean()),
                "under_split_rate": under["rate"],
                "over_split_rate": over["rate"],
                "mean_localization_error_px": _safe_mean(group.get("mean_localization_error_px")),
                "median_localization_error_px": _safe_median(group.get("median_localization_error_px")),
                "mean_runtime_s": _safe_mean(group.get("runtime_s")),
            }
        )
    return pd.DataFrame(rows)


def _safe_mean(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _safe_median(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())
