"""Shared seed list helpers for synthetic benchmark generation."""

from __future__ import annotations


def generate_seed_list(num_seeds: int, *, base: int = 101, step: int = 101) -> list[int]:
    """Return deterministic seed list: base, base+step, base+2*step, ..."""
    if num_seeds <= 0:
        return []
    return [base + index * step for index in range(num_seeds)]


DEFAULT_BENCHMARK_SEEDS = generate_seed_list(3)
EXPANDED_BENCHMARK_SEEDS = generate_seed_list(20)
