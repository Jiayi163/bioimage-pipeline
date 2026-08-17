#!/usr/bin/env python3
"""Quick verification script for Phase 1A implementation."""

from __future__ import annotations

import sys

def test_config_parameter():
    """Verify gmm_peak_combination_max parameter exists."""
    from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
    
    config = PunctaDeclumpConfig()
    assert hasattr(config, "gmm_peak_combination_max")
    assert config.gmm_peak_combination_max == 6
    print("✓ Config parameter exists with default value 6")
    
    # Test validation
    try:
        PunctaDeclumpConfig(gmm_peak_combination_max=-1)
        print("✗ Validation failed - negative values should be rejected")
        return False
    except ValueError as e:
        print("✓ Validation works - negative values rejected")
    
    return True


def test_rank_peak_pairs_exists():
    """Verify _rank_peak_pairs function exists and has correct signature."""
    from bioimage_pipeline.puncta.gmm_multi_start import _rank_peak_pairs
    import inspect
    
    sig = inspect.signature(_rank_peak_pairs)
    params = list(sig.parameters.keys())
    assert "peaks" in params
    assert "n_components" in params
    assert "min_separation" in params
    print("✓ _rank_peak_pairs function exists with correct signature")
    return True


def test_peak_pair_generation():
    """Verify peak-pair strategies are generated."""
    from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
    from bioimage_pipeline.puncta.gmm_multi_start import generate_two_component_init_sets
    from bioimage_pipeline.puncta.types import PeakCandidate, ObjectInfo
    from bioimage_pipeline.puncta.background import build_object_patch
    import numpy as np
    
    # Create simple test patch
    image = np.full((48, 48), 40.0, dtype=np.float64)
    mask = np.ones((48, 48), dtype=bool)
    obj = ObjectInfo(
        label=1,
        area=float(mask.sum()),
        equivalent_diameter=12.0,
        bbox=(0, 0, 48, 48),
        centroid=(24.0, 24.0),
        brightest_row=24.0,
        brightest_col=24.0,
        brightest_intensity=float(image.max()),
        major_axis_length=10.0,
        minor_axis_length=10.0,
        elongation=1.0,
    )
    
    config = PunctaDeclumpConfig(gmm_peak_combination_max=5)
    patch = build_object_patch(image, mask, obj, config)
    
    # Create test peaks with sufficient separation
    peaks = [
        PeakCandidate(row=20.0, col=20.0, intensity=1500.0),
        PeakCandidate(row=20.0, col=28.0, intensity=1400.0),  # 8px separation
        PeakCandidate(row=28.0, col=20.0, intensity=1300.0),  # 8px separation from first
    ]
    
    strategies = generate_two_component_init_sets(
        peaks, patch, obj, config=config, single_component=None
    )
    
    peak_pair_strategies = [k for k in strategies.keys() if k.startswith("peak_pair_")]
    if len(peak_pair_strategies) > 0:
        print(f"✓ Peak-pair strategies generated: {peak_pair_strategies}")
        return True
    else:
        print("✗ No peak-pair strategies generated")
        return False


def test_ordering():
    """Verify peak-pair strategies are ordered correctly."""
    from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
    from bioimage_pipeline.puncta.gmm_multi_start import ordered_multi_start_strategies
    from bioimage_pipeline.puncta.types import PeakCandidate
    
    # Simulate strategies dict
    init_sets = {
        "detector_based": [],
        "peak_pair_0_1": [],
        "peak_pair_0_2": [],
        "residual_peak": [],
        "symmetric_x_sep2": [],
        "offset_x_sep2": [],
    }
    
    config = PunctaDeclumpConfig()
    ordered = ordered_multi_start_strategies(init_sets, config=config)
    
    # Verify ordering
    detector_idx = ordered.index("detector_based")
    peak_pair_indices = [i for i, name in enumerate(ordered) if name.startswith("peak_pair_")]
    residual_idx = ordered.index("residual_peak") if "residual_peak" in ordered else -1
    symmetric_indices = [i for i, name in enumerate(ordered) if name.startswith("symmetric_")]
    
    print(f"  Ordered strategies: {ordered}")
    
    # Check ordering constraints
    if not all(pp_idx > detector_idx for pp_idx in peak_pair_indices):
        print("✗ Peak-pairs should come after detector_based")
        return False
    
    if residual_idx >= 0 and not all(pp_idx < residual_idx for pp_idx in peak_pair_indices):
        print("✗ Peak-pairs should come before residual_peak")
        return False
    
    if symmetric_indices and not all(pp_idx < min(symmetric_indices) for pp_idx in peak_pair_indices):
        print("✗ Peak-pairs should come before symmetric strategies")
        return False
    
    print("✓ Peak-pair strategies are ordered correctly")
    return True


def main():
    """Run all verification tests."""
    print("Phase 1A Implementation Verification")
    print("=" * 50)
    
    tests = [
        ("Config parameter", test_config_parameter),
        ("Function signature", test_rank_peak_pairs_exists),
        ("Peak-pair generation", test_peak_pair_generation),
        ("Strategy ordering", test_ordering),
    ]
    
    results = []
    for name, test_fn in tests:
        print(f"\nTest: {name}")
        try:
            result = test_fn()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Verification: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ Phase 1A implementation verified!")
        print("\nNext steps:")
        print("  1. Run full test suite: pytest tests/test_phase1a_peak_combination_init.py -v")
        print("  2. Run existing GMM tests: pytest tests/test_gmm_multi_start.py -v")
        print("  3. Run benchmarks to measure performance")
        return 0
    else:
        print("\n✗ Some verification tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
