"""Quick verification that Phase B integration compiles and basic imports work."""

from __future__ import annotations

import sys


def verify_imports():
    """Verify all Phase B modules can be imported."""
    print("Verifying Phase B imports...")
    
    try:
        from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
        print("✓ Config imports")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        from bioimage_pipeline.puncta.residual_split import ResidualSplitConfig
        print("✓ ResidualSplitConfig imports")
    except Exception as e:
        print(f"✗ ResidualSplitConfig import failed: {e}")
        return False
    
    try:
        from bioimage_pipeline.puncta.residual_refiner import ResidualSplitRefiner
        print("✓ ResidualSplitRefiner imports")
    except Exception as e:
        print(f"✗ ResidualSplitRefiner import failed: {e}")
        return False
    
    try:
        from bioimage_pipeline.puncta.gaussian_fitter import GaussianModelSelector
        print("✓ GaussianModelSelector imports (with Phase B integration)")
    except Exception as e:
        print(f"✗ GaussianModelSelector import failed: {e}")
        return False
    
    return True


def verify_config():
    """Verify Phase B config parameters exist."""
    print("\nVerifying Phase B config parameters...")
    
    try:
        from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
        
        config = PunctaDeclumpConfig()
        
        assert hasattr(config, "residual_split_enabled")
        assert config.residual_split_enabled is False, "Default should be False"
        print("✓ residual_split_enabled exists (default=False)")
        
        assert hasattr(config, "residual_split_max_iterations")
        assert config.residual_split_max_iterations == 2
        print("✓ residual_split_max_iterations exists (default=2)")
        
        # Test enabled config
        enabled_config = PunctaDeclumpConfig(residual_split_enabled=True)
        assert enabled_config.residual_split_enabled is True
        print("✓ Can enable Phase B via config")
        
    except Exception as e:
        print(f"✗ Config verification failed: {e}")
        return False
    
    return True


def verify_integration():
    """Verify Phase B integration point exists."""
    print("\nVerifying Phase B integration...")
    
    try:
        from bioimage_pipeline.puncta.gaussian_fitter import GaussianModelSelector
        from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
        
        config = PunctaDeclumpConfig()
        selector = GaussianModelSelector(config)
        
        assert hasattr(selector, "_residual_refiner")
        print("✓ GaussianModelSelector has _residual_refiner field")
        
        assert hasattr(selector, "_apply_residual_refinement")
        print("✓ GaussianModelSelector has _apply_residual_refinement method")
        
    except Exception as e:
        print(f"✗ Integration verification failed: {e}")
        return False
    
    return True


def main():
    print("=" * 60)
    print("Phase B Integration Verification")
    print("=" * 60)
    
    all_ok = True
    
    if not verify_imports():
        all_ok = False
    
    if not verify_config():
        all_ok = False
    
    if not verify_integration():
        all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All Phase B verification checks passed!")
        print("\nNext steps:")
        print("1. Run integration tests: pytest tests/test_phase_b_integration.py -v")
        print("2. Run regression tests: pytest tests/test_gmm_multi_start.py -v")
        print("3. Run Phase B spec tests: pytest tests/test_phase_b_residual_guided_split.py -v")
        print("4. Enable Phase B and run benchmarks")
        return 0
    else:
        print("✗ Some verification checks failed")
        print("Review errors above and fix before testing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
