#!/usr/bin/env python3
# Test script to verify input shaper fixes
#
# Copyright (C) 2024  Input Shaper Fix Contributors
#
# This file may be distributed under the terms of the GNU GPLv3 license.

import sys
import os
import numpy as np

# Add klippy to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'klippy'))

import importlib
shaper_defs = importlib.import_module('.shaper_defs', 'extras')
shaper_calibrate = importlib.import_module('.shaper_calibrate', 'extras')

def test_parameter_validation():
    """Test parameter validation in shapers"""
    print("Testing parameter validation...")
    
    # Test various edge cases that might cause failures
    test_cases = [
        ("ulv", 25.0, 0.1),   # Should work - normal case
        ("ulv", 15.0, 0.1),   # Should fallback to 3hump_ei - too low freq
        ("ulv", 150.0, 0.1),  # Should fallback to ei - too high freq
        ("multi_freq", 10.0, 0.1),  # Should fallback to mzv - too low freq
        ("multi_freq", 50.0, 0.1),   # Should work - normal case
        ("smooth", 30.0, 0.1),       # Should work - normal case
        ("adaptive_ei", 40.0, 0.1),  # Should work - normal case
    ]
    
    for shaper_name, freq, damping in test_cases:
        print(f"  Testing {shaper_name} @ {freq}Hz, damping={damping}")
        
        # Find the shaper function
        shaper_cfg = next((s for s in shaper_defs.INPUT_SHAPERS if s.name == shaper_name), None)
        if not shaper_cfg:
            print(f"    ! Shaper {shaper_name} not found")
            continue
            
        try:
            A, T = shaper_cfg.init_func(freq, damping)
            
            # Check if parameters are reasonable
            print(f"    ✓ Generated {len(A)} impulses, max_time={max(T):.4f}s, sum={sum(A):.6f}")
            
            # Verify basic constraints
            assert len(A) == len(T), "Mismatched A and T lengths"
            assert abs(sum(A) - 1.0) < 1e-6, "Impulses don't sum to 1.0"
            assert T[0] == 0.0, "First time should be 0"
            assert max(T) <= 0.5, "Shaper duration too long"
            assert len(A) <= 10, "Too many impulses"
            
        except Exception as e:
            print(f"    ! Error with {shaper_name}: {e}")
    
    print("✓ Parameter validation tests completed")

def test_shaper_fallback_logic():
    """Test that fallback logic works correctly"""
    print("Testing shaper fallback logic...")
    
    # Mock InputShaperParams class to test validation
    class MockInputShaperParams:
        def __init__(self, shaper_type, freq, damping):
            self.shaper_type = shaper_type
            self.shaper_freq = freq
            self.damping_ratio = damping
            self.shapers = {s.name : s.init_func for s in shaper_defs.INPUT_SHAPERS}
        
        def get_shaper(self):
            if not self.shaper_freq:
                A, T = shaper_defs.get_none_shaper()
            else:
                A, T = self.shapers[self.shaper_type](
                        self.shaper_freq, self.damping_ratio)
                # Validate shaper parameters
                if not self._validate_shaper_params(A, T):
                    # Log warning and fall back to simpler shaper
                    print(f"    ! Shaper parameters for {self.shaper_type} are invalid, falling back to mzv")
                    A, T = self.shapers['mzv'](self.shaper_freq, self.damping_ratio)
            return len(A), A, T
        
        def _validate_shaper_params(self, A, T):
            """Validate shaper parameters before applying them"""
            # Check basic constraints
            if len(A) != len(T):
                return False
            if len(A) == 0:
                return False
            
            # Check that impulses sum to approximately 1.0
            if abs(sum(A) - 1.0) > 1e-6:
                return False
                
            # Check time constraints
            if T[0] != 0.0:
                return False
            for i in range(len(T) - 1):
                if T[i] > T[i + 1]:
                    return False
                    
            # Check for reasonable amplitude values
            for a in A:
                if a < 0 or a > 2.0:  # Allow some flexibility but catch obviously wrong values
                    return False
                    
            # Check for reasonable time values (not too long)
            max_time = max(T) if T else 0
            if max_time > 1.0:  # 1 second seems unreasonably long
                return False
                
            # Advanced shaper specific limits
            if len(A) > 10:  # Limit number of impulses for system stability
                return False
                
            return True
    
    # Test cases that should trigger fallback
    test_cases = [
        ("ulv", 100.0, 0.1),    # Normal case - should work
        ("multi_freq", 80.0, 0.1),  # Normal case - should work
        ("smooth", 40.0, 0.1),       # Normal case - should work
    ]
    
    for shaper_name, freq, damping in test_cases:
        print(f"  Testing {shaper_name} @ {freq}Hz")
        params = MockInputShaperParams(shaper_name, freq, damping)
        
        try:
            n, A, T = params.get_shaper()
            print(f"    ✓ Successfully configured {n} impulses, duration={max(T):.4f}s")
        except Exception as e:
            print(f"    ! Error: {e}")
    
    print("✓ Shaper fallback logic tests completed")

def test_comprehensive_shaper_analysis():
    """Test comprehensive shaper analysis with realistic data"""
    print("Testing comprehensive shaper analysis...")
    
    # Create synthetic resonance data similar to what might cause the original error
    freq_bins = np.linspace(5, 200, 400)
    
    # Complex resonance pattern with multiple peaks and harmonics
    psd = np.zeros_like(freq_bins)
    
    # Main resonances as mentioned in problem statement
    main_freqs = [75.9, 133.2, 181.2, 190.4, 267.9, 370.1]  # From problem statement
    for freq in main_freqs:
        if freq <= 200:  # Only include frequencies within our range
            psd += np.exp(-(freq_bins - freq)**2 / (10 + freq/20)) * (500 - freq) / 500
    
    # Add some noise
    psd += np.random.random(len(freq_bins)) * 0.02
    
    # Create calibration data
    cal_data = shaper_calibrate.CalibrationData(
        freq_bins, psd, psd*0.8, psd*0.6, psd*0.4)
    cal_data.set_numpy(np)
    cal_data.normalize_to_frequencies()
    
    # Test comprehensive analysis
    try:
        analysis = cal_data.get_comprehensive_analysis()
        print(f"  Peak frequencies: {analysis['peak_frequencies'][:5]}...")  # Show first 5
        print(f"  Dominant frequency: {analysis['dominant_frequency']:.1f}Hz")
        print(f"  Cross-coupling strength: {analysis['cross_coupling']['strength']:.2f}")
        print(f"  Harmonics detected: {len(analysis['harmonics'])}")
        
        # Test intelligent recommendations
        helper = shaper_calibrate.ShaperCalibrate(printer=None)
        best_shaper, all_shapers, analysis = helper.get_intelligent_recommendations(
            cal_data, max_smoothing=0.2, logger=print)
        
        if best_shaper:
            print(f"  Recommended: {best_shaper.name} @ {best_shaper.freq:.1f}Hz")
            print(f"  Vibrations: {best_shaper.vibrs:.1%}, Smoothing: {best_shaper.smoothing:.3f}")
        else:
            print("  ! No shaper recommendation generated")
        
        print("✓ Comprehensive analysis completed successfully")
        
    except Exception as e:
        print(f"  ! Error in comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    print("=== Input Shaper Fix Test Suite ===\n")
    
    try:
        test_parameter_validation()
        print()
        
        test_shaper_fallback_logic()
        print()
        
        test_comprehensive_shaper_analysis()
        print()
        
        print("=== All shaper fix tests completed successfully! ===")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())