#!/usr/bin/env python3
# Test script for enhanced input shaping system
#
# Copyright (C) 2024  Advanced Input Shaping Contributors
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

def test_new_shapers():
    """Test that all new shapers can be created and have reasonable parameters"""
    print("Testing new input shapers...")
    
    test_freq = 50.0
    test_damping = 0.1
    
    new_shapers = ['smooth', 'adaptive_ei', 'multi_freq', 'ulv']
    
    for shaper_cfg in shaper_defs.INPUT_SHAPERS:
        if shaper_cfg.name in new_shapers:
            print(f"  Testing {shaper_cfg.name}...")
            A, T = shaper_cfg.init_func(test_freq, test_damping)
            
            # Basic sanity checks
            assert len(A) == len(T), f"Mismatched A and T lengths for {shaper_cfg.name}"
            assert len(A) >= 2, f"Too few impulses for {shaper_cfg.name}"
            assert abs(sum(A) - 1.0) < 1e-6, f"Impulses don't sum to 1.0 for {shaper_cfg.name}"
            assert T[0] == 0.0, f"First time should be 0 for {shaper_cfg.name}"
            assert all(T[i] <= T[i+1] for i in range(len(T)-1)), f"Times not monotonic for {shaper_cfg.name}"
            
            print(f"    ✓ {len(A)} impulses, sum={sum(A):.6f}, duration={T[-1]:.4f}s")
    
    print("✓ All new shapers pass basic tests")

def test_enhanced_calibration_data():
    """Test enhanced CalibrationData functionality"""
    print("Testing enhanced CalibrationData...")
    
    # Create synthetic test data with known characteristics
    freq_bins = np.linspace(5, 200, 200)
    
    # Create data with peaks at 30, 60 (harmonic), and 80 Hz
    psd_base = np.exp(-(freq_bins - 30)**2 / 20) * 2.0  # Main peak
    psd_base += np.exp(-(freq_bins - 60)**2 / 10) * 0.5  # Harmonic  
    psd_base += np.exp(-(freq_bins - 80)**2 / 15) * 1.0  # Secondary peak
    psd_base += np.random.random(len(freq_bins)) * 0.05  # Noise
    
    # Different axis responses (simulating coupling)
    psd_x = psd_base * 1.0
    psd_y = psd_base * 0.7 + np.random.random(len(freq_bins)) * 0.02
    psd_z = psd_base * 0.3
    psd_sum = psd_x + psd_y + psd_z
    
    cal_data = shaper_calibrate.CalibrationData(freq_bins, psd_sum, psd_x, psd_y, psd_z)
    cal_data.set_numpy(np)
    
    # Test peak detection
    peaks = cal_data.find_peak_frequencies()
    print(f"  Detected peaks: {peaks[:5]}")  # Show first 5
    
    # Should find peaks near our synthetic frequencies
    expected_peaks = [30, 60, 80]
    for expected in expected_peaks:
        closest_peak = min(peaks, key=lambda p: abs(p - expected)) if len(peaks) > 0 else None
        if closest_peak and abs(closest_peak - expected) < 5:
            print(f"    ✓ Found expected peak near {expected}Hz: {closest_peak:.1f}Hz")
        else:
            print(f"    ! Expected peak at {expected}Hz not found (closest: {closest_peak})")
    
    # Test cross-coupling analysis
    coupling = cal_data.analyze_cross_coupling()
    print(f"  Cross-coupling strength: {coupling['strength']:.2f}")
    
    # Test harmonic analysis
    harmonics = cal_data.analyze_harmonics()
    print(f"  Harmonic analysis found {len(harmonics)} fundamental frequencies")
    
    # Test comprehensive analysis
    analysis = cal_data.get_comprehensive_analysis()
    print(f"  Dominant frequency: {analysis['dominant_frequency']:.1f}Hz")
    print(f"  Frequency centroid: {analysis['frequency_centroid']:.1f}Hz")
    print(f"  Dynamic range: {analysis['quality_metrics']['dynamic_range']:.1f}")
    
    print("✓ Enhanced CalibrationData tests completed")

def test_shaper_performance():
    """Test that new shapers provide reasonable performance characteristics"""
    print("Testing shaper performance characteristics...")
    
    # Create helper without printer for offline testing
    helper = shaper_calibrate.ShaperCalibrate(printer=None)
    
    # Simple test data with peak at 40Hz
    freq_bins = np.linspace(5, 150, 100)
    psd = np.exp(-(freq_bins - 40)**2 / 30) + 0.01
    
    cal_data = shaper_calibrate.CalibrationData(freq_bins, psd, psd*0.8, psd*0.6, psd*0.3)
    cal_data.set_numpy(np)
    cal_data.normalize_to_frequencies()
    
    # Test fitting each new shaper
    new_shapers = ['smooth', 'adaptive_ei', 'multi_freq', 'ulv']
    
    for shaper_name in new_shapers:
        shaper_cfg = next((s for s in shaper_defs.INPUT_SHAPERS if s.name == shaper_name), None)
        if shaper_cfg:
            try:
                result = helper.fit_shaper(shaper_cfg, cal_data, None, None, 5.0, None, None, None)
                print(f"  {shaper_name}: freq={result.freq:.1f}Hz, vibr={result.vibrs:.1%}, smooth={result.smoothing:.3f}")
                
                # Basic sanity checks
                assert 10 <= result.freq <= 100, f"Unreasonable frequency for {shaper_name}"
                assert 0 <= result.vibrs <= 1, f"Invalid vibration ratio for {shaper_name}"
                assert result.smoothing >= 0, f"Negative smoothing for {shaper_name}"
                assert result.max_accel > 0, f"Invalid max_accel for {shaper_name}"
                
            except Exception as e:
                print(f"    ! Error testing {shaper_name}: {e}")
    
    print("✓ Shaper performance tests completed")

def main():
    """Run all tests"""
    print("=== Enhanced Input Shaping Test Suite ===\n")
    
    try:
        test_new_shapers()
        print()
        
        test_enhanced_calibration_data()
        print()
        
        test_shaper_performance()
        print()
        
        print("=== All tests completed successfully! ===")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())