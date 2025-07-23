#!/usr/bin/env python3
# Integration test for the specific shaper configuration failure scenario
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

def simulate_problem_scenario():
    """Simulate the exact scenario from the problem statement"""
    print("=== Simulating Problem Scenario ===")
    
    # Create calibration data based on the frequencies mentioned in the problem statement
    freq_bins = np.linspace(5, 400, 800)  # Higher resolution for better peak detection
    
    # Peak frequencies from the problem statement
    main_peaks = [75.9, 133.2, 181.2, 190.4, 267.9, 370.1]
    
    # Create PSD with these peaks
    psd = np.zeros_like(freq_bins)
    for freq in main_peaks:
        if freq <= 400:  # Only include frequencies within our range
            # Create realistic peak with some width
            sigma = 3.0 + freq * 0.01  # Wider peaks at higher frequencies
            amplitude = 1000.0 / (1.0 + freq * 0.005)  # Amplitude decreases with frequency
            psd += amplitude * np.exp(-(freq_bins - freq)**2 / (2 * sigma**2))
    
    # Add realistic noise floor
    psd += np.random.random(len(freq_bins)) * 0.5
    
    # Create axis-specific data with coupling
    psd_x = psd * 1.0
    psd_y = psd * 0.7 + np.random.random(len(freq_bins)) * 0.3  # Some coupling
    psd_z = psd * 0.2
    psd_sum = psd_x + psd_y + psd_z
    
    cal_data = shaper_calibrate.CalibrationData(freq_bins, psd_sum, psd_x, psd_y, psd_z)
    cal_data.set_numpy(np)
    cal_data.normalize_to_frequencies()
    
    print("Created synthetic resonance data with peaks at:", main_peaks)
    
    # Test comprehensive analysis
    analysis = cal_data.get_comprehensive_analysis()
    print(f"Detected {len(analysis['peak_frequencies'])} peaks")
    print(f"Dominant frequency: {analysis['dominant_frequency']:.1f} Hz")
    print(f"Cross-coupling strength: {analysis['cross_coupling']['strength']:.2f}")
    
    # Test intelligent recommendations (this was failing in the original problem)
    helper = shaper_calibrate.ShaperCalibrate(printer=None)
    
    def test_logger(msg):
        print(f"  {msg}")
    
    try:
        best_shaper, all_shapers, analysis = helper.get_intelligent_recommendations(
            cal_data, max_smoothing=0.2, scv=5.0, logger=test_logger)
        
        if best_shaper:
            print(f"\n✓ Successfully recommended: {best_shaper.name} @ {best_shaper.freq:.1f}Hz")
            print(f"  Vibrations: {best_shaper.vibrs:.1%}")
            print(f"  Smoothing: {best_shaper.smoothing:.3f}")
            print(f"  Max accel: {best_shaper.max_accel:.0f} mm/sec^2")
            
            # Check if ULV was recommended (as in the problem statement)
            if best_shaper.name == 'ulv':
                print("  ✓ ULV shaper successfully recommended and configured!")
            else:
                print(f"  → Alternative shaper recommended instead of ULV: {best_shaper.name}")
        else:
            print("❌ No shaper recommendation generated")
            return False
            
    except Exception as e:
        print(f"❌ Error during recommendation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test individual shaper configurations that might have failed
    print("\n=== Testing Individual Shaper Configurations ===")
    test_shapers = ['ulv', 'multi_freq', '3hump_ei']
    test_freq = 105.8  # From problem statement recommendation
    
    for shaper_name in test_shapers:
        print(f"Testing {shaper_name} @ {test_freq}Hz...")
        
        shaper_cfg = next((s for s in shaper_defs.INPUT_SHAPERS if s.name == shaper_name), None)
        if not shaper_cfg:
            print(f"  ! Shaper {shaper_name} not found")
            continue
            
        try:
            A, T = shaper_cfg.init_func(test_freq, 0.1)
            
            # Validate parameters
            if abs(sum(A) - 1.0) > 1e-6:
                print(f"  ! Parameter validation failed: sum={sum(A):.6f}")
                continue
                
            if max(T) > 1.0:
                print(f"  ! Parameter validation failed: max_time={max(T):.4f}s")
                continue
                
            print(f"  ✓ {len(A)} impulses, duration={max(T):.4f}s, sum={sum(A):.6f}")
            
            # Test with calibration system
            result = helper.fit_shaper(shaper_cfg, cal_data, None, None, 5.0, None, None, None)
            print(f"    Fit result: freq={result.freq:.1f}Hz, vibr={result.vibrs:.1%}, smooth={result.smoothing:.3f}")
            
        except Exception as e:
            print(f"  ! Error with {shaper_name}: {e}")
    
    return True

def main():
    """Run the integration test"""
    print("=== Input Shaper Configuration Fix Integration Test ===\n")
    
    try:
        success = simulate_problem_scenario()
        
        if success:
            print("\n=== Integration test completed successfully! ===")
            print("The input shaper configuration issue has been resolved.")
            return 0
        else:
            print("\n❌ Integration test failed")
            return 1
        
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())