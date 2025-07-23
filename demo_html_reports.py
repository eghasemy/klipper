#!/usr/bin/env python3
"""
Demonstration of HTML Report Generation for Input Shaper Calibration

This script shows how to generate comprehensive HTML reports from calibration data,
providing users with detailed visual analysis and recommendations.
"""

import sys
import os
import numpy as np
import tempfile

# Add klippy to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'klippy'))

def create_sample_calibration_data():
    """Create sample calibration data for demonstration"""
    # Import necessary modules
    import importlib
    shaper_calibrate = importlib.import_module('.shaper_calibrate', 'extras')
    
    # Create frequency bins from 5 to 200 Hz
    freq_bins = np.linspace(5, 200, 400)
    
    # Create realistic PSD data with multiple resonance peaks
    psd_base = np.ones_like(freq_bins) * 100  # Base noise level
    
    # Add resonance peaks at typical frequencies
    for peak_freq, amplitude in [(42.5, 5000), (67.3, 3000), (105.8, 2500), (145.2, 1800)]:
        # Gaussian peak with some width
        width = 2.0
        peak = amplitude * np.exp(-((freq_bins - peak_freq) / width) ** 2)
        psd_base += peak
    
    # Add some randomness
    np.random.seed(42)  # For reproducible results
    psd_base *= (1 + 0.1 * np.random.randn(len(freq_bins)))
    psd_base = np.maximum(psd_base, 50)  # Ensure positive values
    
    # Create X, Y, Z components with some variation
    psd_x = psd_base * (1 + 0.2 * np.random.randn(len(freq_bins)))
    psd_y = psd_base * (0.8 + 0.2 * np.random.randn(len(freq_bins)))
    psd_z = psd_base * (0.3 + 0.1 * np.random.randn(len(freq_bins)))
    
    # Ensure all values are positive
    psd_x = np.maximum(psd_x, 10)
    psd_y = np.maximum(psd_y, 10) 
    psd_z = np.maximum(psd_z, 10)
    psd_sum = psd_x + psd_y + psd_z
    
    # Create CalibrationData object
    calibration_data = shaper_calibrate.CalibrationData(
        freq_bins, psd_sum, psd_x, psd_y, psd_z)
    calibration_data.set_numpy(np)
    
    return calibration_data

def demonstrate_html_report():
    """Demonstrate HTML report generation"""
    print("=== HTML Report Generation Demonstration ===")
    print()
    
    try:
        # Create sample calibration data
        print("1. Creating sample calibration data...")
        calibration_data = create_sample_calibration_data()
        print("   ✓ Sample data created with realistic resonance patterns")
        
        # Import shaper calibration
        import importlib
        shaper_calibrate = importlib.import_module('.shaper_calibrate', 'extras')
        
        # Setup shaper calibration
        print("\n2. Running shaper analysis...")
        helper = shaper_calibrate.ShaperCalibrate(printer=None)
        
        # Find best shapers
        best_shaper, all_shapers = helper.find_best_shaper(
            calibration_data, max_smoothing=None, scv=5.0, max_freq=200.0)
        
        print(f"   ✓ Analyzed {len(all_shapers)} shaper configurations")
        print(f"   ✓ Recommended: {best_shaper.name} at {best_shaper.freq:.1f} Hz")
        
        # Generate HTML report
        print("\n3. Generating comprehensive HTML report...")
        
        # Import report generator
        from extras import resonance_report
        
        # Create report generator
        generator = resonance_report.ResonanceReportGenerator(
            calibration_data, all_shapers, best_shaper)
        
        # Generate report to temporary file
        output_path = os.path.join(tempfile.gettempdir(), "klipper_resonance_demo_report.html")
        generator.generate_html_report(output_path)
        
        print(f"   ✓ HTML report generated successfully!")
        print(f"   ✓ Report saved to: {output_path}")
        
        print("\n4. Report features included:")
        print("   • Interactive frequency response graphs")
        print("   • Shaper performance comparison charts")
        print("   • Peak frequency analysis with source identification")
        print("   • Cross-axis coupling analysis")
        print("   • Harmonic content analysis")
        print("   • Specific configuration recommendations")
        print("   • Mechanical improvement suggestions")
        print("   • Troubleshooting guide")
        print("   • Advanced quality metrics")
        
        print(f"\n🎉 SUCCESS: Open {output_path} in your web browser")
        print("    to view the comprehensive analysis and recommendations!")
        
        return output_path
        
    except ImportError as e:
        print(f"❌ Error: Missing required module - {e}")
        print("   Make sure numpy and matplotlib are installed")
        return None
    except Exception as e:
        print(f"❌ Error generating HTML report: {e}")
        import traceback
        traceback.print_exc()
        return None

def demonstrate_usage_examples():
    """Show usage examples for the HTML report feature"""
    print("\n=== Usage Examples ===")
    print()
    
    print("📋 Command Line Usage:")
    print("python3 scripts/calibrate_shaper.py --html report.html calibration_data.csv")
    print()
    
    print("🖥️  G-code Usage in Klipper:")
    print("SHAPER_CALIBRATE")
    print("GENERATE_SHAPER_REPORT INPUT=/tmp/calibration_data_*.csv OUTPUT=/tmp/report.html")
    print()
    
    print("🔧 Integration with existing workflow:")
    print("1. Run SHAPER_CALIBRATE as usual")
    print("2. Use GENERATE_SHAPER_REPORT to create visual analysis")
    print("3. Open HTML file in browser for detailed insights")
    print("4. Follow recommendations for optimal configuration")
    print()
    
    print("💡 Benefits of HTML Reports:")
    print("• Visual understanding of resonance patterns")
    print("• Educational content explaining recommendations")  
    print("• Interactive charts for detailed analysis")
    print("• Specific improvement suggestions")
    print("• Troubleshooting guidance")
    print("• Historical comparison capabilities")

if __name__ == "__main__":
    print("🔧 Klipper Enhanced Input Shaper - HTML Report Demonstration")
    print("=" * 60)
    
    # Run demonstration
    report_path = demonstrate_html_report()
    
    # Show usage examples
    demonstrate_usage_examples()
    
    if report_path:
        print(f"\n📊 Demo report available at: {report_path}")
        print("Open this file in any web browser to see the comprehensive analysis!")
    
    print("\n✨ This enhancement makes input shaper calibration more accessible")
    print("   and educational for users at all experience levels.")