#!/usr/bin/env python3
"""
Microphone-Enhanced Resonance Testing Demo

This script demonstrates the microphone integration capabilities
for enhanced resonance compensation in Klipper.
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Add the klippy path to import modules
sys.path.insert(0, '/home/runner/work/klipper/klipper/klippy')

def generate_synthetic_audio_data(duration=2.0, sample_rate=44100, 
                                 resonance_freqs=[45.2, 89.7, 123.4], 
                                 noise_level=0.1):
    """Generate synthetic audio data with resonance peaks"""
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Generate base noise
    audio = noise_level * np.random.normal(0, 1, len(t))
    
    # Add resonance frequencies with harmonics
    for freq in resonance_freqs:
        # Fundamental frequency
        amplitude = 1.0 / (1 + (freq - 50) / 100)  # Decay with frequency
        audio += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add some harmonic content
        if freq * 2 < sample_rate / 2:
            audio += 0.3 * amplitude * np.sin(2 * np.pi * freq * 2 * t)
        if freq * 3 < sample_rate / 2:
            audio += 0.1 * amplitude * np.sin(2 * np.pi * freq * 3 * t)
    
    # Add some broadband printer noise
    audio += 0.2 * np.sin(2 * np.pi * 60 * t)  # 60 Hz mains hum
    audio += 0.1 * np.sin(2 * np.pi * 120 * t)  # 120 Hz harmonic
    
    return audio.tolist(), [time.time() + i / sample_rate for i in range(len(audio))]

def generate_synthetic_accelerometer_data(resonance_freqs=[45.2, 89.7, 123.4]):
    """Generate synthetic accelerometer data"""
    freq_bins = np.linspace(5, 200, 1000)
    psd_sum = np.ones_like(freq_bins) * 0.01  # Background noise
    psd_x = np.ones_like(freq_bins) * 0.005
    psd_y = np.ones_like(freq_bins) * 0.005
    psd_z = np.ones_like(freq_bins) * 0.002
    
    # Add resonance peaks
    for freq in resonance_freqs:
        # Find closest frequency bin
        idx = np.argmin(np.abs(freq_bins - freq))
        
        # Add peak with some width
        peak_width = 5  # Hz
        for i in range(max(0, idx-peak_width), min(len(freq_bins), idx+peak_width+1)):
            distance = abs(freq_bins[i] - freq)
            amplitude = np.exp(-distance**2 / (2 * (peak_width/3)**2))
            psd_sum[i] += amplitude * 0.5
            psd_x[i] += amplitude * 0.3
            psd_y[i] += amplitude * 0.2
            psd_z[i] += amplitude * 0.1
    
    return freq_bins, psd_sum, psd_x, psd_y, psd_z

class MockPrinter:
    """Mock printer object for testing"""
    def command_error(self, msg):
        return Exception(msg)

def demo_microphone_analysis():
    """Demonstrate microphone analysis capabilities"""
    print("=== Microphone-Enhanced Resonance Testing Demo ===\n")
    
    # Import our modules
    try:
        from extras.microphone_resonance import MicrophoneData, AudioFrequencyAnalyzer
        from extras.shaper_calibrate import CalibrationData
    except ImportError as e:
        print(f"Import error: {e}")
        print("This demo requires the microphone modules to be in the Python path")
        return
    
    # Create mock printer
    printer = MockPrinter()
    
    print("1. Generating synthetic test data...")
    
    # Generate synthetic audio data with known resonances
    resonance_freqs = [45.2, 89.7, 123.4]
    audio_samples, timestamps = generate_synthetic_audio_data(
        duration=2.0, resonance_freqs=resonance_freqs)
    
    # Create microphone data object
    mic_data = MicrophoneData(
        audio_samples=audio_samples,
        sample_rate=44100,
        timestamps=timestamps
    )
    
    print(f"Generated {mic_data.get_sample_count()} audio samples over {mic_data.duration:.1f} seconds")
    
    # Generate synthetic accelerometer data
    freq_bins, psd_sum, psd_x, psd_y, psd_z = generate_synthetic_accelerometer_data(
        resonance_freqs=resonance_freqs)
    
    accel_data = CalibrationData(freq_bins, psd_sum, psd_x, psd_y, psd_z)
    accel_data.set_numpy(np)
    
    print("\n2. Analyzing audio data...")
    
    # Analyze audio
    analyzer = AudioFrequencyAnalyzer(printer)
    
    # Calculate audio PSD
    audio_freqs, audio_psd = analyzer.calculate_audio_psd(mic_data)
    print(f"Audio analysis: {len(audio_freqs)} frequency bins from {audio_freqs[0]:.1f} to {audio_freqs[-1]:.1f} Hz")
    
    # Detect audio peaks
    audio_peaks = analyzer.detect_audio_resonances(audio_freqs, audio_psd)
    print(f"Detected {len(audio_peaks)} audio peaks:")
    for i, peak in enumerate(audio_peaks[:5]):  # Show top 5
        print(f"  {i+1}. {peak['frequency']:.1f} Hz ({peak['amplitude_db']:.1f} dB)")
    
    print("\n3. Analyzing accelerometer data...")
    
    # Find accelerometer peaks
    accel_peaks = accel_data.find_peak_frequencies()
    print(f"Detected {len(accel_peaks)} accelerometer peaks:")
    for i, freq in enumerate(accel_peaks):
        print(f"  {i+1}. {freq:.1f} Hz")
    
    print("\n4. Cross-correlating audio and accelerometer data...")
    
    # Cross-correlate peaks
    correlations = analyzer.cross_correlate_with_accelerometer(audio_peaks, accel_peaks)
    
    confirmed_peaks = 0
    for correlation in correlations:
        audio_freq = correlation['audio_peak']['frequency']
        accel_match = correlation['accelerometer_match']
        confidence = correlation['confidence']
        
        if accel_match is not None:
            if confidence > 0.8:
                print(f"CONFIRMED: {audio_freq:.1f} Hz (audio) matches {accel_match:.1f} Hz (accel), confidence {confidence*100:.1f}%")
                confirmed_peaks += 1
            else:
                print(f"WEAK MATCH: {audio_freq:.1f} Hz (audio) ~ {accel_match:.1f} Hz (accel), confidence {confidence*100:.1f}%")
        else:
            print(f"NEW PEAK: {audio_freq:.1f} Hz detected in audio only")
    
    print(f"\nCorrelation summary: {confirmed_peaks} out of {len(audio_peaks)} audio peaks confirmed")
    
    print("\n5. Comprehensive analysis...")
    
    # Add microphone data to calibration data
    accel_data._microphone_analysis = {
        'frequencies': audio_freqs,
        'psd': audio_psd,
        'peaks': audio_peaks,
        'sample_rate': mic_data.sample_rate,
        'duration': mic_data.duration
    }
    
    # Get comprehensive analysis
    analysis = accel_data.get_comprehensive_analysis()
    
    print("Peak frequencies (accelerometer):", [f"{f:.1f}" for f in analysis['peak_frequencies']])
    print("Peak frequencies (microphone):", [f"{p['frequency']:.1f}" for p in audio_peaks])
    print(f"Dominant frequency: {analysis['dominant_frequency']:.1f} Hz")
    print(f"Cross-axis coupling: {analysis['cross_coupling']['strength']:.2f}")
    print(f"Frequency centroid: {analysis['frequency_centroid']:.1f} Hz")
    
    print("\n6. Creating visualization...")
    
    # Create visualization
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot accelerometer PSD
        ax1.semilogy(freq_bins, psd_sum, 'b-', label='Accelerometer PSD')
        for peak in accel_peaks:
            ax1.axvline(peak, color='blue', linestyle='--', alpha=0.7, label=f'Accel Peak {peak:.1f}Hz' if peak == accel_peaks[0] else '')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('PSD')
        ax1.set_title('Accelerometer Frequency Response')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot audio PSD
        audio_psd_db = 10 * np.log10(audio_psd + 1e-12)
        ax2.plot(audio_freqs, audio_psd_db, 'r-', label='Microphone PSD')
        for peak in audio_peaks[:5]:  # Show top 5 peaks
            ax2.axvline(peak['frequency'], color='red', linestyle='--', alpha=0.7)
            ax2.text(peak['frequency'], peak['amplitude_db'], f'{peak["frequency"]:.1f}Hz', 
                    rotation=90, ha='right', va='bottom')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('PSD (dB)')
        ax2.set_title('Microphone Frequency Response')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot correlation
        accel_freq_list = list(accel_peaks)
        audio_freq_list = [p['frequency'] for p in audio_peaks]
        
        ax3.scatter(accel_freq_list, [1]*len(accel_freq_list), 
                   c='blue', s=100, alpha=0.7, label='Accelerometer Peaks')
        ax3.scatter(audio_freq_list, [2]*len(audio_freq_list), 
                   c='red', s=100, alpha=0.7, label='Microphone Peaks')
        
        # Draw correlation lines
        for correlation in correlations:
            if correlation['accelerometer_match'] is not None:
                audio_freq = correlation['audio_peak']['frequency']
                accel_freq = correlation['accelerometer_match']
                confidence = correlation['confidence']
                
                color = 'green' if confidence > 0.8 else 'orange'
                ax3.plot([accel_freq, audio_freq], [1, 2], 
                        color=color, alpha=confidence, linewidth=2)
        
        ax3.set_ylim(0.5, 2.5)
        ax3.set_yticks([1, 2])
        ax3.set_yticklabels(['Accelerometer', 'Microphone'])
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_title('Peak Correlation Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/tmp/microphone_demo_analysis.png', dpi=150, bbox_inches='tight')
        print("Visualization saved to /tmp/microphone_demo_analysis.png")
        
    except ImportError:
        print("matplotlib not available - skipping visualization")
    
    print("\n=== Demo Complete ===")
    print("\nThis demo shows how microphone data can enhance resonance testing by:")
    print("1. Providing independent validation of accelerometer measurements")
    print("2. Detecting resonances that might be missed by accelerometers")
    print("3. Improving confidence in frequency identification")
    print("4. Enabling cross-correlation analysis for better accuracy")

if __name__ == '__main__':
    demo_microphone_analysis()