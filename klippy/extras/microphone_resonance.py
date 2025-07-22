# Microphone-based resonance testing
#
# Copyright (C) 2024  Enhanced Input Shaping Project
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging, math, os, time, threading, collections
import importlib

class MicrophoneData:
    def __init__(self, audio_samples, sample_rate, timestamps):
        self.audio_samples = audio_samples
        self.sample_rate = sample_rate
        self.timestamps = timestamps
        self.duration = len(audio_samples) / sample_rate if sample_rate > 0 else 0
        
    def get_sample_count(self):
        return len(self.audio_samples)
        
    def get_duration(self):
        return self.duration

class AudioFrequencyAnalyzer:
    def __init__(self, printer):
        self.printer = printer
        self.error = printer.command_error if printer else Exception
        try:
            self.numpy = importlib.import_module('numpy')
        except ImportError:
            raise self.error(
                "Failed to import `numpy` module required for audio analysis")
        
        # Try to import audio processing libraries
        self.has_scipy = False
        try:
            import scipy.signal
            self.scipy = importlib.import_module('scipy.signal')
            self.has_scipy = True
        except ImportError:
            pass
    
    def calculate_audio_psd(self, microphone_data, window_size=None):
        """Calculate power spectral density from audio data"""
        np = self.numpy
        
        if microphone_data.get_sample_count() == 0:
            return None
            
        audio = np.array(microphone_data.audio_samples)
        sample_rate = microphone_data.sample_rate
        
        # Apply windowing to reduce spectral leakage
        if window_size is None:
            window_size = min(8192, len(audio) // 4)
        
        # Ensure we have enough samples
        if len(audio) < window_size:
            return None
            
        if self.has_scipy:
            # Use scipy for better spectral analysis
            freqs, psd = self.scipy.welch(audio, fs=sample_rate, 
                                        nperseg=window_size, 
                                        window='hann',
                                        overlap=window_size//2)
        else:
            # Fallback to basic FFT
            # Apply Hann window
            window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1)))
            
            # Split audio into overlapping windows
            hop_size = window_size // 2
            n_windows = (len(audio) - window_size) // hop_size + 1
            
            if n_windows < 1:
                return None
                
            psd_accum = None
            for i in range(n_windows):
                start = i * hop_size
                end = start + window_size
                windowed = audio[start:end] * window
                
                # Calculate FFT
                fft = np.fft.rfft(windowed)
                power = np.abs(fft) ** 2
                
                if psd_accum is None:
                    psd_accum = power
                else:
                    psd_accum += power
            
            # Average and normalize
            psd = psd_accum / n_windows
            freqs = np.fft.rfftfreq(window_size, 1.0 / sample_rate)
        
        return freqs, psd
    
    def detect_audio_resonances(self, freqs, psd, min_prominence_db=6):
        """Detect resonance peaks in audio spectrum"""
        np = self.numpy
        
        # Convert to dB scale for better peak detection
        psd_db = 10 * np.log10(psd + 1e-12)
        
        # Simple peak detection
        peaks = []
        min_prominence = min_prominence_db
        
        for i in range(1, len(psd_db) - 1):
            # Check if current point is a local maximum
            if (psd_db[i] > psd_db[i-1] and psd_db[i] > psd_db[i+1]):
                # Check prominence (height above surrounding area)
                left_min = min(psd_db[max(0, i-10):i])
                right_min = min(psd_db[i+1:min(len(psd_db), i+11)])
                prominence = psd_db[i] - max(left_min, right_min)
                
                if prominence >= min_prominence:
                    peaks.append({
                        'frequency': freqs[i],
                        'amplitude_db': psd_db[i],
                        'prominence': prominence,
                        'index': i
                    })
        
        # Sort by amplitude (strongest peaks first)
        peaks.sort(key=lambda x: x['amplitude_db'], reverse=True)
        
        return peaks
    
    def cross_correlate_with_accelerometer(self, audio_peaks, accel_peaks, tolerance_hz=2.0):
        """Cross-correlate audio peaks with accelerometer peaks"""
        correlations = []
        
        for audio_peak in audio_peaks:
            audio_freq = audio_peak['frequency']
            best_match = None
            best_distance = float('inf')
            
            for accel_peak in accel_peaks:
                accel_freq = accel_peak
                distance = abs(audio_freq - accel_freq)
                
                if distance <= tolerance_hz and distance < best_distance:
                    best_distance = distance
                    best_match = accel_freq
            
            correlation = {
                'audio_peak': audio_peak,
                'accelerometer_match': best_match,
                'frequency_difference': best_distance if best_match else None,
                'confidence': 1.0 - (best_distance / tolerance_hz) if best_match else 0.0
            }
            correlations.append(correlation)
        
        return correlations

class MicrophoneResonanceTester:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.audio_device = config.get('audio_device', 'default')
        self.sample_rate = config.getint('sample_rate', 44100, minval=8000, maxval=192000)
        self.buffer_duration = config.getfloat('buffer_duration', 2.0, minval=0.5, maxval=10.0)
        self.noise_threshold_db = config.getfloat('noise_threshold_db', -60.0, maxval=-20.0)
        self.min_frequency = config.getfloat('min_frequency', 5.0, minval=1.0)
        self.max_frequency = config.getfloat('max_frequency', 200.0, maxval=1000.0)
        
        self.analyzer = AudioFrequencyAnalyzer(self.printer)
        self.recording = False
        self.audio_buffer = []
        self.buffer_timestamps = []
        
        # Try to import audio recording library
        self.has_audio = False
        self.audio_lib = None
        try:
            import pyaudio
            self.audio_lib = pyaudio
            self.has_audio = True
        except ImportError:
            # Try alternative audio libraries
            try:
                import sounddevice as sd
                self.audio_lib = sd
                self.has_audio = True
                self.audio_backend = 'sounddevice'
            except ImportError:
                logging.warning("No audio library available. Microphone testing disabled.")
                self.audio_backend = None
        
        if self.has_audio and hasattr(self.audio_lib, 'PyAudio'):
            self.audio_backend = 'pyaudio'
    
    def start_recording(self):
        """Start recording audio during resonance test"""
        if not self.has_audio:
            raise self.printer.command_error(
                "Audio recording not available. Install pyaudio or sounddevice.")
        
        self.recording = True
        self.audio_buffer = []
        self.buffer_timestamps = []
        
        if self.audio_backend == 'pyaudio':
            self._start_pyaudio_recording()
        elif self.audio_backend == 'sounddevice':
            self._start_sounddevice_recording()
        
    def stop_recording(self):
        """Stop recording and return audio data"""
        if not self.recording:
            return None
            
        self.recording = False
        
        if len(self.audio_buffer) == 0:
            return None
            
        return MicrophoneData(
            audio_samples=self.audio_buffer,
            sample_rate=self.sample_rate,
            timestamps=self.buffer_timestamps
        )
    
    def _start_pyaudio_recording(self):
        """Start recording using PyAudio"""
        import threading
        
        def record_audio():
            pa = self.audio_lib.PyAudio()
            stream = pa.open(
                format=self.audio_lib.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024
            )
            
            while self.recording:
                try:
                    data = stream.read(1024, exception_on_overflow=False)
                    import struct
                    samples = struct.unpack('f' * 1024, data)
                    timestamp = time.time()
                    
                    self.audio_buffer.extend(samples)
                    self.buffer_timestamps.extend([timestamp] * len(samples))
                except Exception as e:
                    logging.error("Audio recording error: %s" % str(e))
                    break
            
            stream.stop_stream()
            stream.close()
            pa.terminate()
        
        self.recording_thread = threading.Thread(target=record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def _start_sounddevice_recording(self):
        """Start recording using sounddevice"""
        import threading
        
        def record_audio():
            try:
                while self.recording:
                    duration = 0.1  # Record in 100ms chunks
                    audio = self.audio_lib.rec(
                        int(duration * self.sample_rate),
                        samplerate=self.sample_rate,
                        channels=1,
                        dtype='float32'
                    )
                    self.audio_lib.wait()
                    
                    timestamp = time.time()
                    samples = audio.flatten().tolist()
                    
                    self.audio_buffer.extend(samples)
                    self.buffer_timestamps.extend([timestamp] * len(samples))
            except Exception as e:
                logging.error("Audio recording error: %s" % str(e))
        
        self.recording_thread = threading.Thread(target=record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def analyze_audio_resonances(self, microphone_data):
        """Analyze audio data for resonance frequencies"""
        if microphone_data is None or microphone_data.get_sample_count() == 0:
            return None
            
        # Calculate power spectral density
        freqs, psd = self.analyzer.calculate_audio_psd(microphone_data)
        if freqs is None or psd is None:
            return None
        
        # Filter frequency range
        mask = (freqs >= self.min_frequency) & (freqs <= self.max_frequency)
        freqs = freqs[mask]
        psd = psd[mask]
        
        # Detect resonance peaks
        peaks = self.analyzer.detect_audio_resonances(freqs, psd)
        
        return {
            'frequencies': freqs,
            'psd': psd,
            'peaks': peaks,
            'sample_rate': microphone_data.sample_rate,
            'duration': microphone_data.duration
        }

def load_config(config):
    return MicrophoneResonanceTester(config)