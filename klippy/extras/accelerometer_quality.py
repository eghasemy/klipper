# Real-time accelerometer-based print quality control
#
# Copyright (C) 2025  Klipper Contributors
#
# This file may be distributed under the terms of the GNU GPLv3 license.

import logging, threading, time, math, cmath, collections

class AccelerometerQuality:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.name = config.get_name()
        
        # Configuration parameters
        self.accel_chip_name = config.get('accel_chip', 'adxl345')
        self.vibration_threshold = config.getfloat('vibration_threshold', 5000.0)
        self.speed_reduction_factor = config.getfloat('speed_reduction_factor', 0.8, minval=0.1, maxval=1.0)
        self.speed_increase_factor = config.getfloat('speed_increase_factor', 1.1, minval=1.0, maxval=2.0)
        self.sample_time = config.getfloat('sample_time', 1.0, minval=0.1, maxval=10.0)
        self.stable_threshold = config.getfloat('stable_threshold', 1000.0)
        
        # Enhanced processing configuration
        self.simulation_mode = config.getboolean('simulation_mode', False)
        self.sample_frequency = config.getfloat('sample_frequency', 100.0, minval=10.0, maxval=1000.0)
        self.enable_frequency_analysis = config.getboolean('enable_frequency_analysis', True)
        self.resonance_frequency_threshold = config.getfloat('resonance_frequency_threshold', 50.0, minval=1.0, maxval=200.0)
        self.filter_cutoff_frequency = config.getfloat('filter_cutoff_frequency', 20.0, minval=1.0, maxval=100.0)
        
        # State variables
        self.is_enabled = False
        self.is_monitoring = False
        self.monitor_thread = None
        self.current_speed_factor = 100.0  # Percentage
        self.baseline_speed_factor = 100.0
        self.accel_chip = None
        self.accel_client = None
        self.lock = threading.Lock()
        
        # Simulation mode and decision tracking
        self.simulation_decisions = collections.deque(maxlen=1000)  # Store last 1000 decisions
        self.decision_count = 0
        
        # Signal processing state
        self.sample_buffer = collections.deque(maxlen=int(self.sample_frequency * self.sample_time * 2))
        self.filter_state_x = collections.deque(maxlen=3)
        self.filter_state_y = collections.deque(maxlen=3)
        self.filter_state_z = collections.deque(maxlen=3)
        
        # Register event handlers
        self.printer.register_event_handler("klippy:ready", self._handle_ready)
        self.printer.register_event_handler("klippy:shutdown", self._handle_shutdown)
        
        # Register G-code commands
        gcode = self.printer.lookup_object('gcode')
        gcode.register_command('ACCELEROMETER_QUALITY_ENABLE', 
                              self.cmd_ACCELEROMETER_QUALITY_ENABLE,
                              desc=self.cmd_ACCELEROMETER_QUALITY_ENABLE_help)
        gcode.register_command('ACCELEROMETER_QUALITY_DISABLE',
                              self.cmd_ACCELEROMETER_QUALITY_DISABLE, 
                              desc=self.cmd_ACCELEROMETER_QUALITY_DISABLE_help)
        gcode.register_command('ACCELEROMETER_QUALITY_STATUS',
                              self.cmd_ACCELEROMETER_QUALITY_STATUS,
                              desc=self.cmd_ACCELEROMETER_QUALITY_STATUS_help)
        gcode.register_command('ACCELEROMETER_QUALITY_DECISIONS',
                              self.cmd_ACCELEROMETER_QUALITY_DECISIONS,
                              desc=self.cmd_ACCELEROMETER_QUALITY_DECISIONS_help)
        gcode.register_command('ACCELEROMETER_QUALITY_SIMULATION',
                              self.cmd_ACCELEROMETER_QUALITY_SIMULATION,
                              desc=self.cmd_ACCELEROMETER_QUALITY_SIMULATION_help)
        
    def _handle_ready(self):
        # Get reference to accelerometer chip
        try:
            self.accel_chip = self.printer.lookup_object(self.accel_chip_name)
        except Exception as e:
            logging.warning("AccelerometerQuality: Could not find accelerometer chip '%s': %s", 
                          self.accel_chip_name, e)
            self.accel_chip = None
            
    def _handle_shutdown(self):
        self._stop_monitoring()
    
    def _fft(self, samples):
        """Simple FFT implementation using Cooley-Tukey algorithm"""
        n = len(samples)
        if n <= 1:
            return samples
        if n % 2 != 0:
            # Pad with zeros for power of 2
            samples = samples + [0] * (2 ** math.ceil(math.log2(n)) - n)
            n = len(samples)
        
        # Base case
        if n == 2:
            return [samples[0] + samples[1], samples[0] - samples[1]]
        
        # Divide
        even = [samples[i] for i in range(0, n, 2)]
        odd = [samples[i] for i in range(1, n, 2)]
        
        # Conquer
        even_fft = self._fft(even)
        odd_fft = self._fft(odd)
        
        # Combine
        result = [0] * n
        for i in range(n // 2):
            t = cmath.exp(-2j * cmath.pi * i / n) * odd_fft[i]
            result[i] = even_fft[i] + t
            result[i + n // 2] = even_fft[i] - t
        
        return result
    
    def _low_pass_filter(self, sample, filter_state, cutoff_freq, sample_freq):
        """Simple low-pass filter (1st order Butterworth)"""
        rc = 1.0 / (2 * math.pi * cutoff_freq)
        dt = 1.0 / sample_freq
        alpha = dt / (rc + dt)
        
        if not filter_state:
            filter_state.append(sample)
            return sample
        
        filtered = alpha * sample + (1 - alpha) * filter_state[-1]
        filter_state.append(filtered)
        return filtered
    
    def _extract_frequency_features(self, samples, sample_freq):
        """Extract frequency domain features from accelerometer data"""
        if len(samples) < 8:  # Need minimum samples for meaningful FFT
            return {}
        
        # Apply FFT
        fft_result = self._fft(samples)
        n = len(fft_result)
        
        # Calculate magnitude spectrum (only positive frequencies)
        magnitudes = []
        frequencies = []
        for i in range(n // 2):
            mag = abs(fft_result[i])
            freq = i * sample_freq / n
            magnitudes.append(mag)
            frequencies.append(freq)
        
        if not magnitudes:
            return {}
        
        # Find dominant frequency
        max_mag_idx = magnitudes.index(max(magnitudes))
        dominant_frequency = frequencies[max_mag_idx]
        
        # Calculate frequency distribution features
        total_energy = sum(mag ** 2 for mag in magnitudes)
        low_freq_energy = sum(mag ** 2 for i, mag in enumerate(magnitudes) if frequencies[i] < 10)
        mid_freq_energy = sum(mag ** 2 for i, mag in enumerate(magnitudes) if 10 <= frequencies[i] < 50)
        high_freq_energy = sum(mag ** 2 for i, mag in enumerate(magnitudes) if frequencies[i] >= 50)
        
        # Calculate spectral centroid (frequency "center of mass")
        if total_energy > 0:
            spectral_centroid = sum(freq * mag ** 2 for freq, mag in zip(frequencies, magnitudes)) / total_energy
        else:
            spectral_centroid = 0
        
        return {
            'dominant_frequency': dominant_frequency,
            'dominant_magnitude': magnitudes[max_mag_idx],
            'spectral_centroid': spectral_centroid,
            'low_freq_ratio': low_freq_energy / total_energy if total_energy > 0 else 0,
            'mid_freq_ratio': mid_freq_energy / total_energy if total_energy > 0 else 0,
            'high_freq_ratio': high_freq_energy / total_energy if total_energy > 0 else 0,
            'total_energy': total_energy
        }
        
    def _start_monitoring(self):
        """Start the background monitoring thread"""
        if self.is_monitoring or self.accel_chip is None:
            return False
            
        with self.lock:
            if self.is_monitoring:
                return False
            self.is_monitoring = True
            
        # Store current speed factor as baseline
        gcode_move = self.printer.lookup_object('gcode_move')
        self.baseline_speed_factor = gcode_move.get_status()['speed_factor']
        self.current_speed_factor = self.baseline_speed_factor
        
        # Start accelerometer client
        try:
            self.accel_client = self.accel_chip.start_internal_client()
        except Exception as e:
            logging.error("AccelerometerQuality: Failed to start accelerometer client: %s", e)
            with self.lock:
                self.is_monitoring = False
            return False
            
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logging.info("AccelerometerQuality: Started monitoring with chip '%s'", self.accel_chip_name)
        return True
        
    def _stop_monitoring(self):
        """Stop the background monitoring thread"""
        with self.lock:
            if not self.is_monitoring:
                return
            self.is_monitoring = False
            
        # Wait for thread to finish
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=5.0)
            self.monitor_thread = None
            
        # Stop accelerometer client
        if self.accel_client is not None:
            try:
                self.accel_client.finish_measurements()
            except Exception as e:
                logging.warning("AccelerometerQuality: Error stopping accelerometer: %s", e)
            self.accel_client = None
            
        # Reset speed factor to baseline
        self._set_speed_factor(self.baseline_speed_factor)
        
        logging.info("AccelerometerQuality: Stopped monitoring")
        
    def _monitor_loop(self):
        """Main monitoring loop that runs in background thread"""
        sample_count = 0
        try:
            while True:
                with self.lock:
                    if not self.is_monitoring:
                        break
                
                # Sample accelerometer data
                try:
                    vibration_data = self._measure_vibration()
                    self._adjust_speed_based_on_vibration(vibration_data)
                    
                    # Periodically clean up old samples to prevent memory buildup
                    sample_count += 1
                    if sample_count % 30 == 0:  # Every 30 samples
                        self._cleanup_old_samples()
                        
                except Exception as e:
                    logging.warning("AccelerometerQuality: Error in monitoring loop: %s", e)
                    
                # Sleep until next sample
                time.sleep(self.sample_time)
                
        except Exception as e:
            logging.error("AccelerometerQuality: Monitor loop error: %s", e)
        finally:
            with self.lock:
                self.is_monitoring = False
                
    def _cleanup_old_samples(self):
        """Clean up old accelerometer samples to prevent memory buildup"""
        if self.accel_client is None:
            return
            
        # Get current samples and clear old ones
        # The AccelQueryHelper accumulates samples, so we need to limit this
        try:
            samples = self.accel_client.get_samples()
            if len(samples) > 1000:  # Keep last 1000 samples max
                # Create a new client to reset sample buffer
                old_client = self.accel_client
                self.accel_client = self.accel_chip.start_internal_client()
                old_client.finish_measurements()
        except Exception as e:
            logging.warning("AccelerometerQuality: Error cleaning up samples: %s", e)
                
    def _measure_vibration(self):
        """Enhanced vibration measurement with signal processing pipeline"""
        if self.accel_client is None:
            return {'rms_vibration': 0.0, 'features': {}}
            
        # Data Acquisition: Get recent samples
        samples = self.accel_client.get_samples()
        if not samples:
            return {'rms_vibration': 0.0, 'features': {}}
            
        # Filter samples to recent time window
        current_time = time.time()
        recent_samples = []
        
        for sample in reversed(samples):
            sample_time, x, y, z = sample
            if current_time - sample_time > self.sample_time:
                break
            recent_samples.append((sample_time, x, y, z))
            
        if len(recent_samples) < 3:  # Need minimum samples for meaningful analysis
            return {'rms_vibration': 0.0, 'features': {}}
        
        # Signal Processing: Apply filters to remove noise
        filtered_samples = []
        for sample_time, x, y, z in reversed(recent_samples):  # Reverse to chronological order
            # Apply low-pass filter to each axis
            if self.filter_cutoff_frequency > 0:
                x_filtered = self._low_pass_filter(x, self.filter_state_x, 
                                                 self.filter_cutoff_frequency, self.sample_frequency)
                y_filtered = self._low_pass_filter(y, self.filter_state_y, 
                                                 self.filter_cutoff_frequency, self.sample_frequency)
                z_filtered = self._low_pass_filter(z, self.filter_state_z, 
                                                 self.filter_cutoff_frequency, self.sample_frequency)
            else:
                x_filtered, y_filtered, z_filtered = x, y, z
                
            filtered_samples.append((x_filtered, y_filtered, z_filtered))
        
        # Calculate mean acceleration to estimate static component (gravity)
        mean_x = sum(x for x, y, z in filtered_samples) / len(filtered_samples)
        mean_y = sum(y for x, y, z in filtered_samples) / len(filtered_samples)
        mean_z = sum(z for x, y, z in filtered_samples) / len(filtered_samples)
        
        # Calculate RMS of acceleration variations (vibrations)
        vibration_magnitudes = []
        total_variance = 0.0
        for x, y, z in filtered_samples:
            # Remove DC component (gravity/static acceleration)
            dx = x - mean_x
            dy = y - mean_y  
            dz = z - mean_z
            # Calculate magnitude of vibration vector
            vibration_magnitude = math.sqrt(dx*dx + dy*dy + dz*dz)
            vibration_magnitudes.append(vibration_magnitude)
            total_variance += vibration_magnitude * vibration_magnitude
            
        rms_vibration = math.sqrt(total_variance / len(filtered_samples))
        
        # Feature Extraction: Frequency analysis if enabled
        features = {}
        if self.enable_frequency_analysis and len(vibration_magnitudes) >= 8:
            features = self._extract_frequency_features(vibration_magnitudes, self.sample_frequency)
        
        return {'rms_vibration': rms_vibration, 'features': features}
        
    def _enhanced_decision_engine(self, vibration_data):
        """Enhanced decision engine based on vibration analysis and frequency features"""
        rms_vibration = vibration_data['rms_vibration']
        features = vibration_data.get('features', {})
        
        # Basic vibration threshold logic
        speed_change_reason = "stable"
        recommended_factor = 1.0
        
        if rms_vibration > self.vibration_threshold:
            speed_change_reason = "high_vibration"
            recommended_factor = self.speed_reduction_factor
            
            # Enhanced logic: Check for resonance frequencies
            if features.get('dominant_frequency', 0) > self.resonance_frequency_threshold:
                speed_change_reason = "resonance_detected"
                recommended_factor = self.speed_reduction_factor * 0.9  # More aggressive reduction
                
        elif rms_vibration < self.stable_threshold:
            speed_change_reason = "stable_conditions"
            recommended_factor = self.speed_increase_factor
            
            # Enhanced logic: Be more conservative if high frequency content
            if features.get('high_freq_ratio', 0) > 0.3:
                speed_change_reason = "stable_with_high_freq"
                recommended_factor = min(recommended_factor, 1.05)  # Smaller increase
        
        # Calculate new speed factor
        new_speed_factor = self.current_speed_factor * recommended_factor
        new_speed_factor = max(new_speed_factor, self.baseline_speed_factor * 0.3)  # Don't go below 30%
        new_speed_factor = min(new_speed_factor, self.baseline_speed_factor)  # Don't exceed baseline
        
        # Decision data for logging/simulation
        decision = {
            'timestamp': time.time(),
            'rms_vibration': rms_vibration,
            'features': features,
            'current_speed': self.current_speed_factor,
            'recommended_speed': new_speed_factor,
            'reason': speed_change_reason,
            'applied': False  # Will be set to True if actually applied
        }
        
        return decision
        
    def _adjust_speed_based_on_vibration(self, vibration_data):
        """Adjust print speed based on enhanced vibration analysis"""
        decision = self._enhanced_decision_engine(vibration_data)
        
        # Store decision for tracking
        with self.lock:
            self.simulation_decisions.append(decision)
            self.decision_count += 1
        
        # Apply decision unless in simulation mode
        if not self.simulation_mode:
            new_speed_factor = decision['recommended_speed']
            
            # Only adjust if significant change
            if abs(new_speed_factor - self.current_speed_factor) > 1.0:
                self._set_speed_factor(new_speed_factor)
                decision['applied'] = True
                
                logging.info("AccelerometerQuality: %s (vibration=%.1f), speed %.1f%% -> %.1f%%", 
                           decision['reason'], decision['rms_vibration'],
                           decision['current_speed'], new_speed_factor)
        else:
            # Simulation mode - just log what would have been done
            if abs(decision['recommended_speed'] - decision['current_speed']) > 1.0:
                logging.info("AccelerometerQuality: [SIMULATION] %s (vibration=%.1f), would change speed %.1f%% -> %.1f%%", 
                           decision['reason'], decision['rms_vibration'],
                           decision['current_speed'], decision['recommended_speed'])
                           
    def _set_speed_factor(self, speed_percentage):
        """Set the speed factor using M220 command"""
        try:
            gcode = self.printer.lookup_object('gcode')
            gcode.run_script("M220 S%.1f" % speed_percentage)
            self.current_speed_factor = speed_percentage
        except Exception as e:
            logging.error("AccelerometerQuality: Failed to set speed factor: %s", e)
            
    # G-code command handlers
    cmd_ACCELEROMETER_QUALITY_ENABLE_help = "Enable real-time quality control"
    def cmd_ACCELEROMETER_QUALITY_ENABLE(self, gcmd):
        if self.accel_chip is None:
            raise gcmd.error("Accelerometer chip '%s' not found" % self.accel_chip_name)
            
        if self.is_enabled and self.is_monitoring:
            gcmd.respond_info("AccelerometerQuality: Already enabled")
            return
            
        self.is_enabled = True
        if self._start_monitoring():
            gcmd.respond_info("AccelerometerQuality: Enabled real-time quality control")
        else:
            self.is_enabled = False
            raise gcmd.error("Failed to start accelerometer monitoring")
            
    cmd_ACCELEROMETER_QUALITY_DISABLE_help = "Disable real-time quality control"  
    def cmd_ACCELEROMETER_QUALITY_DISABLE(self, gcmd):
        self.is_enabled = False
        self._stop_monitoring()
        gcmd.respond_info("AccelerometerQuality: Disabled real-time quality control")
        
    cmd_ACCELEROMETER_QUALITY_STATUS_help = "Show quality control status"
    def cmd_ACCELEROMETER_QUALITY_STATUS(self, gcmd):
        status = "AccelerometerQuality Status:\n"
        status += "  Enabled: %s\n" % self.is_enabled
        status += "  Monitoring: %s\n" % self.is_monitoring
        status += "  Simulation Mode: %s\n" % self.simulation_mode
        status += "  Chip: %s\n" % self.accel_chip_name
        status += "  Current Speed Factor: %.1f%%\n" % self.current_speed_factor
        status += "  Baseline Speed Factor: %.1f%%\n" % self.baseline_speed_factor
        status += "  Vibration Threshold: %.1f\n" % self.vibration_threshold
        status += "  Stable Threshold: %.1f\n" % self.stable_threshold
        status += "  Frequency Analysis: %s\n" % self.enable_frequency_analysis
        status += "  Decisions Made: %d" % self.decision_count
        gcmd.respond_info(status)
        
    cmd_ACCELEROMETER_QUALITY_DECISIONS_help = "Show recent quality control decisions"  
    def cmd_ACCELEROMETER_QUALITY_DECISIONS(self, gcmd):
        count = gcmd.get_int('COUNT', 10, minval=1, maxval=100)
        
        with self.lock:
            recent_decisions = list(self.simulation_decisions)[-count:]
        
        if not recent_decisions:
            gcmd.respond_info("No decisions recorded yet")
            return
            
        response = "Recent Quality Control Decisions (last %d):\n" % len(recent_decisions)
        for i, decision in enumerate(recent_decisions):
            timestamp = decision['timestamp']
            time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
            
            response += "  %d. [%s] %s\n" % (i+1, time_str, decision['reason'])
            response += "     Vibration: %.1f, Speed: %.1f%% -> %.1f%%" % (
                decision['rms_vibration'], decision['current_speed'], decision['recommended_speed'])
            
            if decision.get('applied', False):
                response += " (APPLIED)"
            elif self.simulation_mode:
                response += " (SIMULATED)"
            else:
                response += " (NO CHANGE)"
                
            response += "\n"
            
            # Add frequency features if available
            features = decision.get('features', {})
            if features:
                response += "     Dominant Freq: %.1fHz, Spectral Centroid: %.1fHz\n" % (
                    features.get('dominant_frequency', 0), features.get('spectral_centroid', 0))
        
        gcmd.respond_info(response)
        
    cmd_ACCELEROMETER_QUALITY_SIMULATION_help = "Enable/disable simulation mode"
    def cmd_ACCELEROMETER_QUALITY_SIMULATION(self, gcmd):
        enable = gcmd.get_int('ENABLE', None)
        
        if enable is None:
            gcmd.respond_info("Simulation mode: %s" % self.simulation_mode)
            return
            
        old_mode = self.simulation_mode
        self.simulation_mode = bool(enable)
        
        if old_mode != self.simulation_mode:
            mode_str = "enabled" if self.simulation_mode else "disabled"
            gcmd.respond_info("Simulation mode %s" % mode_str)
            logging.info("AccelerometerQuality: Simulation mode %s", mode_str)
        
def load_config(config):
    return AccelerometerQuality(config)