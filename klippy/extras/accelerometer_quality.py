# Real-time accelerometer-based print quality control
#
# Copyright (C) 2025  Klipper Contributors
#
# This file may be distributed under the terms of the GNU GPLv3 license.

import logging, threading, time, math

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
        
        # State variables
        self.is_enabled = False
        self.is_monitoring = False
        self.monitor_thread = None
        self.current_speed_factor = 100.0  # Percentage
        self.baseline_speed_factor = 100.0
        self.accel_chip = None
        self.accel_client = None
        self.lock = threading.Lock()
        
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
        try:
            while True:
                with self.lock:
                    if not self.is_monitoring:
                        break
                
                # Sample accelerometer data
                try:
                    vibration_magnitude = self._measure_vibration()
                    self._adjust_speed_based_on_vibration(vibration_magnitude)
                except Exception as e:
                    logging.warning("AccelerometerQuality: Error in monitoring loop: %s", e)
                    
                # Sleep until next sample
                time.sleep(self.sample_time)
                
        except Exception as e:
            logging.error("AccelerometerQuality: Monitor loop error: %s", e)
        finally:
            with self.lock:
                self.is_monitoring = False
                
    def _measure_vibration(self):
        """Measure current vibration magnitude from accelerometer"""
        if self.accel_client is None:
            return 0.0
            
        # Get recent samples
        samples = self.accel_client.get_samples()
        if not samples:
            return 0.0
            
        # Calculate RMS vibration magnitude from recent samples
        # Use last N samples or samples from last sample_time period
        recent_samples = []
        current_time = time.time()
        
        for sample in reversed(samples):
            sample_time, x, y, z = sample
            if current_time - sample_time > self.sample_time:
                break
            recent_samples.append((x, y, z))
            
        if not recent_samples:
            return 0.0
            
        # Calculate RMS magnitude (remove gravity component roughly)
        total_magnitude = 0.0
        for x, y, z in recent_samples:
            # Calculate magnitude without gravity (rough approximation)
            magnitude = math.sqrt(x*x + y*y + (z-9806.65)*(z-9806.65))  # z gravity ~9.8 m/s^2 in mm/s^2
            total_magnitude += magnitude * magnitude
            
        rms_magnitude = math.sqrt(total_magnitude / len(recent_samples))
        return rms_magnitude
        
    def _adjust_speed_based_on_vibration(self, vibration_magnitude):
        """Adjust print speed based on measured vibration"""
        if vibration_magnitude > self.vibration_threshold:
            # High vibration - reduce speed
            new_speed_factor = self.current_speed_factor * self.speed_reduction_factor
            new_speed_factor = max(new_speed_factor, self.baseline_speed_factor * 0.3)  # Don't go below 30%
            
            if abs(new_speed_factor - self.current_speed_factor) > 1.0:  # Only adjust if significant change
                self._set_speed_factor(new_speed_factor)
                logging.info("AccelerometerQuality: High vibration (%.1f), reducing speed to %.1f%%", 
                           vibration_magnitude, new_speed_factor)
                           
        elif vibration_magnitude < self.stable_threshold:
            # Low vibration - can increase speed
            new_speed_factor = min(self.current_speed_factor * self.speed_increase_factor, 
                                 self.baseline_speed_factor)
            
            if new_speed_factor > self.current_speed_factor + 1.0:  # Only adjust if significant change
                self._set_speed_factor(new_speed_factor)
                logging.info("AccelerometerQuality: Stable conditions (%.1f), increasing speed to %.1f%%",
                           vibration_magnitude, new_speed_factor)
                           
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
        status += "  Chip: %s\n" % self.accel_chip_name
        status += "  Current Speed Factor: %.1f%%\n" % self.current_speed_factor
        status += "  Baseline Speed Factor: %.1f%%\n" % self.baseline_speed_factor
        status += "  Vibration Threshold: %.1f\n" % self.vibration_threshold
        status += "  Stable Threshold: %.1f" % self.stable_threshold
        gcmd.respond_info(status)
        
def load_config(config):
    return AccelerometerQuality(config)