# Huanyang VFD protocol implementation for Modbus spindle control
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging, time
from . import modbus_spindle

class HuanyangSpindle(modbus_spindle.ModbusSpindle):
    def __init__(self, config):
        # Set protocol before calling parent init
        config.set('protocol', 'huanyang')
        super().__init__(config)
        
        # Huanyang-specific registers (common defaults)
        self.speed_register = config.getint('speed_register', 0x1000)
        self.control_register = config.getint('control_register', 0x1001) 
        self.status_register = config.getint('status_register', 0x2000)
        self.frequency_register = config.getint('frequency_register', 0x2001)
        
        # Huanyang frequency settings
        self.rated_frequency = config.getfloat('rated_frequency', 50.0)  # Hz
        self.max_frequency = config.getfloat('max_frequency', 100.0)     # Hz
        
    def _setup_protocol_values(self):
        """Setup Huanyang-specific control values"""
        # Huanyang control register values
        self.run_cw = 0x0001   # Forward rotation
        self.run_ccw = 0x0011  # Reverse rotation  
        self.stop = 0x0008     # Stop
        
        # Speed calculation parameters
        self.speed_multiplier = 100  # 0.01Hz resolution
        
    def _rpm_to_register_value(self, rpm):
        """Convert RPM to Huanyang frequency register value"""
        # Calculate frequency based on rated motor parameters
        # Assume rated_frequency corresponds to max_rpm
        if self.max_rpm > 0:
            frequency = (rpm / self.max_rpm) * self.rated_frequency
            # Limit to max frequency
            frequency = min(frequency, self.max_frequency)
            # Convert to register value (0.01Hz resolution)
            return int(frequency * 100)
        return 0
        
    def _register_value_to_rpm(self, register_value):
        """Convert Huanyang frequency register value to RPM"""
        frequency = register_value / 100.0  # Convert from 0.01Hz to Hz
        if self.rated_frequency > 0:
            rpm = (frequency / self.rated_frequency) * self.max_rpm
            return rpm
        return 0
        
    def _status_worker(self):
        """Huanyang-specific status monitoring"""
        while self.status_running:
            try:
                if self.is_on:
                    # Read current frequency from VFD
                    try:
                        registers = self.modbus.read_holding_registers(
                            self.frequency_register, 1)
                        if registers:
                            current_rpm = self._register_value_to_rpm(registers[0])
                            # Update current speed if significantly different
                            if abs(current_rpm - self.current_speed) > 50:
                                self.current_speed = current_rpm
                                logging.debug(f"Huanyang spindle speed: {current_rpm:.1f} RPM")
                                
                        # Also read status register for error checking
                        status_registers = self.modbus.read_holding_registers(
                            self.status_register, 1)
                        if status_registers:
                            status = status_registers[0]
                            # Check for error conditions (bit pattern depends on VFD model)
                            if status & 0x0008:  # Common error bit
                                logging.warning("Huanyang VFD reports error condition")
                                
                    except Exception as e:
                        logging.debug(f"Failed to read Huanyang status: {e}")
                        
                time.sleep(self.status_interval)
                
            except Exception as e:
                logging.error(f"Huanyang status monitoring error: {e}")
                time.sleep(1.0)
                
    def _set_spindle_state(self, enable, direction=1, speed=0.):
        """Huanyang-specific spindle control"""
        try:
            if enable and speed > 0:
                # For Huanyang VFDs, sometimes need to stop first before changing direction
                if self.is_on and direction != self.current_direction:
                    self.modbus.write_single_register(self.control_register, self.stop)
                    time.sleep(0.1)  # Brief pause
                    
                # Set frequency/speed
                speed_value = self._rpm_to_register_value(speed)
                self.modbus.write_single_register(self.speed_register, speed_value)
                
                # Set control register for direction and run
                if direction > 0:
                    control_value = self.run_cw
                else:
                    control_value = self.run_ccw
                    
                self.modbus.write_single_register(self.control_register, control_value)
                
                # Update state
                self.is_on = True
                self.current_direction = direction
                self.target_speed = speed
                
                logging.info(f"Huanyang VFD: Set to {speed:.1f} RPM, "
                           f"direction {'CW' if direction > 0 else 'CCW'}")
                
            else:
                # Stop spindle
                self.modbus.write_single_register(self.control_register, self.stop)
                
                # Update state
                self.is_on = False
                self.current_speed = 0.
                self.target_speed = 0.
                
                logging.info("Huanyang VFD: Stopped")
                
        except Exception as e:
            raise self.gcode.error(f"Huanyang VFD communication failed: {e}")

def load_config(config):
    return HuanyangSpindle(config)

def load_config_prefix(config):
    return HuanyangSpindle(config)