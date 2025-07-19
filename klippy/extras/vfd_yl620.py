# YL620 VFD protocol implementation for Modbus spindle control
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging
from . import modbus_spindle

class YL620Spindle(modbus_spindle.ModbusSpindle):
    def __init__(self, config):
        # Set protocol before calling parent init
        config.set('protocol', 'yl620')
        super().__init__(config)
        
        # YL620-specific registers
        self.speed_register = config.getint('speed_register', 0x2001)   # Frequency command (x0.1Hz)
        self.control_register = config.getint('control_register', 0x2000) # Command register
        self.status_register = config.getint('status_register', 0x0308)   # Frequency lower limit
        self.frequency_register = config.getint('frequency_register', 0x200B) # Output frequency
        self.max_freq_register = config.getint('max_freq_register', 0x0000)   # Main frequency
        
        # YL620 frequency settings (will be read from VFD)
        self.min_frequency = 100   # Default min frequency (deciHz)
        self.max_frequency = 4000  # Default max frequency (deciHz) 
        
    def _setup_protocol_values(self):
        """Setup YL620-specific control values"""
        # YL620 command register values
        self.run_cw = 0x12   # Start in forward direction
        self.run_ccw = 0x22  # Start in reverse direction  
        self.stop = 0x01     # Disable spindle
        
        # Speed calculation parameters  
        self.speed_multiplier = 1  # Direct deciHz values
        
    def _rpm_to_register_value(self, rpm):
        """Convert RPM to YL620 frequency register value (deciHz)"""
        # Convert RPM to frequency: freq_Hz = RPM / 60 (for 2-pole motor)
        # YL620 uses deciHz (0.1 Hz), so multiply by 10
        # Assume 2-pole motor: RPM = freq * 60
        frequency_hz = rpm / 60.0
        frequency_decihz = int(frequency_hz * 10)
        
        # Clamp to min/max frequency
        return min(max(frequency_decihz, self.min_frequency), self.max_frequency)
        
    def _register_value_to_rpm(self, register_value):
        """Convert YL620 frequency register value (deciHz) to RPM"""
        # Convert deciHz to Hz, then to RPM (assume 2-pole motor)
        frequency_hz = register_value / 10.0
        rpm = frequency_hz * 60.0
        return rpm
        
    def _initialize_vfd(self):
        """Initialize YL620 VFD by reading frequency limits"""
        try:
            # Read minimum frequency (P03.08)
            min_registers = self.modbus.read_holding_registers(self.status_register, 1)
            if min_registers:
                self.min_frequency = min_registers[0]
                logging.debug(f"YL620 VFD: Min frequency = {self.min_frequency} deciHz")
                
            # Read maximum frequency (P00.00)  
            max_registers = self.modbus.read_holding_registers(self.max_freq_register, 1)
            if max_registers:
                self.max_frequency = max_registers[0]
                logging.debug(f"YL620 VFD: Max frequency = {self.max_frequency} deciHz")
                
            # Calculate RPM limits (frequency is in deciHz, RPM = freq * 6)
            min_rpm = self.min_frequency * 6  # deciHz * 6 = RPM
            max_rpm = self.max_frequency * 6
            
            self.min_rpm = min_rpm
            self.max_rpm = max_rpm
            
            logging.info(f"YL620 VFD: Frequency range {self.min_frequency}-{self.max_frequency} deciHz, "
                        f"RPM range {min_rpm}-{max_rpm}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize YL620 VFD: {e}")
            return False
            
    def _status_worker(self):
        """YL620-specific status monitoring"""
        while self.status_running:
            try:
                if self.is_on:
                    # Read current output frequency from VFD
                    try:
                        registers = self.modbus.read_holding_registers(
                            self.frequency_register, 1)
                        if registers:
                            frequency_decihz = registers[0]
                            current_rpm = self._register_value_to_rpm(frequency_decihz)
                            # Update current speed if significantly different
                            if abs(current_rpm - self.current_speed) > 50:
                                self.current_speed = current_rpm
                                logging.debug(f"YL620 spindle speed: {current_rpm:.1f} RPM "
                                            f"({frequency_decihz} deciHz)")
                                
                    except Exception as e:
                        logging.debug(f"Failed to read YL620 status: {e}")
                        
                import time
                time.sleep(self.status_interval)
                
            except Exception as e:
                logging.error(f"YL620 status monitoring error: {e}")
                import time
                time.sleep(1.0)
                
    def _set_spindle_state(self, enable, direction=1, speed=0.):
        """YL620-specific spindle control"""
        try:
            if enable and speed > 0:
                # Set frequency first
                frequency_value = self._rpm_to_register_value(speed)
                self.modbus.write_single_register(self.speed_register, frequency_value)
                
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
                
                logging.info(f"YL620 VFD: Set to {speed:.1f} RPM ({frequency_value} deciHz), "
                           f"direction {'CW' if direction > 0 else 'CCW'}")
                
            else:
                # Stop spindle
                self.modbus.write_single_register(self.control_register, self.stop)
                
                # Update state
                self.is_on = False
                self.current_speed = 0.
                self.target_speed = 0.
                
                logging.info("YL620 VFD: Stopped")
                
        except Exception as e:
            raise self.gcode.error(f"YL620 VFD communication failed: {e}")
            
    def _handle_ready(self):
        """Initialize YL620 VFD when Klipper is ready"""
        try:
            self.modbus.connect()
            if self._initialize_vfd():
                self._start_status_monitoring()
                logging.info(f"YL620 VFD spindle '{self.name}' ready")
            else:
                logging.error("Failed to initialize YL620 VFD")
        except Exception as e:
            logging.error(f"Failed to initialize YL620 VFD spindle: {e}")

def load_config(config):
    return YL620Spindle(config)

def load_config_prefix(config):
    return YL620Spindle(config)