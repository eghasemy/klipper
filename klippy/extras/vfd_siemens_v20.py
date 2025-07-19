# Siemens V20 VFD protocol implementation for Modbus spindle control
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging
from . import modbus_spindle

class SiemensV20Spindle(modbus_spindle.ModbusSpindle):
    def __init__(self, config):
        # Set protocol before calling parent init
        config.set('protocol', 'siemens_v20')
        super().__init__(config)
        
        # Siemens V20-specific registers
        self.speed_register = config.getint('speed_register', 0x0064)   # HSW - Speed setpoint
        self.control_register = config.getint('control_register', 0x0063) # STW1 - Control word
        self.status_register = config.getint('status_register', 0x006D)   # ZSW - Status word
        self.frequency_register = config.getint('frequency_register', 0x006E) # HIW - Actual speed
        
        # Siemens motor parameters for RPM calculation
        self.motor_poles = config.getint('motor_poles', 2, minval=2)
        self.motor_phases = config.getint('motor_phases', 3, minval=3)
        
        # Siemens frequency settings
        self.max_frequency = config.getfloat('max_frequency', 400.0)  # Hz
        self.min_frequency = config.getfloat('min_frequency', 0.0)    # Hz
        
        # Frequency scaler (V20 uses scaled input standardized to 16384)
        self.freq_scaler = 16384.0 / self.max_frequency
        
    def _setup_protocol_values(self):
        """Setup Siemens V20-specific control values"""
        # Siemens V20 control word values (STW1)
        self.run_cw = 0x0C7F   # Forward - ON
        self.run_ccw = 0x047F  # Reverse - ON  
        self.stop = 0x0C7E     # Forward - OFF (safe stop)
        
        # Speed calculation parameters
        self.speed_multiplier = 1  # Direct scaled values
        
    def _rpm_to_register_value(self, rpm):
        """Convert RPM to Siemens V20 scaled register value"""
        # Calculate frequency from RPM
        # RPM = (Frequency * (360 / Phases)) / Poles
        # So: Frequency = (RPM * Poles) / (360 / Phases)
        frequency = (rpm * self.motor_poles) / (360.0 / self.motor_phases)
        
        # Scale frequency to Siemens register value
        # V20 uses scaled input standardized to 16384 for max frequency
        scaled_value = int(frequency * self.freq_scaler)
        
        # Clamp to valid range (-16384 to +16384, but we only use positive)
        return min(max(scaled_value, 0), 16384)
        
    def _register_value_to_rpm(self, register_value):
        """Convert Siemens V20 scaled register value to RPM"""
        # Convert scaled value back to frequency
        if self.freq_scaler > 0:
            frequency = register_value / self.freq_scaler
            # Convert frequency to RPM
            rpm = (frequency * (360.0 / self.motor_phases)) / self.motor_poles
            return max(rpm, 0)
        return 0
        
    def _status_worker(self):
        """Siemens V20-specific status monitoring"""
        while self.status_running:
            try:
                if self.is_on:
                    # Read current speed from VFD (HIW register)
                    try:
                        registers = self.modbus.read_holding_registers(
                            self.frequency_register, 1)
                        if registers:
                            scaled_frequency = registers[0]
                            # Handle signed 16-bit value
                            if scaled_frequency > 32767:
                                scaled_frequency = scaled_frequency - 65536
                                
                            current_rpm = self._register_value_to_rpm(abs(scaled_frequency))
                            # Update current speed if significantly different
                            if abs(current_rpm - self.current_speed) > 50:
                                self.current_speed = current_rpm
                                logging.debug(f"Siemens V20 spindle speed: {current_rpm:.1f} RPM")
                                
                        # Read status word for error checking
                        status_registers = self.modbus.read_holding_registers(
                            self.status_register, 1)
                        if status_registers:
                            status = status_registers[0]
                            # Check for error conditions
                            if status & 0x0008:  # Drive fault active (bit 3)
                                logging.warning("Siemens V20 VFD reports drive fault")
                            if status & 0x0080:  # Drive warning active (bit 7)
                                logging.warning("Siemens V20 VFD reports warning")
                                
                    except Exception as e:
                        logging.debug(f"Failed to read Siemens V20 status: {e}")
                        
                import time
                time.sleep(self.status_interval)
                
            except Exception as e:
                logging.error(f"Siemens V20 status monitoring error: {e}")
                import time
                time.sleep(1.0)
                
    def _set_spindle_state(self, enable, direction=1, speed=0.):
        """Siemens V20-specific spindle control"""
        try:
            if enable and speed > 0:
                # Set speed first (scaled value)
                speed_value = self._rpm_to_register_value(speed)
                self.modbus.write_single_register(self.speed_register, speed_value)
                
                # Set control word for direction and run
                if direction > 0:
                    control_value = self.run_cw
                else:
                    control_value = self.run_ccw
                    
                self.modbus.write_single_register(self.control_register, control_value)
                
                # Update state
                self.is_on = True
                self.current_direction = direction
                self.target_speed = speed
                
                logging.info(f"Siemens V20 VFD: Set to {speed:.1f} RPM (scaled: {speed_value}), "
                           f"direction {'CW' if direction > 0 else 'CCW'}")
                
            else:
                # Stop spindle
                self.modbus.write_single_register(self.control_register, self.stop)
                
                # Update state
                self.is_on = False
                self.current_speed = 0.
                self.target_speed = 0.
                
                logging.info("Siemens V20 VFD: Stopped")
                
        except Exception as e:
            raise self.gcode.error(f"Siemens V20 VFD communication failed: {e}")
            
    def _handle_ready(self):
        """Initialize Siemens V20 VFD when Klipper is ready"""
        try:
            self.modbus.connect()
            
            # Test communication by reading status register
            try:
                registers = self.modbus.read_holding_registers(self.status_register, 1)
                if registers:
                    logging.info("Siemens V20 VFD communication established")
                    
                    # Calculate and set speed limits based on frequency range
                    if self.min_frequency > self.max_frequency:
                        self.min_frequency = self.max_frequency
                        
                    min_rpm = (self.min_frequency * (360.0 / self.motor_phases)) / self.motor_poles
                    max_rpm = (self.max_frequency * (360.0 / self.motor_phases)) / self.motor_poles
                    
                    self.min_rpm = max(min_rpm, 0)
                    self.max_rpm = max_rpm
                    
                    logging.info(f"Siemens V20 VFD: Speed range {self.min_rpm:.0f}-{self.max_rpm:.0f} RPM")
                    
                    self._start_status_monitoring()
                    logging.info(f"Siemens V20 spindle '{self.name}' ready")
                else:
                    logging.error("Failed to communicate with Siemens V20 VFD")
                    
            except Exception as e:
                logging.error(f"Failed to test Siemens V20 VFD communication: {e}")
                
        except Exception as e:
            logging.error(f"Failed to initialize Siemens V20 VFD spindle: {e}")

def load_config(config):
    return SiemensV20Spindle(config)

def load_config_prefix(config):
    return SiemensV20Spindle(config)