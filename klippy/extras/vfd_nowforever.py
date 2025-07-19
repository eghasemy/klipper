# NowForever VFD protocol implementation for Modbus spindle control
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging
from . import modbus_spindle

class NowForeverSpindle(modbus_spindle.ModbusSpindle):
    def __init__(self, config):
        # Set protocol before calling parent init
        config.set('protocol', 'nowforever')
        super().__init__(config)
        
        # NowForever-specific registers
        self.speed_register = config.getint('speed_register', 0x0901)   # Speed in Hz
        self.control_register = config.getint('control_register', 0x0900) # Control register
        self.status_register = config.getint('status_register', 0x0007)   # Max/min speed register
        self.frequency_register = config.getint('frequency_register', 0x0502) # Current frequency
        self.fault_register = config.getint('fault_register', 0x0300)     # Current fault register
        
        # NowForever frequency settings (will be read from VFD)
        self.min_frequency = 100   # Default min frequency (Hz * 100)
        self.max_frequency = 24000 # Default max frequency (Hz * 100)
        
    def _setup_protocol_values(self):
        """Setup NowForever-specific control values"""
        # NowForever control register bit values
        self.run_cw = 0x01   # Bit 0: run=1, Bit 1: direction=0 (CW)
        self.run_ccw = 0x03  # Bit 0: run=1, Bit 1: direction=1 (CCW)  
        self.stop = 0x00     # Bit 0: run=0
        
        # Speed calculation parameters
        self.speed_multiplier = 100  # Hz * 100 format
        
    def _rpm_to_register_value(self, rpm):
        """Convert RPM to NowForever frequency register value (Hz * 100)"""
        # Convert RPM to frequency: freq_Hz = RPM / 60 (for 2-pole motor)
        # NowForever uses Hz * 100 format
        frequency_hz = rpm / 60.0
        frequency_x100 = int(frequency_hz * 100)
        
        # Clamp to min/max frequency
        return min(max(frequency_x100, self.min_frequency), self.max_frequency)
        
    def _register_value_to_rpm(self, register_value):
        """Convert NowForever frequency register value (Hz * 100) to RPM"""
        # Convert Hz * 100 to Hz, then to RPM (assume 2-pole motor)
        frequency_hz = register_value / 100.0
        rpm = frequency_hz * 60.0
        return rpm
        
    def _initialize_vfd(self):
        """Initialize NowForever VFD by reading frequency limits"""
        try:
            # Read max and min frequency (registers 0x0007 and 0x0008)
            registers = self.modbus.read_holding_registers(self.status_register, 2)
            if registers and len(registers) >= 2:
                self.max_frequency = registers[0]  # Max speed in Hz * 100
                self.min_frequency = registers[1]  # Min speed in Hz * 100
                
                logging.debug(f"NowForever VFD: Max frequency = {self.max_frequency} (Hz*100), "
                             f"Min frequency = {self.min_frequency} (Hz*100)")
                
                # Calculate RPM limits
                min_rpm = self._register_value_to_rpm(self.min_frequency)
                max_rpm = self._register_value_to_rpm(self.max_frequency)
                
                self.min_rpm = min_rpm
                self.max_rpm = max_rpm
                
                logging.info(f"NowForever VFD: Speed range {min_rpm:.0f}-{max_rpm:.0f} RPM")
                
                return True
            else:
                logging.error("Failed to read NowForever VFD frequency limits")
                return False
                
        except Exception as e:
            logging.error(f"Failed to initialize NowForever VFD: {e}")
            return False
            
    def _check_fault_status(self):
        """Check NowForever VFD fault status"""
        try:
            registers = self.modbus.read_holding_registers(self.fault_register, 1)
            if registers:
                fault_number = registers[0]
                if fault_number != 0:
                    logging.warning(f"NowForever VFD fault: {fault_number}")
                    return False
            return True
        except Exception as e:
            logging.debug(f"Failed to read NowForever fault status: {e}")
            return True  # Assume OK if can't read
            
    def _status_worker(self):
        """NowForever-specific status monitoring"""
        while self.status_running:
            try:
                if self.is_on:
                    # Read current output frequency from VFD
                    try:
                        registers = self.modbus.read_holding_registers(
                            self.frequency_register, 1)
                        if registers:
                            frequency_x100 = registers[0]
                            current_rpm = self._register_value_to_rpm(frequency_x100)
                            # Update current speed if significantly different
                            if abs(current_rpm - self.current_speed) > 50:
                                self.current_speed = current_rpm
                                logging.debug(f"NowForever spindle speed: {current_rpm:.1f} RPM "
                                            f"({frequency_x100/100:.1f} Hz)")
                                
                        # Check for faults
                        self._check_fault_status()
                        
                    except Exception as e:
                        logging.debug(f"Failed to read NowForever status: {e}")
                        
                import time
                time.sleep(self.status_interval)
                
            except Exception as e:
                logging.error(f"NowForever status monitoring error: {e}")
                import time
                time.sleep(1.0)
                
    def _write_multiple_registers(self, start_register, values):
        """Write multiple registers using function code 0x10"""
        data = bytearray()
        data.extend([(start_register >> 8) & 0xFF, start_register & 0xFF])  # Start register
        data.extend([0x00, len(values)])  # Number of registers
        data.extend([len(values) * 2])    # Byte count
        
        for value in values:
            data.extend([(value >> 8) & 0xFF, value & 0xFF])
            
        return self.modbus.send_command(0x10, data)
        
    def _set_spindle_state(self, enable, direction=1, speed=0.):
        """NowForever-specific spindle control"""
        try:
            if enable and speed > 0:
                # NowForever requires writing speed and control together
                frequency_value = self._rpm_to_register_value(speed)
                
                # Determine control value
                if direction > 0:
                    control_value = self.run_cw
                else:
                    control_value = self.run_ccw
                
                # Write both registers using multiple register write
                values = [control_value, frequency_value]
                response = self._write_multiple_registers(self.control_register, values)
                
                if not response:
                    raise Exception("No response from NowForever VFD")
                
                # Update state
                self.is_on = True
                self.current_direction = direction
                self.target_speed = speed
                
                logging.info(f"NowForever VFD: Set to {speed:.1f} RPM ({frequency_value/100:.1f} Hz), "
                           f"direction {'CW' if direction > 0 else 'CCW'}")
                
            else:
                # Stop spindle
                values = [self.stop, 0]
                response = self._write_multiple_registers(self.control_register, values)
                
                if not response:
                    raise Exception("No response from NowForever VFD")
                
                # Update state
                self.is_on = False
                self.current_speed = 0.
                self.target_speed = 0.
                
                logging.info("NowForever VFD: Stopped")
                
        except Exception as e:
            raise self.gcode.error(f"NowForever VFD communication failed: {e}")
            
    def _handle_ready(self):
        """Initialize NowForever VFD when Klipper is ready"""
        try:
            self.modbus.connect()
            if self._initialize_vfd():
                self._start_status_monitoring()
                logging.info(f"NowForever VFD spindle '{self.name}' ready")
            else:
                logging.error("Failed to initialize NowForever VFD")
        except Exception as e:
            logging.error(f"Failed to initialize NowForever VFD spindle: {e}")

def load_config(config):
    return NowForeverSpindle(config)

def load_config_prefix(config):
    return NowForeverSpindle(config)