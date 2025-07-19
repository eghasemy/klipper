# Danfoss VLT2800 VFD protocol implementation for Modbus spindle control
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging, time
from . import modbus_spindle

class DanfossVLT2800Spindle(modbus_spindle.ModbusSpindle):
    def __init__(self, config):
        # Set protocol before calling parent init
        config.set('protocol', 'danfoss_vlt2800')
        super().__init__(config)
        
        # Danfoss VLT2800-specific registers
        self.speed_register = config.getint('speed_register', 0x0000)   # Control coil start
        self.control_register = config.getint('control_register', 0x0000) # Control coil start  
        self.status_register = config.getint('status_register', 0x0020)   # Status coil start
        self.frequency_register = config.getint('frequency_register', 0x143B) # Current frequency
        
        # Danfoss frequency settings
        self.max_frequency = config.getfloat('max_frequency', 400.0)  # Hz
        
        # Cached spindle state for Danfoss (needs full state on every command)
        self.cached_state = {
            'mode': None,
            'speed': 0
        }
        
    def _setup_protocol_values(self):
        """Setup Danfoss VLT2800-specific control values"""
        # Danfoss uses coil control, not simple register values
        # Control is done via multiple coils in control word
        pass
        
    def _rpm_to_register_value(self, rpm):
        """Convert RPM to Danfoss frequency register value"""
        # Calculate frequency based on motor parameters
        # Assume max_frequency corresponds to max_rpm
        if self.max_rpm > 0:
            frequency = (rpm / self.max_rpm) * self.max_frequency
            return int(frequency)
        return 0
        
    def _build_control_word(self, mode, speed):
        """Build Danfoss control word with all required bits"""
        # Danfoss VLT2800 control word structure
        control_word = 0x0000
        
        # Base control bits (always set for normal operation)
        control_word |= (1 << 1)   # OFF2: Electrical stop (enabled)
        control_word |= (1 << 2)   # OFF3: Fast stop (enabled)
        control_word |= (1 << 3)   # Pulse enabled
        control_word |= (1 << 4)   # RFG enabled  
        control_word |= (1 << 5)   # RFG start
        control_word |= (1 << 10)  # Controller of AG
        
        # Mode-specific control
        if mode == 'cw':
            control_word |= (1 << 0)   # ON/OFF1 (start)
            control_word |= (1 << 6)   # Enable setpoint
            control_word |= (1 << 11)  # Reversing (CW = 1)
        elif mode == 'ccw':
            control_word |= (1 << 0)   # ON/OFF1 (start)
            control_word |= (1 << 6)   # Enable setpoint
            control_word &= ~(1 << 11) # Reversing (CCW = 0)
        else:  # stop
            control_word &= ~(1 << 0)  # ON/OFF1 (stop)
            control_word &= ~(1 << 6)  # Disable setpoint
            
        return control_word
        
    def _write_vfd_state(self, mode, speed):
        """Write combined state to Danfoss VFD"""
        try:
            # Build control word
            control_word = self._build_control_word(mode, speed)
            
            # Convert speed to frequency
            frequency = self._rpm_to_register_value(speed) if mode != 'stop' else 0
            
            # Write multiple coils (function code 0x0F)
            # This is a complex operation for Danfoss - we write control word + frequency
            # For simplicity, we'll use multiple single writes
            
            # Write control word as coils (16 bits = 2 bytes)
            data = bytearray()
            data.extend([0x00, 0x00])  # Start coil address (0x0000)
            data.extend([0x00, 0x20])  # Number of coils (32)
            data.extend([0x04])        # Byte count (4 bytes = 32 bits)
            data.extend([control_word & 0xFF, (control_word >> 8) & 0xFF])  # Control word
            data.extend([frequency & 0xFF, (frequency >> 8) & 0xFF])        # Frequency
            
            response = self.modbus.send_command(0x0F, data)  # Write multiple coils
            
            if not response:
                raise Exception("No response from Danfoss VFD")
                
            # Update cached state
            self.cached_state['mode'] = mode
            self.cached_state['speed'] = speed
            
        except Exception as e:
            raise Exception(f"Danfoss VFD communication failed: {e}")
            
    def _status_worker(self):
        """Danfoss-specific status monitoring"""
        while self.status_running:
            try:
                if self.is_on:
                    # Read current frequency from VFD
                    try:
                        registers = self.modbus.read_holding_registers(
                            self.frequency_register, 1)
                        if registers:
                            frequency = registers[0]
                            # Convert frequency to RPM
                            if self.max_frequency > 0:
                                current_rpm = (frequency / self.max_frequency) * self.max_rpm
                                # Update current speed if significantly different
                                if abs(current_rpm - self.current_speed) > 50:
                                    self.current_speed = current_rpm
                                    logging.debug(f"Danfoss spindle speed: {current_rpm:.1f} RPM")
                                    
                        # Read status coils for error checking
                        status_response = self.modbus.send_command(0x01, 
                            bytearray([0x00, 0x20, 0x00, 0x10]))  # Read 16 status coils
                        if status_response and len(status_response) >= 5:
                            status_word = status_response[3] | (status_response[4] << 8)
                            # Check for error conditions (simplified)
                            if status_word & 0x0008:  # Trip status bit
                                logging.warning("Danfoss VFD reports trip condition")
                                
                    except Exception as e:
                        logging.debug(f"Failed to read Danfoss status: {e}")
                        
                time.sleep(self.status_interval)
                
            except Exception as e:
                logging.error(f"Danfoss status monitoring error: {e}")
                time.sleep(1.0)
                
    def _set_spindle_state(self, enable, direction=1, speed=0.):
        """Danfoss-specific spindle control"""
        try:
            if enable and speed > 0:
                # Determine mode
                mode = 'cw' if direction > 0 else 'ccw'
                
                # Write combined state
                self._write_vfd_state(mode, speed)
                
                # Update state
                self.is_on = True
                self.current_direction = direction
                self.target_speed = speed
                
                logging.info(f"Danfoss VFD: Set to {speed:.1f} RPM, "
                           f"direction {'CW' if direction > 0 else 'CCW'}")
                
            else:
                # Stop spindle
                self._write_vfd_state('stop', 0)
                
                # Update state
                self.is_on = False
                self.current_speed = 0.
                self.target_speed = 0.
                
                logging.info("Danfoss VFD: Stopped")
                
        except Exception as e:
            raise self.gcode.error(f"Danfoss VFD communication failed: {e}")
            
    def _handle_ready(self):
        """Initialize Danfoss VFD when Klipper is ready"""
        try:
            self.modbus.connect()
            
            # Test VFD communication by reading status
            try:
                status_response = self.modbus.send_command(0x01, 
                    bytearray([0x00, 0x20, 0x00, 0x10]))  # Read 16 status coils
                if status_response:
                    logging.info("Danfoss VFD communication established")
                    self._start_status_monitoring()
                    logging.info(f"Danfoss VLT2800 spindle '{self.name}' ready")
                else:
                    logging.error("Failed to communicate with Danfoss VFD")
            except Exception as e:
                logging.error(f"Failed to test Danfoss VFD communication: {e}")
                
        except Exception as e:
            logging.error(f"Failed to initialize Danfoss VFD spindle: {e}")

def load_config(config):
    return DanfossVLT2800Spindle(config)

def load_config_prefix(config):
    return DanfossVLT2800Spindle(config)