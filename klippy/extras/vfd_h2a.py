# H2A VFD protocol implementation for Modbus spindle control
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging
from . import modbus_spindle

class H2AVFDSpindle(modbus_spindle.ModbusSpindle):
    def __init__(self, config):
        # Set protocol before calling parent init
        config.set('protocol', 'h2a')
        super().__init__(config)
        
        # H2A-specific registers  
        self.speed_register = config.getint('speed_register', 0x1000)   # Speed in percentage (0-10000 = 0-100%)
        self.control_register = config.getint('control_register', 0x2000) # Control word
        self.status_register = config.getint('status_register', 0xB005)   # Max RPM register
        self.frequency_register = config.getint('frequency_register', 0x700C) # Current speed register
        
        # H2A max RPM (will be read from VFD)
        self.max_vfd_rpm = 24000  # Default, will be updated from VFD
        
    def _setup_protocol_values(self):
        """Setup H2A-specific control values"""
        # H2A control register values
        self.run_cw = 0x0001   # Forward rotation
        self.run_ccw = 0x0002  # Reverse rotation  
        self.stop = 0x0006     # Stop
        
        # Speed calculation parameters
        self.speed_multiplier = 1  # Direct percentage values
        
    def _rpm_to_register_value(self, rpm):
        """Convert RPM to H2A percentage register value"""
        # H2A expects speed as percentage (0-10000 = 0-100%)
        if self.max_vfd_rpm > 0:
            percentage = int((rpm / self.max_vfd_rpm) * 10000)
            return min(max(percentage, 0), 10000)  # Clamp to 0-10000
        return 0
        
    def _register_value_to_rpm(self, register_value):
        """Convert H2A register value to RPM"""
        # Direct RPM value from H2A VFD
        return register_value
        
    def _initialize_vfd(self):
        """Initialize H2A VFD by reading max RPM"""
        try:
            # Read max RPM from register 0xB005 (2 registers)
            registers = self.modbus.read_holding_registers(self.status_register, 2)
            if registers and len(registers) >= 2:
                # Max RPM is in the first register
                self.max_vfd_rpm = registers[0]
                logging.info(f"H2A VFD: Max RPM = {self.max_vfd_rpm}")
                
                # Update speed limits
                if self.max_vfd_rpm > 0:
                    # Set minimum to 25% of max RPM  
                    min_rpm = self.max_vfd_rpm // 4
                    self.min_rpm = min_rpm
                    self.max_rpm = self.max_vfd_rpm
                    
            return True
        except Exception as e:
            logging.error(f"Failed to initialize H2A VFD: {e}")
            return False
            
    def _status_worker(self):
        """H2A-specific status monitoring"""
        while self.status_running:
            try:
                if self.is_on:
                    # Read current speed from VFD (register 0x700C, 2 registers)
                    try:
                        registers = self.modbus.read_holding_registers(
                            self.frequency_register, 2)
                        if registers and len(registers) >= 2:
                            # Current RPM is in the first register
                            current_rpm = registers[0]
                            # Update current speed if significantly different
                            if abs(current_rpm - self.current_speed) > 50:
                                self.current_speed = current_rpm
                                logging.debug(f"H2A spindle speed: {current_rpm:.1f} RPM")
                                
                    except Exception as e:
                        logging.debug(f"Failed to read H2A status: {e}")
                        
                import time
                time.sleep(self.status_interval)
                
            except Exception as e:
                logging.error(f"H2A status monitoring error: {e}")
                import time
                time.sleep(1.0)
                
    def _set_spindle_state(self, enable, direction=1, speed=0.):
        """H2A-specific spindle control"""
        try:
            if enable and speed > 0:
                # Set speed first (as percentage 0-10000)
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
                
                logging.info(f"H2A VFD: Set to {speed:.1f} RPM ({speed_value/100:.1f}%), "
                           f"direction {'CW' if direction > 0 else 'CCW'}")
                
            else:
                # Stop spindle
                self.modbus.write_single_register(self.control_register, self.stop)
                
                # Update state
                self.is_on = False
                self.current_speed = 0.
                self.target_speed = 0.
                
                logging.info("H2A VFD: Stopped")
                
        except Exception as e:
            raise self.gcode.error(f"H2A VFD communication failed: {e}")
            
    def _handle_ready(self):
        """Initialize H2A VFD when Klipper is ready"""
        try:
            self.modbus.connect()
            if self._initialize_vfd():
                self._start_status_monitoring()
                logging.info(f"H2A VFD spindle '{self.name}' ready")
            else:
                logging.error("Failed to initialize H2A VFD")
        except Exception as e:
            logging.error(f"Failed to initialize H2A VFD spindle: {e}")

def load_config(config):
    return H2AVFDSpindle(config)

def load_config_prefix(config):
    return H2AVFDSpindle(config)