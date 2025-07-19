# Modbus VFD spindle control support for CNC machines
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging, threading, time
from . import spindle

class ModbusSpindle:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object('gcode')
        self.name = config.get_name().split()[-1]
        
        # Modbus configuration
        self.modbus = None
        modbus_config = config.getsection('modbus')
        if modbus_config:
            self.modbus = self.printer.load_object(modbus_config, 'modbus')
        else:
            raise config.error("Modbus spindle requires [modbus] section")
            
        # VFD Protocol selection
        self.protocol = config.get('protocol', 'generic').lower()
        
        # Speed mapping configuration
        self.min_rpm = config.getfloat('min_rpm', 0., minval=0.)
        self.max_rpm = config.getfloat('max_rpm', 24000., above=self.min_rpm)
        
        # Modbus register addresses (configurable for different VFDs)
        self.speed_register = config.getint('speed_register', 0x2000)
        self.control_register = config.getint('control_register', 0x2001)
        self.status_register = config.getint('status_register', 0x3000)
        self.frequency_register = config.getint('frequency_register', 0x3001)
        
        # Control values for different VFD protocols
        self._setup_protocol_values()
        
        # Current state
        self.is_on = False
        self.current_speed = 0.
        self.current_direction = 1  # 1 = CW (M3), -1 = CCW (M4)
        self.target_speed = 0.
        
        # Status monitoring
        self.status_thread = None
        self.status_running = False
        self.status_interval = config.getfloat('status_interval', 1.0, minval=0.1)
        
        # Register G-code commands
        self.gcode.register_command('M3', self.cmd_M3, 
                                  desc=self.cmd_M3_help)
        self.gcode.register_command('M4', self.cmd_M4,
                                  desc=self.cmd_M4_help)
        self.gcode.register_command('M5', self.cmd_M5,
                                  desc=self.cmd_M5_help)
        
        # Register for shutdown
        self.printer.register_event_handler('klippy:ready', self._handle_ready)
        self.printer.register_event_handler('klippy:shutdown', self._shutdown)
        
    def _setup_protocol_values(self):
        """Setup protocol-specific control values"""
        if self.protocol == 'huanyang':
            # Huanyang VFD values
            self.run_cw = 0x0001
            self.run_ccw = 0x0011
            self.stop = 0x0008
            self.speed_multiplier = 100  # RPM to register value multiplier
        elif self.protocol == 'h100':
            # H100 series VFD values
            self.run_cw = 0x0001
            self.run_ccw = 0x0002
            self.stop = 0x0000
            self.speed_multiplier = 1
        elif self.protocol == 'h2a':
            # H2A series VFD values
            self.run_cw = 0x0001
            self.run_ccw = 0x0002
            self.stop = 0x0006
            self.speed_multiplier = 1
        elif self.protocol == 'danfoss_vlt2800':
            # Danfoss VLT2800 VFD uses complex control word
            self.run_cw = 0x0C7F
            self.run_ccw = 0x047F
            self.stop = 0x0C7E
            self.speed_multiplier = 1
        elif self.protocol == 'siemens_v20':
            # Siemens V20 VFD values
            self.run_cw = 0x0C7F   # Forward - ON
            self.run_ccw = 0x047F  # Reverse - ON
            self.stop = 0x0C7E     # Forward - OFF
            self.speed_multiplier = 1
        elif self.protocol == 'yl620':
            # YL620 VFD values
            self.run_cw = 0x12     # Start in forward direction
            self.run_ccw = 0x22    # Start in reverse direction
            self.stop = 0x01       # Disable spindle
            self.speed_multiplier = 1
        elif self.protocol == 'nowforever':
            # NowForever VFD values
            self.run_cw = 0x01     # Run CW
            self.run_ccw = 0x03    # Run CCW
            self.stop = 0x00       # Stop
            self.speed_multiplier = 100
        else:
            # Generic Modbus VFD values
            self.run_cw = 0x0001
            self.run_ccw = 0x0002
            self.stop = 0x0000
            self.speed_multiplier = 1
            
    def _handle_ready(self):
        """Initialize Modbus connection when Klipper is ready"""
        try:
            self.modbus.connect()
            self._start_status_monitoring()
            logging.info(f"Modbus spindle '{self.name}' ready")
        except Exception as e:
            logging.error(f"Failed to initialize Modbus spindle: {e}")
            
    def _shutdown(self):
        """Cleanup on shutdown"""
        self._stop_status_monitoring()
        if self.is_on:
            try:
                self._set_spindle_state(False)
            except Exception:
                pass  # Ignore errors during shutdown
                
    def _start_status_monitoring(self):
        """Start status monitoring thread"""
        if not self.status_running:
            self.status_running = True
            self.status_thread = threading.Thread(target=self._status_worker)
            self.status_thread.daemon = True
            self.status_thread.start()
            
    def _stop_status_monitoring(self):
        """Stop status monitoring thread"""
        self.status_running = False
        if self.status_thread and self.status_thread.is_alive():
            self.status_thread.join(timeout=2.0)
            
    def _status_worker(self):
        """Status monitoring thread worker"""
        while self.status_running:
            try:
                if self.is_on:
                    # Read current speed from VFD
                    try:
                        registers = self.modbus.read_holding_registers(
                            self.frequency_register, 1)
                        if registers:
                            # Convert register value to RPM
                            current_rpm = registers[0] / self.speed_multiplier
                            if abs(current_rpm - self.current_speed) > 10:
                                logging.debug(f"Spindle speed: {current_rpm:.1f} RPM")
                    except Exception as e:
                        logging.debug(f"Failed to read spindle status: {e}")
                        
                time.sleep(self.status_interval)
                
            except Exception as e:
                logging.error(f"Status monitoring error: {e}")
                time.sleep(1.0)
                
    def _rpm_to_register_value(self, rpm):
        """Convert RPM to VFD register value"""
        if self.protocol == 'huanyang':
            # Huanyang expects frequency in Hz (typically 50Hz = rated RPM)
            # Assume 50Hz = max_rpm for now (configurable in real implementation)
            max_freq = 50.0
            freq = (rpm / self.max_rpm) * max_freq
            return int(freq * 100)  # Huanyang uses 0.01Hz resolution
        else:
            # Generic: direct RPM or scaled value
            return int(rpm * self.speed_multiplier)
            
    def _set_spindle_state(self, enable, direction=1, speed=0.):
        """Set the VFD spindle state via Modbus"""
        try:
            if enable and speed > 0:
                # Set speed first
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
                self.current_speed = speed
                self.target_speed = speed
                
            else:
                # Stop spindle
                self.modbus.write_single_register(self.control_register, self.stop)
                
                # Update state
                self.is_on = False
                self.current_speed = 0.
                self.target_speed = 0.
                
        except Exception as e:
            raise self.gcode.error(f"Modbus spindle communication failed: {e}")
            
    cmd_M3_help = "Turn spindle on clockwise (M3 S[rpm])"
    def cmd_M3(self, gcmd):
        """M3 - Spindle on, clockwise"""
        speed = gcmd.get_float('S', 0., minval=0.)
        if speed > self.max_rpm:
            gcmd.respond_info("Warning: Requested speed %.1f exceeds maximum %.1f RPM" 
                            % (speed, self.max_rpm))
            speed = self.max_rpm
        self._set_spindle_state(True, 1, speed)
        gcmd.respond_info("Modbus spindle on CW at %.1f RPM" % speed)
        
    cmd_M4_help = "Turn spindle on counter-clockwise (M4 S[rpm])"  
    def cmd_M4(self, gcmd):
        """M4 - Spindle on, counter-clockwise"""
        speed = gcmd.get_float('S', 0., minval=0.)
        if speed > self.max_rpm:
            gcmd.respond_info("Warning: Requested speed %.1f exceeds maximum %.1f RPM"
                            % (speed, self.max_rpm))
            speed = self.max_rpm
        self._set_spindle_state(True, -1, speed)
        gcmd.respond_info("Modbus spindle on CCW at %.1f RPM" % speed)
        
    cmd_M5_help = "Turn spindle off"
    def cmd_M5(self, gcmd):
        """M5 - Spindle off"""
        self._set_spindle_state(False)
        gcmd.respond_info("Modbus spindle off")
        
    def get_status(self, eventtime):
        """Return current spindle status"""
        return {
            'enabled': self.is_on,
            'speed': self.current_speed,
            'target_speed': self.target_speed,
            'direction': 'CW' if self.current_direction > 0 else 'CCW',
            'protocol': self.protocol
        }

def load_config(config):
    return ModbusSpindle(config)

def load_config_prefix(config):
    return ModbusSpindle(config)