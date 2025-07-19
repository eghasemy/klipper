# Spindle control support for CNC machines
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging

class Spindle:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object('gcode')
        self.name = config.get_name().split()[-1]
        
        # Pin configuration
        ppins = self.printer.lookup_object('pins')
        self.enable_pin = None
        self.direction_pin = None
        self.pwm_pin = None
        
        # Setup enable pin (for relay control)
        enable_pin = config.get('enable_pin', None)
        if enable_pin is not None:
            self.enable_pin = ppins.setup_pin('digital_out', enable_pin)
            self.enable_pin.setup_max_duration(0.)
            
        # Setup direction pin (for M3/M4 CW/CCW control)
        direction_pin = config.get('direction_pin', None)
        if direction_pin is not None:
            self.direction_pin = ppins.setup_pin('digital_out', direction_pin)
            self.direction_pin.setup_max_duration(0.)
            
        # Setup PWM pin (for speed control)
        pwm_pin = config.get('pwm_pin', None)
        if pwm_pin is not None:
            self.pwm_pin = ppins.setup_pin('pwm', pwm_pin)
            cycle_time = config.getfloat('cycle_time', 0.001, above=0.)
            hardware_pwm = config.getboolean('hardware_pwm', False)
            self.pwm_pin.setup_cycle_time(cycle_time, hardware_pwm)
            self.pwm_pin.setup_max_duration(0.)
            
        # Speed mapping configuration
        self.min_rpm = config.getfloat('min_rpm', 0., minval=0.)
        self.max_rpm = config.getfloat('max_rpm', 1000., above=self.min_rpm)
        self.min_power = config.getfloat('min_power', 0., minval=0., maxval=1.)
        self.max_power = config.getfloat('max_power', 1., above=self.min_power, maxval=1.)
        
        # Current state
        self.is_on = False
        self.current_speed = 0.
        self.current_direction = 1  # 1 = CW (M3), -1 = CCW (M4)
        
        # Register G-code commands
        self.gcode.register_command('M3', self.cmd_M3, 
                                  desc=self.cmd_M3_help)
        self.gcode.register_command('M4', self.cmd_M4,
                                  desc=self.cmd_M4_help)
        self.gcode.register_command('M5', self.cmd_M5,
                                  desc=self.cmd_M5_help)
                                  
    def _set_spindle_state(self, enable, direction=1, speed=0.):
        """Set the physical spindle state"""
        # Update enable pin
        if self.enable_pin is not None:
            self.enable_pin.set_digital(self.printer.lookup_object('toolhead').get_last_move_time(), 
                                      1 if enable else 0)
                                      
        # Update direction pin (only if spindle is enabled)
        if self.direction_pin is not None and enable:
            # Assuming CW = 0, CCW = 1 (configurable via pin inversion)
            dir_value = 0 if direction > 0 else 1
            self.direction_pin.set_digital(self.printer.lookup_object('toolhead').get_last_move_time(),
                                         dir_value)
                                         
        # Update PWM pin for speed control
        if self.pwm_pin is not None and enable:
            # Map RPM to PWM duty cycle
            if speed > 0:
                rpm_range = self.max_rpm - self.min_rpm
                power_range = self.max_power - self.min_power
                if rpm_range > 0:
                    normalized_speed = max(0., min(1., (speed - self.min_rpm) / rpm_range))
                    pwm_value = self.min_power + normalized_speed * power_range
                else:
                    pwm_value = self.max_power
            else:
                pwm_value = 0.
            self.pwm_pin.set_pwm(self.printer.lookup_object('toolhead').get_last_move_time(), 
                               pwm_value)
        elif self.pwm_pin is not None:
            # Turn off PWM when spindle is disabled
            self.pwm_pin.set_pwm(self.printer.lookup_object('toolhead').get_last_move_time(), 0.)
            
        # Update state
        self.is_on = enable
        self.current_direction = direction
        self.current_speed = speed if enable else 0.
        
    cmd_M3_help = "Turn spindle on clockwise (M3 S[rpm])"
    def cmd_M3(self, gcmd):
        """M3 - Spindle on, clockwise"""
        speed = gcmd.get_float('S', 0., minval=0.)
        if speed > self.max_rpm:
            gcmd.respond_info("Warning: Requested speed %.1f exceeds maximum %.1f RPM" 
                            % (speed, self.max_rpm))
            speed = self.max_rpm
        self._set_spindle_state(True, 1, speed)
        gcmd.respond_info("Spindle on CW at %.1f RPM" % speed)
        
    cmd_M4_help = "Turn spindle on counter-clockwise (M4 S[rpm])"  
    def cmd_M4(self, gcmd):
        """M4 - Spindle on, counter-clockwise"""
        speed = gcmd.get_float('S', 0., minval=0.)
        if speed > self.max_rpm:
            gcmd.respond_info("Warning: Requested speed %.1f exceeds maximum %.1f RPM"
                            % (speed, self.max_rpm))
            speed = self.max_rpm
        self._set_spindle_state(True, -1, speed)
        gcmd.respond_info("Spindle on CCW at %.1f RPM" % speed)
        
    cmd_M5_help = "Turn spindle off"
    def cmd_M5(self, gcmd):
        """M5 - Spindle off"""
        self._set_spindle_state(False)
        gcmd.respond_info("Spindle off")
        
    def get_status(self, eventtime):
        """Return current spindle status"""
        return {
            'enabled': self.is_on,
            'speed': self.current_speed,
            'direction': 'CW' if self.current_direction > 0 else 'CCW'
        }

def load_config(config):
    return Spindle(config)

def load_config_prefix(config):
    return Spindle(config)