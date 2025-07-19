# Coolant control support for CNC machines
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging

class Coolant:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object('gcode')
        self.name = config.get_name().split()[-1]
        
        # Pin configuration
        ppins = self.printer.lookup_object('pins')
        self.mist_pin = None
        self.flood_pin = None
        
        # Setup mist coolant pin
        mist_pin = config.get('mist_pin', None)
        if mist_pin is not None:
            self.mist_pin = ppins.setup_pin('digital_out', mist_pin)
            self.mist_pin.setup_max_duration(0.)
            
        # Setup flood coolant pin  
        flood_pin = config.get('flood_pin', None)
        if flood_pin is not None:
            self.flood_pin = ppins.setup_pin('digital_out', flood_pin)
            self.flood_pin.setup_max_duration(0.)
            
        # Current state
        self.mist_on = False
        self.flood_on = False
        
        # Register G-code commands
        self.gcode.register_command('M7', self.cmd_M7,
                                  desc=self.cmd_M7_help)
        self.gcode.register_command('M8', self.cmd_M8,
                                  desc=self.cmd_M8_help)
        self.gcode.register_command('M9', self.cmd_M9,
                                  desc=self.cmd_M9_help)
                                  
    def _set_mist(self, enable):
        """Set mist coolant state"""
        if self.mist_pin is not None:
            self.mist_pin.set_digital(self.printer.lookup_object('toolhead').get_last_move_time(),
                                    1 if enable else 0)
        self.mist_on = enable
        
    def _set_flood(self, enable):
        """Set flood coolant state"""
        if self.flood_pin is not None:
            self.flood_pin.set_digital(self.printer.lookup_object('toolhead').get_last_move_time(),
                                     1 if enable else 0)
        self.flood_on = enable
        
    cmd_M7_help = "Turn mist coolant on"
    def cmd_M7(self, gcmd):
        """M7 - Mist coolant on"""
        self._set_mist(True)
        gcmd.respond_info("Mist coolant on")
        
    cmd_M8_help = "Turn flood coolant on"
    def cmd_M8(self, gcmd):
        """M8 - Flood coolant on"""
        self._set_flood(True)
        gcmd.respond_info("Flood coolant on")
        
    cmd_M9_help = "Turn all coolant off"
    def cmd_M9(self, gcmd):
        """M9 - All coolant off"""
        self._set_mist(False)
        self._set_flood(False)
        gcmd.respond_info("All coolant off")
        
    def get_status(self, eventtime):
        """Return current coolant status"""
        return {
            'mist': self.mist_on,
            'flood': self.flood_on
        }

def load_config(config):
    return Coolant(config)

def load_config_prefix(config):
    return Coolant(config)