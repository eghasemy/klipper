# CNC probing support (G38.x commands)
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging

class CncProbing:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object('gcode')
        self.name = config.get_name().split()[-1]
        
        # Probe configuration
        ppins = self.printer.lookup_object('pins')
        probe_pin = config.get('probe_pin')
        self.probe_pin = ppins.setup_pin('endstop', probe_pin)
        
        # Probe settings
        self.speed = config.getfloat('speed', 5.0, above=0.)
        self.lift_speed = config.getfloat('lift_speed', None)
        if self.lift_speed is None:
            self.lift_speed = self.speed
        self.samples = config.getint('samples', 1, minval=1)
        self.sample_retract_dist = config.getfloat('sample_retract_dist', 1.0, above=0.)
        self.samples_tolerance = config.getfloat('samples_tolerance', 0.100, minval=0.)
        self.samples_tolerance_retries = config.getint('samples_tolerance_retries', 0, minval=0)
        
        # Results storage
        self.last_probe_result = [0., 0., 0.]
        
        # Register G38.x commands
        for i in range(2, 6):  # G38.2, G38.3, G38.4, G38.5
            self.gcode.register_command('G38.%d' % i, getattr(self, '_cmd_G38_%d' % i),
                                      desc="Probe toward workpiece (G38.%d)" % i)
                                      
    def _get_probe_position(self):
        """Get current probe pin status"""
        toolhead = self.printer.lookup_object('toolhead')
        print_time = toolhead.get_last_move_time()
        return self.probe_pin.query_endstop(print_time)
        
    def _probe_move(self, pos, speed, error_on_trigger=False, error_on_no_trigger=False):
        """Perform a probing move"""
        toolhead = self.printer.lookup_object('toolhead')
        start_pos = toolhead.get_position()
        
        # Setup endstop checking
        def check_probe(pos):
            triggered = self._get_probe_position()
            if error_on_trigger and triggered:
                raise self.gcode.error("Probe triggered before move started")
            return triggered
            
        # Perform the move with endstop checking
        toolhead.set_position(start_pos)
        phoming = self.printer.lookup_object('homing')
        
        try:
            # Use the homing system's probing capability
            curpos = phoming.probing_move(None, pos, speed)
            triggered = self._get_probe_position()
            
            if error_on_no_trigger and not triggered:
                raise self.gcode.error("Probe did not trigger during move")
                
            self.last_probe_result = list(curpos)
            return curpos, triggered
            
        except Exception as e:
            if error_on_no_trigger or error_on_trigger:
                raise
            # For non-error variants, return current position
            return toolhead.get_position(), self._get_probe_position()
            
    def _cmd_G38_2(self, gcmd):
        """G38.2 - Probe toward workpiece, stop on contact, signal error if failure"""
        pos = self._parse_probe_command(gcmd)
        result_pos, triggered = self._probe_move(pos, self.speed, 
                                                error_on_no_trigger=True)
        gcmd.respond_info("Probe triggered at X:%.3f Y:%.3f Z:%.3f" 
                        % (result_pos[0], result_pos[1], result_pos[2]))
        
    def _cmd_G38_3(self, gcmd):
        """G38.3 - Probe toward workpiece, stop on contact"""
        pos = self._parse_probe_command(gcmd)
        result_pos, triggered = self._probe_move(pos, self.speed)
        if triggered:
            gcmd.respond_info("Probe triggered at X:%.3f Y:%.3f Z:%.3f"
                            % (result_pos[0], result_pos[1], result_pos[2]))
        else:
            gcmd.respond_info("Probe move completed without trigger")
            
    def _cmd_G38_4(self, gcmd):
        """G38.4 - Probe away from workpiece, stop on loss of contact, signal error if failure"""
        pos = self._parse_probe_command(gcmd)
        # Check that probe is initially triggered
        if not self._get_probe_position():
            raise gcmd.error("Probe not triggered at start of G38.4 move")
        result_pos, triggered = self._probe_move(pos, self.speed,
                                                error_on_trigger=True)
        gcmd.respond_info("Probe cleared at X:%.3f Y:%.3f Z:%.3f"
                        % (result_pos[0], result_pos[1], result_pos[2]))
        
    def _cmd_G38_5(self, gcmd):
        """G38.5 - Probe away from workpiece, stop on loss of contact"""
        pos = self._parse_probe_command(gcmd)
        result_pos, triggered = self._probe_move(pos, self.speed)
        if not triggered:
            gcmd.respond_info("Probe cleared at X:%.3f Y:%.3f Z:%.3f"
                            % (result_pos[0], result_pos[1], result_pos[2]))
        else:
            gcmd.respond_info("Probe move completed, still triggered")
            
    def _parse_probe_command(self, gcmd):
        """Parse G38.x command parameters"""
        toolhead = self.printer.lookup_object('toolhead')
        current_pos = toolhead.get_position()
        
        # Get target position (relative to current position)
        x = gcmd.get_float('X', current_pos[0])
        y = gcmd.get_float('Y', current_pos[1])
        z = gcmd.get_float('Z', current_pos[2])
        
        # Get feedrate
        speed = gcmd.get_float('F', self.speed * 60.) / 60.  # Convert from mm/min to mm/s
        
        return [x, y, z, current_pos[3]]  # Keep E axis unchanged
        
    def get_status(self, eventtime):
        """Return current probing status"""
        return {
            'last_probe': self.last_probe_result,
            'probe_pin_state': self._get_probe_position()
        }

def load_config(config):
    return CncProbing(config)

def load_config_prefix(config):
    return CncProbing(config)