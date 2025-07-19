# Canned drilling cycles for CNC machines
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging

class CannedCycles:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object('gcode')
        self.name = config.get_name().split()[-1]
        
        # Default cycle parameters
        self.retract_mode = 0  # 0 = G98 (return to initial), 1 = G99 (return to R plane)
        self.feed_rate = config.getfloat('default_feed_rate', 100., above=0.)
        self.rapid_rate = config.getfloat('default_rapid_rate', 500., above=0.)
        self.dwell_time = config.getfloat('default_dwell_time', 0., minval=0.)
        
        # Current cycle parameters (set by G-code)
        self.cycle_params = {
            'X': None, 'Y': None, 'Z': None,  # Current position
            'R': None,  # Retract/Reference plane
            'F': self.feed_rate,  # Feed rate
            'Q': None,  # Peck increment (for G83)
            'P': self.dwell_time,  # Dwell time
        }
        
        # Register G-code commands
        self.gcode.register_command('G81', self.cmd_G81,
                                  desc=self.cmd_G81_help)
        self.gcode.register_command('G82', self.cmd_G82,
                                  desc=self.cmd_G82_help)
        self.gcode.register_command('G83', self.cmd_G83,
                                  desc=self.cmd_G83_help)
        self.gcode.register_command('G80', self.cmd_G80,
                                  desc=self.cmd_G80_help)
        self.gcode.register_command('G98', self.cmd_G98,
                                  desc=self.cmd_G98_help)
        self.gcode.register_command('G99', self.cmd_G99,
                                  desc=self.cmd_G99_help)
                                  
    def _update_cycle_params(self, gcmd):
        """Update cycle parameters from G-code command"""
        toolhead = self.printer.lookup_object('toolhead')
        current_pos = toolhead.get_position()
        
        # Update position parameters
        self.cycle_params['X'] = gcmd.get_float('X', current_pos[0])
        self.cycle_params['Y'] = gcmd.get_float('Y', current_pos[1])
        self.cycle_params['Z'] = gcmd.get_float('Z', self.cycle_params['Z'])
        
        # Update other parameters
        if gcmd.get_float('R', None) is not None:
            self.cycle_params['R'] = gcmd.get_float('R')
        if gcmd.get_float('F', None) is not None:
            self.cycle_params['F'] = gcmd.get_float('F') / 60.  # Convert mm/min to mm/s
        if gcmd.get_float('Q', None) is not None:
            self.cycle_params['Q'] = gcmd.get_float('Q')
        if gcmd.get_float('P', None) is not None:
            self.cycle_params['P'] = gcmd.get_float('P')
            
    def _validate_cycle_params(self):
        """Validate that required cycle parameters are set"""
        if self.cycle_params['R'] is None:
            raise self.gcode.error("R parameter (retract plane) required for drilling cycle")
        if self.cycle_params['Z'] is None:
            raise self.gcode.error("Z parameter (drill depth) required for drilling cycle")
        if self.cycle_params['R'] <= self.cycle_params['Z']:
            raise self.gcode.error("R plane must be above Z depth")
            
    def _rapid_move(self, pos):
        """Perform rapid positioning move"""
        toolhead = self.printer.lookup_object('toolhead')
        toolhead.move(pos, self.rapid_rate)
        
    def _feed_move(self, pos):
        """Perform feed rate move"""
        toolhead = self.printer.lookup_object('toolhead')
        toolhead.move(pos, self.cycle_params['F'])
        
    def _dwell(self, time_seconds):
        """Dwell for specified time"""
        if time_seconds > 0:
            toolhead = self.printer.lookup_object('toolhead')
            toolhead.dwell(time_seconds)
            
    def _execute_drill_cycle(self, gcmd, peck_depth=None, dwell=False):
        """Execute basic drilling cycle"""
        self._update_cycle_params(gcmd)
        self._validate_cycle_params()
        
        toolhead = self.printer.lookup_object('toolhead')
        start_pos = list(toolhead.get_position())
        
        # Move to XY position at current Z
        xy_pos = [self.cycle_params['X'], self.cycle_params['Y'], start_pos[2], start_pos[3]]
        self._rapid_move(xy_pos)
        
        # Move to R plane
        r_pos = [self.cycle_params['X'], self.cycle_params['Y'], self.cycle_params['R'], start_pos[3]]
        self._rapid_move(r_pos)
        
        # Drilling operation
        target_z = self.cycle_params['Z']
        current_z = self.cycle_params['R']
        
        if peck_depth is not None and peck_depth > 0:
            # Peck drilling (G83)
            while current_z > target_z:
                # Calculate next peck depth
                next_z = max(target_z, current_z - peck_depth)
                
                # Drill to peck depth
                drill_pos = [self.cycle_params['X'], self.cycle_params['Y'], next_z, start_pos[3]]
                self._feed_move(drill_pos)
                
                if dwell:
                    self._dwell(self.cycle_params['P'])
                    
                # Retract to R plane for chip clearing (except on final pass)
                if next_z > target_z:
                    self._rapid_move(r_pos)
                    
                current_z = next_z
        else:
            # Simple drilling (G81/G82)
            drill_pos = [self.cycle_params['X'], self.cycle_params['Y'], target_z, start_pos[3]]
            self._feed_move(drill_pos)
            
            if dwell:
                self._dwell(self.cycle_params['P'])
                
        # Retract according to mode
        if self.retract_mode == 0:  # G98 - return to initial Z
            retract_pos = [self.cycle_params['X'], self.cycle_params['Y'], start_pos[2], start_pos[3]]
        else:  # G99 - return to R plane
            retract_pos = r_pos
        self._rapid_move(retract_pos)
        
    cmd_G81_help = "Drilling cycle (G81 X Y Z R F)"
    def cmd_G81(self, gcmd):
        """G81 - Simple drilling cycle"""
        self._execute_drill_cycle(gcmd)
        gcmd.respond_info("G81 drilling cycle completed")
        
    cmd_G82_help = "Drilling cycle with dwell (G82 X Y Z R F P)"
    def cmd_G82(self, gcmd):
        """G82 - Drilling cycle with dwell"""
        self._execute_drill_cycle(gcmd, dwell=True)
        gcmd.respond_info("G82 drilling cycle with dwell completed")
        
    cmd_G83_help = "Peck drilling cycle (G83 X Y Z R F Q)"
    def cmd_G83(self, gcmd):
        """G83 - Peck drilling cycle"""
        peck_depth = self.cycle_params.get('Q', None)
        if peck_depth is None:
            raise gcmd.error("Q parameter (peck depth) required for G83")
        self._execute_drill_cycle(gcmd, peck_depth=peck_depth)
        gcmd.respond_info("G83 peck drilling cycle completed")
        
    cmd_G80_help = "Cancel canned cycle"
    def cmd_G80(self, gcmd):
        """G80 - Cancel canned cycle"""
        # Reset cycle parameters
        self.cycle_params = {
            'X': None, 'Y': None, 'Z': None,
            'R': None, 'F': self.feed_rate,
            'Q': None, 'P': self.dwell_time,
        }
        gcmd.respond_info("Canned cycle cancelled")
        
    cmd_G98_help = "Set retract mode to initial Z"
    def cmd_G98(self, gcmd):
        """G98 - Return to initial Z level in canned cycles"""
        self.retract_mode = 0
        gcmd.respond_info("Retract mode: return to initial Z")
        
    cmd_G99_help = "Set retract mode to R plane"
    def cmd_G99(self, gcmd):
        """G99 - Return to R plane in canned cycles"""
        self.retract_mode = 1
        gcmd.respond_info("Retract mode: return to R plane")
        
    def get_status(self, eventtime):
        """Return current canned cycle status"""
        return {
            'retract_mode': 'G98' if self.retract_mode == 0 else 'G99',
            'cycle_params': dict(self.cycle_params)
        }

def load_config(config):
    return CannedCycles(config)

def load_config_prefix(config):
    return CannedCycles(config)