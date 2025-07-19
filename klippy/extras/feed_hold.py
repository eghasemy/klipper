# Feed hold and pause functionality for CNC machines
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging

class FeedHold:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object('gcode')
        self.name = config.get_name().split()[-1]
        
        # State tracking
        self.is_paused = False
        self.pause_position = None
        self.pause_state = {}
        
        # Optional feed hold pin (for external button)
        ppins = self.printer.lookup_object('pins')
        feed_hold_pin = config.get('feed_hold_pin', None)
        self.feed_hold_pin = None
        if feed_hold_pin is not None:
            self.feed_hold_pin = ppins.setup_pin('endstop', feed_hold_pin)
            self.printer.register_event_handler("klippy:connect", self._setup_feed_hold_monitoring)
            
        # Configuration
        self.retract_length = config.getfloat('retract_length', 0.5, minval=0.)
        self.retract_speed = config.getfloat('retract_speed', 20., above=0.)
        self.lift_z = config.getfloat('lift_z', 2., minval=0.)
        self.move_speed = config.getfloat('move_speed', 50., above=0.)
        
        # Register G-code commands
        self.gcode.register_command('M0', self.cmd_M0,
                                  desc=self.cmd_M0_help)
        self.gcode.register_command('M1', self.cmd_M1,
                                  desc=self.cmd_M1_help)
        self.gcode.register_command('FEED_HOLD', self.cmd_FEED_HOLD,
                                  desc=self.cmd_FEED_HOLD_help)
        self.gcode.register_command('CYCLE_START', self.cmd_CYCLE_START,
                                  desc=self.cmd_CYCLE_START_help)
        self.gcode.register_command('RESUME_CNC', self.cmd_RESUME_CNC,
                                  desc=self.cmd_RESUME_CNC_help)
                                  
    def _setup_feed_hold_monitoring(self):
        """Setup feed hold pin monitoring"""
        if self.feed_hold_pin is not None:
            # This would be implemented with a timer to periodically check the pin
            pass
            
    def _save_current_state(self):
        """Save current machine state for resume"""
        toolhead = self.printer.lookup_object('toolhead')
        self.pause_position = toolhead.get_position()
        
        # Save spindle state
        try:
            spindle = self.printer.lookup_object('spindle', None)
            if spindle:
                self.pause_state['spindle_on'] = spindle.is_on
                self.pause_state['spindle_speed'] = spindle.current_speed
                self.pause_state['spindle_direction'] = spindle.current_direction
            else:
                self.pause_state['spindle_on'] = False
        except:
            self.pause_state['spindle_on'] = False
            
        # Save coolant state  
        try:
            coolant = self.printer.lookup_object('coolant', None)
            if coolant:
                self.pause_state['mist_on'] = coolant.mist_on
                self.pause_state['flood_on'] = coolant.flood_on
            else:
                self.pause_state['mist_on'] = False
                self.pause_state['flood_on'] = False
        except:
            self.pause_state['mist_on'] = False
            self.pause_state['flood_on'] = False
            
    def _execute_pause_sequence(self):
        """Execute the pause sequence"""
        if self.is_paused:
            return
            
        # Save current state
        self._save_current_state()
        
        # Stop spindle
        try:
            self.gcode.run_script_from_command("M5")
        except:
            pass
            
        # Stop coolant
        try:
            self.gcode.run_script_from_command("M9")
        except:
            pass
            
        # Retract and lift
        if self.pause_position:
            # Retract E axis
            if self.retract_length > 0:
                self.gcode.run_script_from_command(
                    "G91\nG1 E-%.3f F%.1f\nG90" % (self.retract_length, self.retract_speed * 60))
                    
            # Lift Z axis
            if self.lift_z > 0:
                lift_pos = self.pause_position[2] + self.lift_z
                self.gcode.run_script_from_command(
                    "G1 Z%.3f F%.1f" % (lift_pos, self.move_speed * 60))
                    
        self.is_paused = True
        
    def _execute_resume_sequence(self):
        """Execute the resume sequence"""
        if not self.is_paused or not self.pause_position:
            return
            
        # Move back to position
        self.gcode.run_script_from_command(
            "G1 X%.3f Y%.3f F%.1f" % (self.pause_position[0], self.pause_position[1], 
                                     self.move_speed * 60))
        self.gcode.run_script_from_command(
            "G1 Z%.3f F%.1f" % (self.pause_position[2], self.move_speed * 60))
            
        # Restore E position
        if self.retract_length > 0:
            self.gcode.run_script_from_command(
                "G91\nG1 E%.3f F%.1f\nG90" % (self.retract_length, self.retract_speed * 60))
                
        # Restore spindle state
        if self.pause_state.get('spindle_on', False):
            direction_cmd = "M3" if self.pause_state.get('spindle_direction', 1) > 0 else "M4"
            speed = self.pause_state.get('spindle_speed', 0)
            self.gcode.run_script_from_command("%s S%.1f" % (direction_cmd, speed))
            
        # Restore coolant state
        if self.pause_state.get('mist_on', False):
            self.gcode.run_script_from_command("M7")
        if self.pause_state.get('flood_on', False):
            self.gcode.run_script_from_command("M8")
            
        self.is_paused = False
        self.pause_position = None
        self.pause_state = {}
        
    cmd_M0_help = "Program stop (M0 [P<seconds>])"
    def cmd_M0(self, gcmd):
        """M0 - Program stop"""
        # Optional pause time
        pause_time = gcmd.get_float('P', None, minval=0.)
        
        self._execute_pause_sequence()
        
        if pause_time is not None:
            # Automatic resume after specified time
            import time
            time.sleep(pause_time)
            self._execute_resume_sequence()
            gcmd.respond_info("Program resumed after %.1f seconds" % pause_time)
        else:
            gcmd.respond_info("Program paused - use CYCLE_START or RESUME_CNC to continue")
            
    cmd_M1_help = "Optional stop (M1)"
    def cmd_M1(self, gcmd):
        """M1 - Optional stop (same as M0 in this implementation)"""
        self.cmd_M0(gcmd)
        
    cmd_FEED_HOLD_help = "Immediate feed hold"
    def cmd_FEED_HOLD(self, gcmd):
        """Immediate feed hold"""
        # Stop all motion immediately
        toolhead = self.printer.lookup_object('toolhead')
        toolhead.wait_moves()
        
        self._execute_pause_sequence()
        gcmd.respond_info("Feed hold engaged - use CYCLE_START to resume")
        
    cmd_CYCLE_START_help = "Resume from feed hold or pause"
    def cmd_CYCLE_START(self, gcmd):
        """Resume from feed hold or pause"""
        if not self.is_paused:
            gcmd.respond_info("Machine not paused")
            return
            
        self._execute_resume_sequence()
        gcmd.respond_info("Cycle resumed")
        
    cmd_RESUME_CNC_help = "Resume CNC operation"
    def cmd_RESUME_CNC(self, gcmd):
        """Resume CNC operation (alias for CYCLE_START)"""
        self.cmd_CYCLE_START(gcmd)
        
    def get_status(self, eventtime):
        """Return current pause status"""
        return {
            'is_paused': self.is_paused,
            'pause_position': self.pause_position,
            'feed_hold_pin_state': self._get_feed_hold_pin_state() if self.feed_hold_pin else None
        }
        
    def _get_feed_hold_pin_state(self):
        """Get current feed hold pin state"""
        if self.feed_hold_pin is None:
            return None
        toolhead = self.printer.lookup_object('toolhead')
        print_time = toolhead.get_last_move_time()
        return self.feed_hold_pin.query_endstop(print_time)

def load_config(config):
    return FeedHold(config)

def load_config_prefix(config):
    return FeedHold(config)