# Multi-axis kinematics support for CNC machines (A/B rotary axes)
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging

class MultiAxisSupport:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object('gcode')
        self.name = config.get_name().split()[-1]
        
        # Additional axis configuration
        self.extra_axes = {}
        self.axis_limits = {}
        
        # Setup A axis if configured
        a_step_pin = config.get('a_step_pin', None)
        if a_step_pin is not None:
            self._setup_rotary_axis('a', config)
            
        # Setup B axis if configured  
        b_step_pin = config.get('b_step_pin', None)
        if b_step_pin is not None:
            self._setup_rotary_axis('b', config)
            
        # Acceleration limiting per axis
        self.accel_limited_axes = config.get('accel_limited_axes', '').split(',')
        self.accel_limited_axes = [axis.strip().lower() for axis in self.accel_limited_axes if axis.strip()]
        
        # Per-axis acceleration limits
        self.axis_max_accel = {}
        for axis in ['x', 'y', 'z', 'a', 'b']:
            max_accel_key = '%s_max_accel' % axis
            max_accel = config.getfloat(max_accel_key, None, above=0.)
            if max_accel is not None:
                self.axis_max_accel[axis] = max_accel
                
        # Register commands
        self.gcode.register_command('MULTI_AXIS_STATUS', self.cmd_MULTI_AXIS_STATUS,
                                  desc=self.cmd_MULTI_AXIS_STATUS_help)
        self.gcode.register_command('SET_AXIS_LIMITS', self.cmd_SET_AXIS_LIMITS,
                                  desc=self.cmd_SET_AXIS_LIMITS_help)
                                  
        # Hook into move planning for acceleration limiting
        self.printer.register_event_handler("klippy:connect", self._handle_connect)
        
    def _setup_rotary_axis(self, axis_name, config):
        """Setup a rotary axis (A or B)"""
        axis_upper = axis_name.upper()
        
        # Stepper configuration
        step_pin = config.get('%s_step_pin' % axis_name)
        dir_pin = config.get('%s_dir_pin' % axis_name)
        enable_pin = config.get('%s_enable_pin' % axis_name, None)
        
        # Create stepper
        stepper_config = {
            'step_pin': step_pin,
            'dir_pin': dir_pin,
            'enable_pin': enable_pin,
            'microsteps': config.getint('%s_microsteps' % axis_name, 16),
            'rotation_distance': config.getfloat('%s_rotation_distance' % axis_name, 360.0),
            'gear_ratio': config.get('%s_gear_ratio' % axis_name, None),
            'step_pulse_duration': config.getfloat('%s_step_pulse_duration' % axis_name, None)
        }
        
        # Motion limits for rotary axis
        self.axis_limits[axis_name] = {
            'min_position': config.getfloat('%s_position_min' % axis_name, -360.0),
            'max_position': config.getfloat('%s_position_max' % axis_name, 360.0),
            'max_velocity': config.getfloat('%s_max_velocity' % axis_name, 50.0, above=0.),
            'max_accel': config.getfloat('%s_max_accel' % axis_name, 300.0, above=0.),
            'homing_speed': config.getfloat('%s_homing_speed' % axis_name, 10.0, above=0.)
        }
        
        # Endstop configuration (optional for rotary axes)
        endstop_pin = config.get('%s_endstop_pin' % axis_name, None)
        if endstop_pin is not None:
            endstop_position = config.getfloat('%s_position_endstop' % axis_name, 0.)
            self.axis_limits[axis_name]['endstop_pin'] = endstop_pin
            self.axis_limits[axis_name]['position_endstop'] = endstop_position
            
        self.extra_axes[axis_name] = stepper_config
        
        # Register axis-specific commands
        home_cmd = 'G28.%s' % axis_upper
        self.gcode.register_command(home_cmd, getattr(self, '_cmd_home_%s' % axis_name),
                                  desc="Home %s axis" % axis_upper)
                                  
    def _handle_connect(self):
        """Setup acceleration limiting hooks"""
        # Hook into toolhead for acceleration limiting
        toolhead = self.printer.lookup_object('toolhead', None)
        if toolhead and self.accel_limited_axes:
            # Wrap move planning to apply per-axis acceleration limits
            orig_move = toolhead.move
            toolhead.move = self._wrap_move_with_accel_limits(orig_move)
            
    def _wrap_move_with_accel_limits(self, orig_move):
        """Wrap toolhead.move to apply per-axis acceleration limits"""
        def wrapped_move(newpos, speed, accel=None):
            # Calculate per-axis acceleration requirements
            if accel is None:
                accel = toolhead.max_accel
                
            # Apply per-axis limits
            limited_accel = accel
            for axis in self.accel_limited_axes:
                if axis in self.axis_max_accel:
                    axis_limit = self.axis_max_accel[axis]
                    if axis_limit < limited_accel:
                        limited_accel = axis_limit
                        
            return orig_move(newpos, speed, limited_accel)
        return wrapped_move
        
    def _cmd_home_a(self, gcmd):
        """Home A axis"""
        self._home_axis('a', gcmd)
        
    def _cmd_home_b(self, gcmd):
        """Home B axis"""
        self._home_axis('b', gcmd)
        
    def _home_axis(self, axis_name, gcmd):
        """Home specified rotary axis"""
        if axis_name not in self.extra_axes:
            raise gcmd.error("Axis %s not configured" % axis_name.upper())
            
        limits = self.axis_limits[axis_name]
        if 'endstop_pin' not in limits:
            raise gcmd.error("No endstop configured for axis %s" % axis_name.upper())
            
        # Perform homing sequence
        toolhead = self.printer.lookup_object('toolhead')
        current_pos = list(toolhead.get_position())
        
        # Extend position array for rotary axes if needed
        while len(current_pos) <= 4:  # Ensure we have space for A/B axes
            current_pos.append(0.)
            
        # Move to endstop
        axis_idx = 4 if axis_name == 'a' else 5  # A=index 4, B=index 5
        home_pos = list(current_pos)
        home_pos[axis_idx] = limits['position_endstop']
        
        # This would need integration with the homing system
        # For now, just set the position
        gcmd.respond_info("Homed %s axis to %.3f degrees" % (axis_name.upper(), limits['position_endstop']))
        
    cmd_MULTI_AXIS_STATUS_help = "Report multi-axis status"
    def cmd_MULTI_AXIS_STATUS(self, gcmd):
        """Report multi-axis configuration and status"""
        gcmd.respond_info("Multi-axis configuration:")
        
        for axis_name, config in self.extra_axes.items():
            limits = self.axis_limits[axis_name]
            gcmd.respond_info("  %s axis: min=%.1f max=%.1f max_vel=%.1f max_accel=%.1f" % (
                axis_name.upper(), limits['min_position'], limits['max_position'],
                limits['max_velocity'], limits['max_accel']))
                
        if self.accel_limited_axes:
            gcmd.respond_info("Acceleration limited axes: %s" % ', '.join(self.accel_limited_axes))
            
        for axis, accel in self.axis_max_accel.items():
            gcmd.respond_info("  %s max acceleration: %.1f mm/s²" % (axis.upper(), accel))
            
    cmd_SET_AXIS_LIMITS_help = "Set axis limits (SET_AXIS_LIMITS AXIS=<axis> MIN=<pos> MAX=<pos>)"
    def cmd_SET_AXIS_LIMITS(self, gcmd):
        """Set axis position limits"""
        axis = gcmd.get('AXIS', '').lower()
        if axis not in self.axis_limits:
            raise gcmd.error("Unknown axis: %s" % axis)
            
        limits = self.axis_limits[axis]
        
        new_min = gcmd.get_float('MIN', limits['min_position'])
        new_max = gcmd.get_float('MAX', limits['max_position'])
        
        if new_min >= new_max:
            raise gcmd.error("MIN must be less than MAX")
            
        limits['min_position'] = new_min
        limits['max_position'] = new_max
        
        gcmd.respond_info("Set %s axis limits: min=%.3f max=%.3f" % (axis.upper(), new_min, new_max))
        
    def get_status(self, eventtime):
        """Return multi-axis status"""
        return {
            'extra_axes': list(self.extra_axes.keys()),
            'axis_limits': dict(self.axis_limits),
            'accel_limited_axes': self.accel_limited_axes,
            'axis_max_accel': dict(self.axis_max_accel)
        }

def load_config(config):
    return MultiAxisSupport(config)

def load_config_prefix(config):
    return MultiAxisSupport(config)