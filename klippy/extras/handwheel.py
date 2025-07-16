# Handwheel/jog wheel support for manual CNC control
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging

class Handwheel:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object('gcode')
        self.name = config.get_name().split()[-1]
        
        # Pin configuration
        ppins = self.printer.lookup_object('pins')
        
        # Encoder pins
        self.pin_a = None
        self.pin_b = None
        encoder_pin_a = config.get('encoder_pin_a', None)
        encoder_pin_b = config.get('encoder_pin_b', None)
        
        if encoder_pin_a is not None and encoder_pin_b is not None:
            self.pin_a = ppins.setup_pin('endstop', encoder_pin_a)
            self.pin_b = ppins.setup_pin('endstop', encoder_pin_b)
            
        # Axis selection pins (optional)
        self.axis_pins = {}
        for axis in ['x', 'y', 'z', 'a', 'b']:
            pin_name = 'axis_%s_pin' % axis
            pin = config.get(pin_name, None)
            if pin is not None:
                self.axis_pins[axis] = ppins.setup_pin('endstop', pin)
                
        # Step multiplier pins (optional - for different jog distances)
        self.multiplier_pins = {}
        multipliers = config.get('step_multipliers', '0.001,0.01,0.1,1.0').split(',')
        self.step_multipliers = [float(x.strip()) for x in multipliers]
        
        for i, mult in enumerate(self.step_multipliers):
            pin_name = 'multiplier_%d_pin' % i
            pin = config.get(pin_name, None)
            if pin is not None:
                self.multiplier_pins[i] = ppins.setup_pin('endstop', pin)
                
        # Configuration
        self.encoder_resolution = config.getint('encoder_resolution', 100, minval=1)
        self.max_velocity = config.getfloat('max_velocity', 25., above=0.)
        self.default_step = config.getfloat('default_step', 0.1, above=0.)
        
        # State tracking
        self.encoder_position = 0
        self.last_encoder_state = [False, False]
        self.active_axis = 'x'
        self.active_multiplier = 2  # Default to 0.1mm
        self.enabled = False
        
        # Register commands
        self.gcode.register_command('HANDWHEEL_ENABLE', self.cmd_HANDWHEEL_ENABLE,
                                  desc=self.cmd_HANDWHEEL_ENABLE_help)
        self.gcode.register_command('HANDWHEEL_DISABLE', self.cmd_HANDWHEEL_DISABLE,
                                  desc=self.cmd_HANDWHEEL_DISABLE_help)
        self.gcode.register_command('HANDWHEEL_JOG', self.cmd_HANDWHEEL_JOG,
                                  desc=self.cmd_HANDWHEEL_JOG_help)
        self.gcode.register_command('HANDWHEEL_SET_AXIS', self.cmd_HANDWHEEL_SET_AXIS,
                                  desc=self.cmd_HANDWHEEL_SET_AXIS_help)
        self.gcode.register_command('HANDWHEEL_SET_STEP', self.cmd_HANDWHEEL_SET_STEP,
                                  desc=self.cmd_HANDWHEEL_SET_STEP_help)
                                  
        # Setup monitoring if encoder pins are configured
        if self.pin_a is not None and self.pin_b is not None:
            self.printer.register_event_handler("klippy:connect", self._setup_monitoring)
            
    def _setup_monitoring(self):
        """Setup encoder monitoring"""
        # In a real implementation, this would set up interrupt handlers
        # or periodic polling for encoder state changes
        reactor = self.printer.get_reactor()
        reactor.register_timer(self._check_encoder_state, reactor.monotonic() + 0.01)
        
    def _get_pin_state(self, pin):
        """Get current state of a pin"""
        if pin is None:
            return False
        toolhead = self.printer.lookup_object('toolhead')
        print_time = toolhead.get_last_move_time()
        return pin.query_endstop(print_time)
        
    def _check_encoder_state(self, eventtime):
        """Check encoder state and handle changes"""
        if not self.enabled or self.pin_a is None or self.pin_b is None:
            return eventtime + 0.01
            
        # Read current encoder state
        state_a = self._get_pin_state(self.pin_a)
        state_b = self._get_pin_state(self.pin_b)
        current_state = [state_a, state_b]
        
        # Detect state changes and determine direction
        if current_state != self.last_encoder_state:
            direction = self._decode_encoder_direction(self.last_encoder_state, current_state)
            if direction != 0:
                self._handle_encoder_movement(direction)
                
        self.last_encoder_state = current_state
        
        # Check axis selection pins
        self._update_active_axis()
        
        # Check multiplier pins
        self._update_active_multiplier()
        
        return eventtime + 0.01  # Check every 10ms
        
    def _decode_encoder_direction(self, old_state, new_state):
        """Decode encoder direction from state changes"""
        # Standard quadrature encoder decoding
        old_a, old_b = old_state
        new_a, new_b = new_state
        
        # Gray code sequence for clockwise: 00 -> 01 -> 11 -> 10 -> 00
        transitions = {
            (False, False, False, True): 1,   # 00 -> 01: CW
            (False, True, True, True): 1,     # 01 -> 11: CW
            (True, True, True, False): 1,     # 11 -> 10: CW
            (True, False, False, False): 1,   # 10 -> 00: CW
            (False, True, False, False): -1,  # 01 -> 00: CCW
            (True, True, False, True): -1,    # 11 -> 01: CCW
            (True, False, True, True): -1,    # 10 -> 11: CCW
            (False, False, True, False): -1,  # 00 -> 10: CCW
        }
        
        key = (old_a, old_b, new_a, new_b)
        return transitions.get(key, 0)
        
    def _update_active_axis(self):
        """Update active axis based on pin states"""
        for axis, pin in self.axis_pins.items():
            if self._get_pin_state(pin):
                self.active_axis = axis
                break
                
    def _update_active_multiplier(self):
        """Update active step multiplier based on pin states"""
        for multiplier_idx, pin in self.multiplier_pins.items():
            if self._get_pin_state(pin):
                self.active_multiplier = multiplier_idx
                break
                
    def _handle_encoder_movement(self, direction):
        """Handle encoder movement"""
        if not self.enabled:
            return
            
        # Calculate step size
        step_size = self.step_multipliers[self.active_multiplier] if self.active_multiplier < len(self.step_multipliers) else self.default_step
        move_distance = direction * step_size
        
        # Execute jog move
        self._jog_axis(self.active_axis, move_distance)
        
    def _jog_axis(self, axis, distance):
        """Perform jog movement on specified axis"""
        try:
            toolhead = self.printer.lookup_object('toolhead')
            current_pos = list(toolhead.get_position())
            
            # Map axis to position index
            axis_map = {'x': 0, 'y': 1, 'z': 2, 'e': 3, 'a': 3, 'b': 3}  # A/B would need multi-axis support
            
            if axis.lower() in axis_map:
                axis_idx = axis_map[axis.lower()]
                current_pos[axis_idx] += distance
                
                # Move at limited velocity for safety
                velocity = min(self.max_velocity, toolhead.max_velocity)
                toolhead.move(current_pos, velocity)
                
        except Exception as e:
            logging.warning("Handwheel jog failed: %s", str(e))
            
    cmd_HANDWHEEL_ENABLE_help = "Enable handwheel control"
    def cmd_HANDWHEEL_ENABLE(self, gcmd):
        """Enable handwheel control"""
        self.enabled = True
        gcmd.respond_info("Handwheel enabled")
        
    cmd_HANDWHEEL_DISABLE_help = "Disable handwheel control"
    def cmd_HANDWHEEL_DISABLE(self, gcmd):
        """Disable handwheel control"""
        self.enabled = False
        gcmd.respond_info("Handwheel disabled")
        
    cmd_HANDWHEEL_JOG_help = "Manual jog command (HANDWHEEL_JOG AXIS=<x|y|z> DISTANCE=<mm>)"
    def cmd_HANDWHEEL_JOG(self, gcmd):
        """Manual jog command"""
        axis = gcmd.get('AXIS', 'x').lower()
        distance = gcmd.get_float('DISTANCE', 0.)
        
        if distance != 0:
            self._jog_axis(axis, distance)
            gcmd.respond_info("Jogged %s axis %.3fmm" % (axis.upper(), distance))
        else:
            gcmd.respond_info("No jog distance specified")
            
    cmd_HANDWHEEL_SET_AXIS_help = "Set active handwheel axis (HANDWHEEL_SET_AXIS AXIS=<x|y|z>)"
    def cmd_HANDWHEEL_SET_AXIS(self, gcmd):
        """Set active handwheel axis"""
        axis = gcmd.get('AXIS', self.active_axis).lower()
        valid_axes = ['x', 'y', 'z', 'a', 'b', 'e']
        
        if axis in valid_axes:
            self.active_axis = axis
            gcmd.respond_info("Handwheel axis set to %s" % axis.upper())
        else:
            gcmd.respond_info("Invalid axis. Valid axes: %s" % ', '.join(valid_axes))
            
    cmd_HANDWHEEL_SET_STEP_help = "Set handwheel step size (HANDWHEEL_SET_STEP SIZE=<mm>)"
    def cmd_HANDWHEEL_SET_STEP(self, gcmd):
        """Set handwheel step size"""
        step_size = gcmd.get_float('SIZE', self.default_step, above=0.)
        
        # Find closest multiplier or add new one
        if step_size not in self.step_multipliers:
            self.step_multipliers.append(step_size)
            self.step_multipliers.sort()
            
        self.active_multiplier = self.step_multipliers.index(step_size)
        gcmd.respond_info("Handwheel step size set to %.4fmm" % step_size)
        
    def get_status(self, eventtime):
        """Return handwheel status"""
        return {
            'enabled': self.enabled,
            'active_axis': self.active_axis,
            'step_size': self.step_multipliers[self.active_multiplier] if self.active_multiplier < len(self.step_multipliers) else self.default_step,
            'encoder_position': self.encoder_position
        }

def load_config(config):
    return Handwheel(config)

def load_config_prefix(config):
    return Handwheel(config)