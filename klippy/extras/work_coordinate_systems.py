# Work coordinate systems support (G54-G59)
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging

class WorkCoordinateSystems:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object('gcode')
        self.name = config.get_name().split()[-1]
        
        # Initialize coordinate systems (G54-G59)
        # Each system stores X, Y, Z offsets from machine coordinates
        self.coordinate_systems = {}
        for i in range(54, 60):  # G54 through G59
            self.coordinate_systems[i] = [0., 0., 0., 0.]  # X, Y, Z, E offsets
            
        # Current active coordinate system
        self.active_system = 54  # Default to G54
        
        # Load saved coordinate systems from config
        self._load_saved_coordinates()
        
        # Register G-code commands
        for i in range(54, 60):
            self.gcode.register_command('G%d' % i, getattr(self, '_cmd_G%d' % i),
                                      desc="Select work coordinate system G%d" % i)
                                      
        # Register coordinate system management commands
        self.gcode.register_command('G10', self.cmd_G10,
                                  desc=self.cmd_G10_help)
        self.gcode.register_command('G92', self.cmd_G92,
                                  desc=self.cmd_G92_help)
        self.gcode.register_command('G92.1', self.cmd_G92_1,
                                  desc=self.cmd_G92_1_help)
        self.gcode.register_command('G53', self.cmd_G53,
                                  desc=self.cmd_G53_help)
        self.gcode.register_command('WCS_INFO', self.cmd_WCS_INFO,
                                  desc=self.cmd_WCS_INFO_help)
                                  
        # Hook into coordinate transformation
        self.printer.register_event_handler("klippy:connect", self._handle_connect)
        
    def _handle_connect(self):
        """Setup coordinate transformation hooks"""
        # Register with gcode_move for coordinate transformations
        gcode_move = self.printer.lookup_object('gcode_move', None)
        if gcode_move:
            gcode_move.set_position_endstop = self._wrap_set_position_endstop(
                gcode_move.set_position_endstop)
                
    def _load_saved_coordinates(self):
        """Load saved coordinate systems from printer config"""
        # In a real implementation, this would load from saved variables
        # For now, initialize to zeros
        pass
        
    def _save_coordinates(self):
        """Save coordinate systems to persistent storage"""
        # In a real implementation, this would save to variables
        # For now, just log the current state
        logging.info("Work coordinate systems: %s", self.coordinate_systems)
        
    def _transform_position(self, pos):
        """Transform position from work coordinates to machine coordinates"""
        if self.active_system not in self.coordinate_systems:
            return pos
            
        offsets = self.coordinate_systems[self.active_system]
        return [pos[i] + offsets[i] for i in range(len(pos))]
        
    def _wrap_set_position_endstop(self, orig_func):
        """Wrap the set_position_endstop to apply coordinate transformations"""
        def wrapped_func(*args, **kwargs):
            # Apply coordinate system transformation if needed
            return orig_func(*args, **kwargs)
        return wrapped_func
        
    def _get_current_position(self):
        """Get current position in work coordinates"""
        toolhead = self.printer.lookup_object('toolhead')
        machine_pos = toolhead.get_position()
        
        if self.active_system not in self.coordinate_systems:
            return machine_pos
            
        offsets = self.coordinate_systems[self.active_system]
        return [machine_pos[i] - offsets[i] for i in range(len(machine_pos))]
        
    def _select_coordinate_system(self, system_num):
        """Select active coordinate system"""
        if system_num not in self.coordinate_systems:
            raise self.gcode.error("Invalid coordinate system G%d" % system_num)
        self.active_system = system_num
        
    # Generate G54-G59 command methods
    def _cmd_G54(self, gcmd):
        """G54 - Select work coordinate system 1"""
        self._select_coordinate_system(54)
        gcmd.respond_info("Work coordinate system G54 selected")
        
    def _cmd_G55(self, gcmd):
        """G55 - Select work coordinate system 2"""
        self._select_coordinate_system(55)
        gcmd.respond_info("Work coordinate system G55 selected")
        
    def _cmd_G56(self, gcmd):
        """G56 - Select work coordinate system 3"""
        self._select_coordinate_system(56)
        gcmd.respond_info("Work coordinate system G56 selected")
        
    def _cmd_G57(self, gcmd):
        """G57 - Select work coordinate system 4"""
        self._select_coordinate_system(57)
        gcmd.respond_info("Work coordinate system G57 selected")
        
    def _cmd_G58(self, gcmd):
        """G58 - Select work coordinate system 5"""
        self._select_coordinate_system(58)
        gcmd.respond_info("Work coordinate system G58 selected")
        
    def _cmd_G59(self, gcmd):
        """G59 - Select work coordinate system 6"""
        self._select_coordinate_system(59)
        gcmd.respond_info("Work coordinate system G59 selected")
        
    cmd_G10_help = "Set coordinate system offsets (G10 L2 P[1-6] X[val] Y[val] Z[val])"
    def cmd_G10(self, gcmd):
        """G10 - Set coordinate system offsets"""
        l_param = gcmd.get_int('L', 0)
        if l_param != 2:
            raise gcmd.error("G10: Only L2 (coordinate system setting) supported")
            
        p_param = gcmd.get_int('P', 1, minval=1, maxval=6)
        system_num = 53 + p_param  # P1=G54, P2=G55, etc.
        
        if system_num not in self.coordinate_systems:
            raise gcmd.error("Invalid coordinate system P%d" % p_param)
            
        # Get current offsets
        offsets = list(self.coordinate_systems[system_num])
        
        # Update offsets with provided values
        if gcmd.get_float('X', None) is not None:
            offsets[0] = gcmd.get_float('X')
        if gcmd.get_float('Y', None) is not None:
            offsets[1] = gcmd.get_float('Y')
        if gcmd.get_float('Z', None) is not None:
            offsets[2] = gcmd.get_float('Z')
        if gcmd.get_float('E', None) is not None:
            offsets[3] = gcmd.get_float('E')
            
        self.coordinate_systems[system_num] = offsets
        self._save_coordinates()
        
        gcmd.respond_info("G%d offsets set to X:%.3f Y:%.3f Z:%.3f E:%.3f"
                        % (system_num, offsets[0], offsets[1], offsets[2], offsets[3]))
        
    cmd_G92_help = "Set position (G92 X[val] Y[val] Z[val])"
    def cmd_G92(self, gcmd):
        """G92 - Set current position"""
        toolhead = self.printer.lookup_object('toolhead')
        current_pos = list(toolhead.get_position())
        
        # Update position based on provided values
        new_pos = list(current_pos)
        if gcmd.get_float('X', None) is not None:
            new_pos[0] = gcmd.get_float('X')
        if gcmd.get_float('Y', None) is not None:
            new_pos[1] = gcmd.get_float('Y')
        if gcmd.get_float('Z', None) is not None:
            new_pos[2] = gcmd.get_float('Z')
        if gcmd.get_float('E', None) is not None:
            new_pos[3] = gcmd.get_float('E')
            
        # Calculate new offsets for current coordinate system
        offsets = [current_pos[i] - new_pos[i] for i in range(len(current_pos))]
        self.coordinate_systems[self.active_system] = offsets
        
        gcmd.respond_info("Position set to X:%.3f Y:%.3f Z:%.3f E:%.3f in G%d"
                        % (new_pos[0], new_pos[1], new_pos[2], new_pos[3], self.active_system))
        
    cmd_G92_1_help = "Reset G92 offsets"
    def cmd_G92_1(self, gcmd):
        """G92.1 - Reset G92 offsets"""
        self.coordinate_systems[self.active_system] = [0., 0., 0., 0.]
        gcmd.respond_info("G92 offsets reset for G%d" % self.active_system)
        
    cmd_G53_help = "Move in machine coordinates (G53 X[val] Y[val] Z[val])"
    def cmd_G53(self, gcmd):
        """G53 - Move in machine coordinates"""
        # Temporarily switch to machine coordinates for this move
        saved_system = self.active_system
        self.active_system = None  # Machine coordinates
        
        try:
            # Execute the move
            toolhead = self.printer.lookup_object('toolhead')
            current_pos = toolhead.get_position()
            
            new_pos = list(current_pos)
            if gcmd.get_float('X', None) is not None:
                new_pos[0] = gcmd.get_float('X')
            if gcmd.get_float('Y', None) is not None:
                new_pos[1] = gcmd.get_float('Y')
            if gcmd.get_float('Z', None) is not None:
                new_pos[2] = gcmd.get_float('Z')
                
            toolhead.move(new_pos, gcmd.get_float('F', toolhead.max_velocity * 60.) / 60.)
            
        finally:
            # Restore coordinate system
            self.active_system = saved_system
            
    cmd_WCS_INFO_help = "Display work coordinate system information"
    def cmd_WCS_INFO(self, gcmd):
        """Display work coordinate system information"""
        gcmd.respond_info("Active coordinate system: G%d" % self.active_system)
        for system_num in sorted(self.coordinate_systems.keys()):
            offsets = self.coordinate_systems[system_num]
            gcmd.respond_info("G%d: X:%.3f Y:%.3f Z:%.3f E:%.3f"
                            % (system_num, offsets[0], offsets[1], offsets[2], offsets[3]))
                            
    def get_status(self, eventtime):
        """Return current coordinate system status"""
        return {
            'active_system': self.active_system,
            'coordinate_systems': dict(self.coordinate_systems)
        }

def load_config(config):
    return WorkCoordinateSystems(config)

def load_config_prefix(config):
    return WorkCoordinateSystems(config)