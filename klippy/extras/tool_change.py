# Tool change support for CNC machines
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging

class ToolChange:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object('gcode')
        self.name = config.get_name().split()[-1]
        
        # Tool configuration
        self.max_tool = config.getint('max_tool', 99, minval=0)
        self.current_tool = 0
        
        # Tool change macros
        self.tool_change_macro = config.get('tool_change_macro', 'TOOL_CHANGE')
        self.tool_pre_change_macro = config.get('tool_pre_change_macro', None)
        self.tool_post_change_macro = config.get('tool_post_change_macro', None)
        
        # Position configuration
        self.park_x = config.getfloat('park_x', None)
        self.park_y = config.getfloat('park_y', None) 
        self.park_z = config.getfloat('park_z', None)
        
        # Register T-commands dynamically
        for i in range(self.max_tool + 1):
            self.gcode.register_command('T%d' % i, self._cmd_T,
                                      desc="Select tool %d" % i)
                                      
        # Register utility commands
        self.gcode.register_command('TOOL_CHANGE', self.cmd_TOOL_CHANGE,
                                  desc=self.cmd_TOOL_CHANGE_help)
        self.gcode.register_command('GET_CURRENT_TOOL', self.cmd_GET_CURRENT_TOOL,
                                  desc=self.cmd_GET_CURRENT_TOOL_help)
                                  
    def _park_spindle(self):
        """Park spindle at configured position"""
        if self.park_x is not None or self.park_y is not None or self.park_z is not None:
            gcode_cmd = "G0"
            if self.park_x is not None:
                gcode_cmd += " X%.3f" % self.park_x
            if self.park_y is not None:
                gcode_cmd += " Y%.3f" % self.park_y
            if self.park_z is not None:
                gcode_cmd += " Z%.3f" % self.park_z
            self.gcode.run_script_from_command(gcode_cmd)
            
    def _run_macro(self, macro_name, **kwargs):
        """Run a macro if it exists"""
        if macro_name:
            try:
                # Build macro command with parameters
                cmd = macro_name
                for key, value in kwargs.items():
                    cmd += " %s=%s" % (key.upper(), value)
                self.gcode.run_script_from_command(cmd)
            except Exception as e:
                logging.warning("Tool change macro '%s' failed: %s", macro_name, str(e))
                
    def _change_tool(self, new_tool):
        """Perform tool change sequence"""
        if new_tool == self.current_tool:
            return  # No change needed
            
        old_tool = self.current_tool
        
        # Run pre-change macro
        if self.tool_pre_change_macro:
            self._run_macro(self.tool_pre_change_macro, 
                          OLD_TOOL=old_tool, NEW_TOOL=new_tool)
            
        # Turn off spindle before tool change
        try:
            spindle = self.printer.lookup_object('spindle', None)
            if spindle and spindle.is_on:
                self.gcode.run_script_from_command("M5")
        except:
            pass  # Spindle not configured
            
        # Park spindle
        self._park_spindle()
        
        # Run main tool change macro
        self._run_macro(self.tool_change_macro,
                       OLD_TOOL=old_tool, NEW_TOOL=new_tool)
                       
        # Update current tool
        self.current_tool = new_tool
        
        # Run post-change macro
        if self.tool_post_change_macro:
            self._run_macro(self.tool_post_change_macro,
                          OLD_TOOL=old_tool, NEW_TOOL=new_tool)
                          
    def _cmd_T(self, gcmd):
        """Handle T-commands (T0, T1, etc.)"""
        # Extract tool number from command
        command = gcmd.get_command()
        if command.startswith('T') and len(command) > 1:
            try:
                tool_num = int(command[1:])
                if 0 <= tool_num <= self.max_tool:
                    self._change_tool(tool_num)
                    gcmd.respond_info("Tool changed to T%d" % tool_num)
                else:
                    raise gcmd.error("Tool number %d out of range (0-%d)" 
                                   % (tool_num, self.max_tool))
            except ValueError:
                raise gcmd.error("Invalid tool number in command: %s" % command)
                
    cmd_TOOL_CHANGE_help = "Manually change tool (TOOL_CHANGE TOOL=<num>)"
    def cmd_TOOL_CHANGE(self, gcmd):
        """Manual tool change command"""
        tool_num = gcmd.get_int('TOOL', self.current_tool, minval=0, maxval=self.max_tool)
        self._change_tool(tool_num)
        gcmd.respond_info("Tool changed to T%d" % tool_num)
        
    cmd_GET_CURRENT_TOOL_help = "Get current tool number"
    def cmd_GET_CURRENT_TOOL(self, gcmd):
        """Get current tool"""
        gcmd.respond_info("Current tool: T%d" % self.current_tool)
        
    def get_status(self, eventtime):
        """Return current tool status"""
        return {
            'current_tool': self.current_tool,
            'max_tool': self.max_tool
        }

def load_config(config):
    return ToolChange(config)

def load_config_prefix(config):
    return ToolChange(config)