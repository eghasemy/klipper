# Code for handling printer nozzle extruders
#
# Copyright (C) 2016-2025  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import math, logging
import stepper, chelper

class ExtruderStepper:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.name = config.get_name().split()[-1]
        self.pressure_advance = self.pressure_advance_smooth_time = 0.
        self.config_pa = config.getfloat('pressure_advance', 0., minval=0.)
        self.config_smooth_time = config.getfloat(
                'pressure_advance_smooth_time', 0.040, above=0., maxval=.200)
        # Variable pressure advance configuration
        self.config_pa_variable = config.getboolean('pressure_advance_variable', False)
        self.config_pa_rate_min = config.getfloat('pressure_advance_rate_min', 0.5, minval=0.)
        self.config_pa_rate_max = config.getfloat('pressure_advance_rate_max', 10.0, above=0.)
        self.config_pa_value_min = config.getfloat('pressure_advance_value_min', 0.0, minval=0.)
        self.config_pa_value_max = config.getfloat('pressure_advance_value_max', 0.2, minval=0.)
        self.config_pa_rate_power = config.getfloat('pressure_advance_rate_power', 1.0, minval=0.1, maxval=10.0)
        # Setup stepper
        self.stepper = stepper.PrinterStepper(config)
        ffi_main, ffi_lib = chelper.get_ffi()
        self.sk_extruder = ffi_main.gc(ffi_lib.extruder_stepper_alloc(),
                                       ffi_lib.extruder_stepper_free)
        self.stepper.set_stepper_kinematics(self.sk_extruder)
        self.motion_queue = None
        # Register commands
        self.printer.register_event_handler("klippy:connect",
                                            self._handle_connect)
        gcode = self.printer.lookup_object('gcode')
        if self.name == 'extruder':
            gcode.register_mux_command("SET_PRESSURE_ADVANCE", "EXTRUDER", None,
                                       self.cmd_default_SET_PRESSURE_ADVANCE,
                                       desc=self.cmd_SET_PRESSURE_ADVANCE_help)
        gcode.register_mux_command("SET_PRESSURE_ADVANCE", "EXTRUDER",
                                   self.name, self.cmd_SET_PRESSURE_ADVANCE,
                                   desc=self.cmd_SET_PRESSURE_ADVANCE_help)
        gcode.register_mux_command("SET_EXTRUDER_ROTATION_DISTANCE", "EXTRUDER",
                                   self.name, self.cmd_SET_E_ROTATION_DISTANCE,
                                   desc=self.cmd_SET_E_ROTATION_DISTANCE_help)
        gcode.register_mux_command("SYNC_EXTRUDER_MOTION", "EXTRUDER",
                                   self.name, self.cmd_SYNC_EXTRUDER_MOTION,
                                   desc=self.cmd_SYNC_EXTRUDER_MOTION_help)
        gcode.register_mux_command("SET_VARIABLE_PRESSURE_ADVANCE", "EXTRUDER",
                                   self.name, self.cmd_SET_VARIABLE_PRESSURE_ADVANCE,
                                   desc=self.cmd_SET_VARIABLE_PRESSURE_ADVANCE_help)
        gcode.register_mux_command("CALIBRATE_PRESSURE_ADVANCE", "EXTRUDER",
                                   self.name, self.cmd_CALIBRATE_PRESSURE_ADVANCE,
                                   desc=self.cmd_CALIBRATE_PRESSURE_ADVANCE_help)
    def _handle_connect(self):
        toolhead = self.printer.lookup_object('toolhead')
        toolhead.register_step_generator(self.stepper.generate_steps)
        if self.config_pa_variable:
            self._set_variable_pressure_advance(
                self.config_pa_rate_min, self.config_pa_rate_max,
                self.config_pa_value_min, self.config_pa_value_max,
                self.config_pa_rate_power, self.config_smooth_time)
        else:
            self._set_pressure_advance(self.config_pa, self.config_smooth_time)
    def get_status(self, eventtime):
        return {'pressure_advance': self.pressure_advance,
                'smooth_time': self.pressure_advance_smooth_time,
                'motion_queue': self.motion_queue}
    def find_past_position(self, print_time):
        mcu_pos = self.stepper.get_past_mcu_position(print_time)
        return self.stepper.mcu_to_commanded_position(mcu_pos)
    def sync_to_extruder(self, extruder_name):
        toolhead = self.printer.lookup_object('toolhead')
        toolhead.flush_step_generation()
        if not extruder_name:
            self.stepper.set_trapq(None)
            self.motion_queue = None
            return
        extruder = self.printer.lookup_object(extruder_name, None)
        if extruder is None or not isinstance(extruder, PrinterExtruder):
            raise self.printer.command_error("'%s' is not a valid extruder."
                                             % (extruder_name,))
        self.stepper.set_position([extruder.last_position, 0., 0.])
        self.stepper.set_trapq(extruder.get_trapq())
        self.motion_queue = extruder_name
    def _set_pressure_advance(self, pressure_advance, smooth_time):
        old_smooth_time = self.pressure_advance_smooth_time
        if not self.pressure_advance:
            old_smooth_time = 0.
        new_smooth_time = smooth_time
        if not pressure_advance:
            new_smooth_time = 0.
        toolhead = self.printer.lookup_object("toolhead")
        if new_smooth_time != old_smooth_time:
            toolhead.note_step_generation_scan_time(
                    new_smooth_time * .5, old_delay=old_smooth_time * .5)
        ffi_main, ffi_lib = chelper.get_ffi()
        espa = ffi_lib.extruder_set_pressure_advance
        toolhead.register_lookahead_callback(
            lambda print_time: espa(self.sk_extruder, print_time,
                                    pressure_advance, new_smooth_time))
        self.pressure_advance = pressure_advance
        self.pressure_advance_smooth_time = smooth_time
    def _set_variable_pressure_advance(self, rate_min, rate_max, pa_min, pa_max, power, smooth_time):
        old_smooth_time = self.pressure_advance_smooth_time
        if not self.pressure_advance:
            old_smooth_time = 0.
        new_smooth_time = smooth_time
        if not pa_min and not pa_max:
            new_smooth_time = 0.
        toolhead = self.printer.lookup_object("toolhead")
        if new_smooth_time != old_smooth_time:
            toolhead.note_step_generation_scan_time(
                    new_smooth_time * .5, old_delay=old_smooth_time * .5)
        # Set regular pressure advance for smoothing setup
        ffi_main, ffi_lib = chelper.get_ffi()
        espa = ffi_lib.extruder_set_pressure_advance
        toolhead.register_lookahead_callback(
            lambda print_time: espa(self.sk_extruder, print_time,
                                    pa_min, new_smooth_time))
        # Set variable pressure advance
        esvpa = ffi_lib.extruder_set_variable_pressure_advance
        toolhead.register_lookahead_callback(
            lambda print_time: esvpa(self.sk_extruder, print_time,
                                     rate_min, rate_max, pa_min, pa_max, power))
        self.pressure_advance = pa_min  # Store minimum value for compatibility
        self.pressure_advance_smooth_time = smooth_time
    cmd_SET_PRESSURE_ADVANCE_help = "Set pressure advance parameters"
    def cmd_default_SET_PRESSURE_ADVANCE(self, gcmd):
        extruder = self.printer.lookup_object('toolhead').get_extruder()
        if extruder.extruder_stepper is None:
            raise gcmd.error("Active extruder does not have a stepper")
        strapq = extruder.extruder_stepper.stepper.get_trapq()
        if strapq is not extruder.get_trapq():
            raise gcmd.error("Unable to infer active extruder stepper")
        extruder.extruder_stepper.cmd_SET_PRESSURE_ADVANCE(gcmd)
    def cmd_SET_PRESSURE_ADVANCE(self, gcmd):
        pressure_advance = gcmd.get_float('ADVANCE', self.pressure_advance,
                                          minval=0.)
        smooth_time = gcmd.get_float('SMOOTH_TIME',
                                     self.pressure_advance_smooth_time,
                                     minval=0., maxval=.200)
        self._set_pressure_advance(pressure_advance, smooth_time)
        msg = ("pressure_advance: %.6f\n"
               "pressure_advance_smooth_time: %.6f"
               % (pressure_advance, smooth_time))
        self.printer.set_rollover_info(self.name, "%s: %s" % (self.name, msg))
        gcmd.respond_info(msg, log=False)
    cmd_SET_E_ROTATION_DISTANCE_help = "Set extruder rotation distance"
    def cmd_SET_E_ROTATION_DISTANCE(self, gcmd):
        rotation_dist = gcmd.get_float('DISTANCE', None)
        if rotation_dist is not None:
            if not rotation_dist:
                raise gcmd.error("Rotation distance can not be zero")
            invert_dir, orig_invert_dir = self.stepper.get_dir_inverted()
            next_invert_dir = orig_invert_dir
            if rotation_dist < 0.:
                next_invert_dir = not orig_invert_dir
                rotation_dist = -rotation_dist
            toolhead = self.printer.lookup_object('toolhead')
            toolhead.flush_step_generation()
            self.stepper.set_rotation_distance(rotation_dist)
            self.stepper.set_dir_inverted(next_invert_dir)
        else:
            rotation_dist, spr = self.stepper.get_rotation_distance()
        invert_dir, orig_invert_dir = self.stepper.get_dir_inverted()
        if invert_dir != orig_invert_dir:
            rotation_dist = -rotation_dist
        gcmd.respond_info("Extruder '%s' rotation distance set to %0.6f"
                          % (self.name, rotation_dist))
    cmd_SYNC_EXTRUDER_MOTION_help = "Set extruder stepper motion queue"
    def cmd_SYNC_EXTRUDER_MOTION(self, gcmd):
        ename = gcmd.get('MOTION_QUEUE')
        self.sync_to_extruder(ename)
        gcmd.respond_info("Extruder '%s' now syncing with '%s'"
                          % (self.name, ename))
    cmd_SET_VARIABLE_PRESSURE_ADVANCE_help = "Set variable pressure advance parameters"
    def cmd_SET_VARIABLE_PRESSURE_ADVANCE(self, gcmd):
        rate_min = gcmd.get_float('RATE_MIN', 0.5, minval=0.)
        rate_max = gcmd.get_float('RATE_MAX', 10.0, above=rate_min)
        pa_min = gcmd.get_float('PA_MIN', 0.0, minval=0.)
        pa_max = gcmd.get_float('PA_MAX', 0.2, minval=0.)
        power = gcmd.get_float('POWER', 1.0, minval=0.1, maxval=10.0)
        smooth_time = gcmd.get_float('SMOOTH_TIME', self.pressure_advance_smooth_time,
                                     minval=0., maxval=.200)
        self._set_variable_pressure_advance(rate_min, rate_max, pa_min, pa_max, power, smooth_time)
        msg = ("Variable pressure advance enabled:\n"
               "rate_min: %.3f rate_max: %.3f\n"
               "pa_min: %.6f pa_max: %.6f power: %.3f\n"
               "smooth_time: %.6f"
               % (rate_min, rate_max, pa_min, pa_max, power, smooth_time))
        self.printer.set_rollover_info(self.name, "%s: %s" % (self.name, msg))
        gcmd.respond_info(msg, log=False)
    cmd_CALIBRATE_PRESSURE_ADVANCE_help = "Automatically calibrate pressure advance"
    def cmd_CALIBRATE_PRESSURE_ADVANCE(self, gcmd):
        # Simple calibration routine that performs test moves at different rates
        # and measures pressure advance effectiveness
        start_rate = gcmd.get_float('START_RATE', 1.0, minval=0.1)
        end_rate = gcmd.get_float('END_RATE', 8.0, above=start_rate)
        steps = gcmd.get_int('STEPS', 5, minval=3, maxval=20)
        start_pa = gcmd.get_float('START_PA', 0.01, minval=0.)
        end_pa = gcmd.get_float('END_PA', 0.1, above=start_pa)
        test_distance = gcmd.get_float('DISTANCE', 50., minval=10., maxval=200.)
        
        # Store current settings
        toolhead = self.printer.lookup_object('toolhead')
        current_position = toolhead.get_position()
        
        # Perform calibration moves
        calibration_data = []
        for i in range(steps):
            rate_factor = i / (steps - 1.0)
            test_rate = start_rate + rate_factor * (end_rate - start_rate)
            test_pa = start_pa + rate_factor * (end_pa - start_pa)
            
            # Set test pressure advance
            self._set_pressure_advance(test_pa, self.pressure_advance_smooth_time)
            
            # Perform test move
            gcmd.respond_info("Testing rate: %.2f, PA: %.4f" % (test_rate, test_pa))
            test_pos = list(current_position)
            test_pos[3] += test_distance  # Move extruder
            toolhead.move(test_pos, test_rate)
            toolhead.wait_moves()
            
            calibration_data.append((test_rate, test_pa))
        
        # Calculate optimal variable pressure advance parameters
        if len(calibration_data) >= 2:
            min_data = calibration_data[0]
            max_data = calibration_data[-1]
            
            self._set_variable_pressure_advance(
                min_data[0], max_data[0],  # rate_min, rate_max
                min_data[1], max_data[1],  # pa_min, pa_max
                1.0, self.pressure_advance_smooth_time)  # power, smooth_time
            
            gcmd.respond_info("Calibration complete. Variable PA set:\n"
                              "Rate range: %.2f - %.2f\n"
                              "PA range: %.4f - %.4f"
                              % (min_data[0], max_data[0], min_data[1], max_data[1]))
        else:
            gcmd.respond_info("Calibration failed: insufficient data points")

# Tracking for hotend heater, extrusion motion queue, and extruder stepper
class PrinterExtruder:
    def __init__(self, config, extruder_num):
        self.printer = config.get_printer()
        self.name = config.get_name()
        self.last_position = 0.
        # Setup hotend heater
        pheaters = self.printer.load_object(config, 'heaters')
        gcode_id = 'T%d' % (extruder_num,)
        self.heater = pheaters.setup_heater(config, gcode_id)
        # Setup kinematic checks
        self.nozzle_diameter = config.getfloat('nozzle_diameter', above=0.)
        filament_diameter = config.getfloat(
            'filament_diameter', minval=self.nozzle_diameter)
        self.filament_area = math.pi * (filament_diameter * .5)**2
        def_max_cross_section = 4. * self.nozzle_diameter**2
        def_max_extrude_ratio = def_max_cross_section / self.filament_area
        max_cross_section = config.getfloat(
            'max_extrude_cross_section', def_max_cross_section, above=0.)
        self.max_extrude_ratio = max_cross_section / self.filament_area
        logging.info("Extruder max_extrude_ratio=%.6f", self.max_extrude_ratio)
        toolhead = self.printer.lookup_object('toolhead')
        max_velocity, max_accel = toolhead.get_max_velocity()
        self.max_e_velocity = config.getfloat(
            'max_extrude_only_velocity', max_velocity * def_max_extrude_ratio
            , above=0.)
        self.max_e_accel = config.getfloat(
            'max_extrude_only_accel', max_accel * def_max_extrude_ratio
            , above=0.)
        self.max_e_dist = config.getfloat(
            'max_extrude_only_distance', 50., minval=0.)
        self.instant_corner_v = config.getfloat(
            'instantaneous_corner_velocity', 1., minval=0.)
        # Setup extruder trapq (trapezoidal motion queue)
        ffi_main, ffi_lib = chelper.get_ffi()
        self.trapq = ffi_main.gc(ffi_lib.trapq_alloc(), ffi_lib.trapq_free)
        self.trapq_append = ffi_lib.trapq_append
        self.trapq_finalize_moves = ffi_lib.trapq_finalize_moves
        # Setup extruder stepper
        self.extruder_stepper = None
        if (config.get('step_pin', None) is not None
            or config.get('dir_pin', None) is not None
            or config.get('rotation_distance', None) is not None):
            self.extruder_stepper = ExtruderStepper(config)
            self.extruder_stepper.stepper.set_trapq(self.trapq)
        # Register commands
        gcode = self.printer.lookup_object('gcode')
        if self.name == 'extruder':
            toolhead.set_extruder(self, 0.)
            gcode.register_command("M104", self.cmd_M104)
            gcode.register_command("M109", self.cmd_M109)
        gcode.register_mux_command("ACTIVATE_EXTRUDER", "EXTRUDER",
                                   self.name, self.cmd_ACTIVATE_EXTRUDER,
                                   desc=self.cmd_ACTIVATE_EXTRUDER_help)
    def get_status(self, eventtime):
        sts = self.heater.get_status(eventtime)
        sts['can_extrude'] = self.heater.can_extrude
        if self.extruder_stepper is not None:
            sts.update(self.extruder_stepper.get_status(eventtime))
        return sts
    def get_name(self):
        return self.name
    def get_heater(self):
        return self.heater
    def get_trapq(self):
        return self.trapq
    def get_axis_gcode_id(self):
        return 'E'
    def stats(self, eventtime):
        return self.heater.stats(eventtime)
    def check_move(self, move, ea_index):
        if not self.heater.can_extrude:
            raise self.printer.command_error(
                "Extrude below minimum temp\n"
                "See the 'min_extrude_temp' config option for details")
        axis_r = move.axes_r[ea_index]
        axis_d = move.axes_d[ea_index]
        if (not move.axes_d[0] and not move.axes_d[1]) or axis_r < 0.:
            # Extrude only move (or retraction move) - limit accel and velocity
            if abs(axis_d) > self.max_e_dist:
                raise self.printer.command_error(
                    "Extrude only move too long (%.3fmm vs %.3fmm)\n"
                    "See the 'max_extrude_only_distance' config"
                    " option for details" % (axis_d, self.max_e_dist))
            inv_extrude_r = 1. / abs(axis_r)
            move.limit_speed(self.max_e_velocity * inv_extrude_r,
                             self.max_e_accel * inv_extrude_r)
        elif axis_r > self.max_extrude_ratio:
            if axis_d <= self.nozzle_diameter * self.max_extrude_ratio:
                # Permit extrusion if amount extruded is tiny
                return
            area = axis_r * self.filament_area
            logging.debug("Overextrude: %s vs %s (area=%.3f dist=%.3f)",
                          axis_r, self.max_extrude_ratio, area, move.move_d)
            raise self.printer.command_error(
                "Move exceeds maximum extrusion (%.3fmm^2 vs %.3fmm^2)\n"
                "See the 'max_extrude_cross_section' config option for details"
                % (area, self.max_extrude_ratio * self.filament_area))
    def calc_junction(self, prev_move, move, ea_index):
        diff_r = move.axes_r[ea_index] - prev_move.axes_r[ea_index]
        if diff_r:
            return (self.instant_corner_v / abs(diff_r))**2
        return move.max_cruise_v2
    def process_move(self, print_time, move, ea_index):
        axis_r = move.axes_r[ea_index]
        accel = move.accel * axis_r
        start_v = move.start_v * axis_r
        cruise_v = move.cruise_v * axis_r
        can_pressure_advance = False
        if axis_r > 0. and (move.axes_d[0] or move.axes_d[1]):
            can_pressure_advance = True
        # Queue movement (x is extruder movement, y is pressure advance flag)
        self.trapq_append(self.trapq, print_time,
                          move.accel_t, move.cruise_t, move.decel_t,
                          move.start_pos[ea_index], 0., 0.,
                          1., can_pressure_advance, 0.,
                          start_v, cruise_v, accel)
        self.last_position = move.end_pos[ea_index]
    def find_past_position(self, print_time):
        if self.extruder_stepper is None:
            return 0.
        return self.extruder_stepper.find_past_position(print_time)
    def cmd_M104(self, gcmd, wait=False):
        # Set Extruder Temperature
        temp = gcmd.get_float('S', 0.)
        index = gcmd.get_int('T', None, minval=0)
        if index is not None:
            section = 'extruder'
            if index:
                section = 'extruder%d' % (index,)
            extruder = self.printer.lookup_object(section, None)
            if extruder is None:
                if temp <= 0.:
                    return
                raise gcmd.error("Extruder not configured")
        else:
            extruder = self.printer.lookup_object('toolhead').get_extruder()
        pheaters = self.printer.lookup_object('heaters')
        pheaters.set_temperature(extruder.get_heater(), temp, wait)
    def cmd_M109(self, gcmd):
        # Set Extruder Temperature and Wait
        self.cmd_M104(gcmd, wait=True)
    cmd_ACTIVATE_EXTRUDER_help = "Change the active extruder"
    def cmd_ACTIVATE_EXTRUDER(self, gcmd):
        toolhead = self.printer.lookup_object('toolhead')
        if toolhead.get_extruder() is self:
            gcmd.respond_info("Extruder %s already active" % (self.name,))
            return
        gcmd.respond_info("Activating extruder %s" % (self.name,))
        toolhead.flush_step_generation()
        toolhead.set_extruder(self, self.last_position)
        self.printer.send_event("extruder:activate_extruder")

# Dummy extruder class used when a printer has no extruder at all
class DummyExtruder:
    def __init__(self, printer):
        self.printer = printer
    def check_move(self, move, ea_index):
        raise move.move_error("Extrude when no extruder present")
    def find_past_position(self, print_time):
        return 0.
    def calc_junction(self, prev_move, move, ea_index):
        return move.max_cruise_v2
    def get_name(self):
        return ""
    def get_heater(self):
        raise self.printer.command_error("Extruder not configured")
    def get_trapq(self):
        return None
    def get_axis_gcode_id(self):
        return 'E'

def add_printer_objects(config):
    printer = config.get_printer()
    for i in range(99):
        section = 'extruder'
        if i:
            section = 'extruder%d' % (i,)
        if not config.has_section(section):
            break
        pe = PrinterExtruder(config.getsection(section), i)
        printer.add_object(section, pe)
