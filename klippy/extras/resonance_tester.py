# A utility class to test resonances of the printer
#
# Copyright (C) 2020-2024  Dmitry Butyugin <dmbutyugin@google.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging, math, os, time
from . import shaper_calibrate

class TestAxis:
    def __init__(self, axis=None, vib_dir=None):
        if axis is None:
            self._name = "axis=%.3f,%.3f" % (vib_dir[0], vib_dir[1])
        else:
            self._name = axis
        if vib_dir is None:
            self._vib_dir = (1., 0.) if axis == 'x' else (0., 1.)
        else:
            s = math.sqrt(sum([d*d for d in vib_dir]))
            self._vib_dir = [d / s for d in vib_dir]
    def matches(self, chip_axis):
        if self._vib_dir[0] and 'x' in chip_axis:
            return True
        if self._vib_dir[1] and 'y' in chip_axis:
            return True
        return False
    def get_name(self):
        return self._name
    def get_point(self, l):
        return (self._vib_dir[0] * l, self._vib_dir[1] * l)

def _parse_axis(gcmd, raw_axis):
    if raw_axis is None:
        return None
    raw_axis = raw_axis.lower()
    if raw_axis in ['x', 'y']:
        return TestAxis(axis=raw_axis)
    dirs = raw_axis.split(',')
    if len(dirs) != 2:
        raise gcmd.error("Invalid format of axis '%s'" % (raw_axis,))
    try:
        dir_x = float(dirs[0].strip())
        dir_y = float(dirs[1].strip())
    except:
        raise gcmd.error(
                "Unable to parse axis direction '%s'" % (raw_axis,))
    return TestAxis(vib_dir=(dir_x, dir_y))

class VibrationPulseTestGenerator:
    def __init__(self, config):
        self.min_freq = config.getfloat('min_freq', 5., minval=1.)
        self.max_freq = config.getfloat('max_freq', 135.,
                                        minval=self.min_freq, maxval=300.)
        self.accel_per_hz = config.getfloat('accel_per_hz', 60., above=0.)
        self.hz_per_sec = config.getfloat('hz_per_sec', 1.,
                                          minval=0.1, maxval=2.)
    def prepare_test(self, gcmd):
        self.freq_start = gcmd.get_float("FREQ_START", self.min_freq, minval=1.)
        self.freq_end = gcmd.get_float("FREQ_END", self.max_freq,
                                       minval=self.freq_start, maxval=300.)
        self.test_accel_per_hz = gcmd.get_float("ACCEL_PER_HZ",
                                                self.accel_per_hz, above=0.)
        self.test_hz_per_sec = gcmd.get_float("HZ_PER_SEC", self.hz_per_sec,
                                              above=0., maxval=2.)
    def gen_test(self):
        freq = self.freq_start
        res = []
        sign = 1.
        time = 0.
        while freq <= self.freq_end + 0.000001:
            t_seg = .25 / freq
            accel = self.test_accel_per_hz * freq
            time += t_seg
            res.append((time, sign * accel, freq))
            time += t_seg
            res.append((time, -sign * accel, freq))
            freq += 2. * t_seg * self.test_hz_per_sec
            sign = -sign
        return res
    def get_max_freq(self):
        return self.freq_end

class SweepingVibrationsTestGenerator:
    def __init__(self, config):
        self.vibration_generator = VibrationPulseTestGenerator(config)
        self.sweeping_accel = config.getfloat('sweeping_accel', 400., above=0.)
        self.sweeping_period = config.getfloat('sweeping_period', 1.2,
                                               minval=0.)
    def prepare_test(self, gcmd):
        self.vibration_generator.prepare_test(gcmd)
        self.test_sweeping_accel = gcmd.get_float(
                "SWEEPING_ACCEL", self.sweeping_accel, above=0.)
        self.test_sweeping_period = gcmd.get_float(
                "SWEEPING_PERIOD", self.sweeping_period, minval=0.)
    def gen_test(self):
        test_seq = self.vibration_generator.gen_test()
        accel_fraction = math.sqrt(2.0) * 0.125
        if self.test_sweeping_period:
            t_rem = self.test_sweeping_period * accel_fraction
            sweeping_accel = self.test_sweeping_accel
        else:
            t_rem = float('inf')
            sweeping_accel = 0.
        res = []
        last_t = 0.
        sig = 1.
        accel_fraction += 0.25
        for next_t, accel, freq in test_seq:
            t_seg = next_t - last_t
            while t_rem <= t_seg:
                last_t += t_rem
                res.append((last_t, accel + sweeping_accel * sig, freq))
                t_seg -= t_rem
                t_rem = self.test_sweeping_period * accel_fraction
                accel_fraction = 0.5
                sig = -sig
            t_rem -= t_seg
            res.append((next_t, accel + sweeping_accel * sig, freq))
            last_t = next_t
        return res
    def get_max_freq(self):
        return self.vibration_generator.get_max_freq()

class ResonanceTestExecutor:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.gcode = self.printer.lookup_object('gcode')
    def run_test(self, test_seq, axis, gcmd):
        reactor = self.printer.get_reactor()
        toolhead = self.printer.lookup_object('toolhead')
        tpos = toolhead.get_position()
        X, Y = tpos[:2]
        # Override maximum acceleration and acceleration to
        # deceleration based on the maximum test frequency
        systime = reactor.monotonic()
        toolhead_info = toolhead.get_status(systime)
        old_max_accel = toolhead_info['max_accel']
        old_minimum_cruise_ratio = toolhead_info['minimum_cruise_ratio']
        max_accel = max([abs(a) for _, a, _ in test_seq])
        self.gcode.run_script_from_command(
            "SET_VELOCITY_LIMIT ACCEL=%.3f MINIMUM_CRUISE_RATIO=0"
            % (max_accel,))
        input_shaper = self.printer.lookup_object('input_shaper', None)
        if input_shaper is not None and not gcmd.get_int('INPUT_SHAPING', 0):
            input_shaper.disable_shaping()
            gcmd.respond_info("Disabled [input_shaper] for resonance testing")
        else:
            input_shaper = None
        last_v = last_t = last_accel = last_freq = 0.
        for next_t, accel, freq in test_seq:
            t_seg = next_t - last_t
            toolhead.cmd_M204(self.gcode.create_gcode_command(
                "M204", "M204", {"S": abs(accel)}))
            v = last_v + accel * t_seg
            abs_v = abs(v)
            if abs_v < 0.000001:
                v = abs_v = 0.
            abs_last_v = abs(last_v)
            v2 = v * v
            last_v2 = last_v * last_v
            half_inv_accel = .5 / accel
            d = (v2 - last_v2) * half_inv_accel
            dX, dY = axis.get_point(d)
            nX = X + dX
            nY = Y + dY
            toolhead.limit_next_junction_speed(abs_last_v)
            if v * last_v < 0:
                # The move first goes to a complete stop, then changes direction
                d_decel = -last_v2 * half_inv_accel
                decel_X, decel_Y = axis.get_point(d_decel)
                toolhead.move([X + decel_X, Y + decel_Y] + tpos[2:], abs_last_v)
                toolhead.move([nX, nY] + tpos[2:], abs_v)
            else:
                toolhead.move([nX, nY] + tpos[2:], max(abs_v, abs_last_v))
            if math.floor(freq) > math.floor(last_freq):
                gcmd.respond_info("Testing frequency %.0f Hz" % (freq,))
                reactor.pause(reactor.monotonic() + 0.01)
            X, Y = nX, nY
            last_t = next_t
            last_v = v
            last_accel = accel
            last_freq = freq
        if last_v:
            d_decel = -.5 * last_v2 / old_max_accel
            decel_X, decel_Y = axis.get_point(d_decel)
            toolhead.cmd_M204(self.gcode.create_gcode_command(
                "M204", "M204", {"S": old_max_accel}))
            toolhead.move([X + decel_X, Y + decel_Y] + tpos[2:], abs(last_v))
        # Restore the original acceleration values
        self.gcode.run_script_from_command(
            "SET_VELOCITY_LIMIT ACCEL=%.3f MINIMUM_CRUISE_RATIO=%.3f"
            % (old_max_accel, old_minimum_cruise_ratio))
        # Restore input shaper if it was disabled for resonance testing
        if input_shaper is not None:
            input_shaper.enable_shaping()
            gcmd.respond_info("Re-enabled [input_shaper]")

class ResonanceTester:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.move_speed = config.getfloat('move_speed', 50., above=0.)
        self.generator = SweepingVibrationsTestGenerator(config)
        self.executor = ResonanceTestExecutor(config)
        if not config.get('accel_chip_x', None):
            self.accel_chip_names = [('xy', config.get('accel_chip').strip())]
        else:
            self.accel_chip_names = [
                ('x', config.get('accel_chip_x').strip()),
                ('y', config.get('accel_chip_y').strip())]
            if self.accel_chip_names[0][1] == self.accel_chip_names[1][1]:
                self.accel_chip_names = [('xy', self.accel_chip_names[0][1])]
        self.max_smoothing = config.getfloat('max_smoothing', None, minval=0.05)
        self.probe_points = config.getlists('probe_points', seps=(',', '\n'),
                                            parser=float, count=3)
        
        # Microphone support
        self.microphone_enabled = config.getboolean('enable_microphone', False)
        self.microphone_tester = None
        if self.microphone_enabled:
            try:
                from . import microphone_resonance
                self.microphone_tester = microphone_resonance.MicrophoneResonanceTester(config)
            except ImportError as e:
                logging.warning("Failed to load microphone support: %s" % str(e))
                self.microphone_enabled = False

        self.gcode = self.printer.lookup_object('gcode')
        self.gcode.register_command("MEASURE_AXES_NOISE",
                                    self.cmd_MEASURE_AXES_NOISE,
                                    desc=self.cmd_MEASURE_AXES_NOISE_help)
        self.gcode.register_command("TEST_RESONANCES",
                                    self.cmd_TEST_RESONANCES,
                                    desc=self.cmd_TEST_RESONANCES_help)
        self.gcode.register_command("SHAPER_CALIBRATE",
                                    self.cmd_SHAPER_CALIBRATE,
                                    desc=self.cmd_SHAPER_CALIBRATE_help)
        self.gcode.register_command("COMPREHENSIVE_RESONANCE_TEST",
                                    self.cmd_COMPREHENSIVE_RESONANCE_TEST,
                                    desc=self.cmd_COMPREHENSIVE_RESONANCE_TEST_help)
        self.printer.register_event_handler("klippy:connect", self.connect)

    def connect(self):
        self.accel_chips = [
                (chip_axis, self.printer.lookup_object(chip_name))
                for chip_axis, chip_name in self.accel_chip_names]

    def _run_test(self, gcmd, axes, helper, raw_name_suffix=None,
                  accel_chips=None, test_point=None, use_microphone=False):
        toolhead = self.printer.lookup_object('toolhead')
        calibration_data = {axis: None for axis in axes}
        microphone_data = None

        self.generator.prepare_test(gcmd)

        test_points = [test_point] if test_point else self.probe_points

        for point in test_points:
            toolhead.manual_move(point, self.move_speed)
            if len(test_points) > 1 or test_point is not None:
                gcmd.respond_info(
                        "Probing point (%.3f, %.3f, %.3f)" % tuple(point))
            for axis in axes:
                toolhead.wait_moves()
                toolhead.dwell(0.500)
                if len(axes) > 1:
                    gcmd.respond_info("Testing axis %s" % axis.get_name())

                raw_values = []
                if accel_chips is None:
                    for chip_axis, chip in self.accel_chips:
                        if axis.matches(chip_axis):
                            aclient = chip.start_internal_client()
                            raw_values.append((chip_axis, aclient, chip.name))
                else:
                    for chip in accel_chips:
                        aclient = chip.start_internal_client()
                        raw_values.append((axis, aclient, chip.name))

                # Start microphone recording if enabled
                if use_microphone and self.microphone_enabled and self.microphone_tester:
                    gcmd.respond_info("Starting microphone recording for audio analysis...")
                    try:
                        self.microphone_tester.start_recording()
                    except Exception as e:
                        gcmd.respond_info("Microphone recording failed: %s" % str(e))
                        use_microphone = False

                # Generate moves
                test_seq = self.generator.gen_test()
                self.executor.run_test(test_seq, axis, gcmd)
                
                # Stop microphone recording and get data
                if use_microphone and self.microphone_enabled and self.microphone_tester:
                    try:
                        mic_data = self.microphone_tester.stop_recording()
                        if mic_data:
                            audio_analysis = self.microphone_tester.analyze_audio_resonances(mic_data)
                            if audio_analysis:
                                microphone_data = audio_analysis
                                gcmd.respond_info("Microphone data captured: %.1fs audio, %d peaks detected" % 
                                                (mic_data.duration, len(audio_analysis['peaks'])))
                            else:
                                gcmd.respond_info("Microphone analysis failed")
                        else:
                            gcmd.respond_info("No microphone data captured")
                    except Exception as e:
                        gcmd.respond_info("Microphone analysis error: %s" % str(e))
                
                for chip_axis, aclient, chip_name in raw_values:
                    aclient.finish_measurements()
                    if raw_name_suffix is not None:
                        raw_name = self.get_filename(
                                'raw_data', raw_name_suffix, axis,
                                point if len(test_points) > 1 else None,
                                chip_name if accel_chips is not None else None,)
                        aclient.write_to_file(raw_name)
                        gcmd.respond_info(
                                "Writing raw accelerometer data to "
                                "%s file" % (raw_name,))
                if helper is None:
                    continue
                for chip_axis, aclient, chip_name in raw_values:
                    if not aclient.has_valid_samples():
                        raise gcmd.error(
                            "accelerometer '%s' measured no data" % (
                                chip_name,))
                    new_data = helper.process_accelerometer_data(aclient)
                    
                    # Enhance with microphone data if available
                    if microphone_data and self.microphone_enabled:
                        new_data = self._enhance_with_microphone_data(new_data, microphone_data, gcmd)
                    
                    if calibration_data[axis] is None:
                        calibration_data[axis] = new_data
                    else:
                        calibration_data[axis].add_data(new_data)
        return calibration_data
    def _parse_chips(self, accel_chips):
        parsed_chips = []
        for chip_name in accel_chips.split(','):
            chip = self.printer.lookup_object(chip_name.strip())
            parsed_chips.append(chip)
        return parsed_chips
    def _get_max_calibration_freq(self):
        return 1.5 * self.generator.get_max_freq()
    cmd_TEST_RESONANCES_help = ("Runs the resonance test for a specifed axis")
    def cmd_TEST_RESONANCES(self, gcmd):
        # Parse parameters
        axis = _parse_axis(gcmd, gcmd.get("AXIS").lower())
        chips_str = gcmd.get("CHIPS", None)
        test_point = gcmd.get("POINT", None)
        use_microphone = gcmd.get_int("MICROPHONE", 0) and self.microphone_enabled

        if test_point:
            test_coords = test_point.split(',')
            if len(test_coords) != 3:
                raise gcmd.error("Invalid POINT parameter, must be 'x,y,z'")
            try:
                test_point = [float(p.strip()) for p in test_coords]
            except ValueError:
                raise gcmd.error("Invalid POINT parameter, must be 'x,y,z'"
                " where x, y and z are valid floating point numbers")

        accel_chips = self._parse_chips(chips_str) if chips_str else None

        outputs = gcmd.get("OUTPUT", "resonances").lower().split(',')
        for output in outputs:
            if output not in ['resonances', 'raw_data']:
                raise gcmd.error("Unsupported output '%s', only 'resonances'"
                                 " and 'raw_data' are supported" % (output,))
        if not outputs:
            raise gcmd.error("No output specified, at least one of 'resonances'"
                             " or 'raw_data' must be set in OUTPUT parameter")
        name_suffix = gcmd.get("NAME", time.strftime("%Y%m%d_%H%M%S"))
        if not self.is_valid_name_suffix(name_suffix):
            raise gcmd.error("Invalid NAME parameter")
        csv_output = 'resonances' in outputs
        raw_output = 'raw_data' in outputs

        if use_microphone:
            gcmd.respond_info("Microphone-enhanced resonance testing enabled")

        # Setup calculation of resonances
        if csv_output:
            helper = shaper_calibrate.ShaperCalibrate(self.printer)
        else:
            helper = None

        data = self._run_test(
                gcmd, [axis], helper,
                raw_name_suffix=name_suffix if raw_output else None,
                accel_chips=accel_chips, test_point=test_point, 
                use_microphone=use_microphone)[axis]
        if csv_output:
            csv_name = self.save_calibration_data(
                    'resonances', name_suffix, helper, axis, data,
                    point=test_point, max_freq=self._get_max_calibration_freq())
            gcmd.respond_info(
                    "Resonances data written to %s file" % (csv_name,))
    cmd_SHAPER_CALIBRATE_help = (
        "Similar to TEST_RESONANCES but suggest input shaper config")
    def cmd_SHAPER_CALIBRATE(self, gcmd):
        # Parse parameters
        axis = gcmd.get("AXIS", None)
        if not axis:
            calibrate_axes = [TestAxis('x'), TestAxis('y')]
        elif axis.lower() not in 'xy':
            raise gcmd.error("Unsupported axis '%s'" % (axis,))
        else:
            calibrate_axes = [TestAxis(axis.lower())]
        chips_str = gcmd.get("CHIPS", None)
        accel_chips = self._parse_chips(chips_str) if chips_str else None

        max_smoothing = gcmd.get_float(
                "MAX_SMOOTHING", self.max_smoothing, minval=0.05)
        
        # Enhanced calibration options
        comprehensive = gcmd.get_int("COMPREHENSIVE", 0)
        multi_point = gcmd.get_int("MULTI_POINT", 0)
        use_microphone = gcmd.get_int("MICROPHONE", 0) and self.microphone_enabled

        name_suffix = gcmd.get("NAME", time.strftime("%Y%m%d_%H%M%S"))
        if not self.is_valid_name_suffix(name_suffix):
            raise gcmd.error("Invalid NAME parameter")

        if use_microphone:
            gcmd.respond_info("Microphone-enhanced calibration enabled")

        input_shaper = self.printer.lookup_object('input_shaper', None)

        # Setup shaper calibration
        helper = shaper_calibrate.ShaperCalibrate(self.printer)

        if multi_point:
            gcmd.respond_info("=== Multi-Point Calibration ===")
            calibration_data = self._run_multi_point_calibration(
                gcmd, calibrate_axes, helper, accel_chips, use_microphone)
        else:
            calibration_data = self._run_test(gcmd, calibrate_axes, helper,
                                              accel_chips=accel_chips, 
                                              use_microphone=use_microphone)

        configfile = self.printer.lookup_object('configfile')
        for axis in calibrate_axes:
            axis_name = axis.get_name()
            gcmd.respond_info(
                    "Calculating the best input shaper parameters for %s axis"
                    % (axis_name,))
            calibration_data[axis].normalize_to_frequencies()
            systime = self.printer.get_reactor().monotonic()
            toolhead = self.printer.lookup_object('toolhead')
            toolhead_info = toolhead.get_status(systime)
            scv = toolhead_info['square_corner_velocity']
            max_freq = self._get_max_calibration_freq()
            
            if comprehensive or use_microphone:
                # Use intelligent recommendations (enhanced with microphone data)
                best_shaper, all_shapers, analysis = helper.get_intelligent_recommendations(
                    calibration_data[axis], max_smoothing=max_smoothing,
                    scv=scv, logger=gcmd.respond_info, use_microphone=use_microphone)
                
                # Save detailed analysis
                analysis_name = self.get_filename(
                    'analysis', name_suffix, axis, None)
                self._save_analysis_data(analysis_name, analysis)
                gcmd.respond_info(
                    "Detailed analysis saved to %s" % (analysis_name,))
            else:
                # Standard calibration
                best_shaper, all_shapers = helper.find_best_shaper(
                        calibration_data[axis], max_smoothing=max_smoothing,
                        scv=scv, max_freq=max_freq, logger=gcmd.respond_info)
            
            gcmd.respond_info(
                    "Recommended shaper_type_%s = %s, shaper_freq_%s = %.1f Hz"
                    % (axis_name, best_shaper.name,
                       axis_name, best_shaper.freq))
            if input_shaper is not None:
                helper.apply_params(input_shaper, axis_name,
                                    best_shaper.name, best_shaper.freq)
            helper.save_params(configfile, axis_name,
                               best_shaper.name, best_shaper.freq)
            csv_name = self.save_calibration_data(
                    'calibration_data', name_suffix, helper, axis,
                    calibration_data[axis], all_shapers, max_freq=max_freq)
            gcmd.respond_info(
                    "Shaper calibration data written to %s file" % (csv_name,))
        gcmd.respond_info(
            "The SAVE_CONFIG command will update the printer config file\n"
            "with these parameters and restart the printer.")
            
    def _run_multi_point_calibration(self, gcmd, axes, helper, accel_chips=None, use_microphone=False):
        """Run calibration at multiple points across the bed"""
        toolhead = self.printer.lookup_object('toolhead')
        
        # Use all configured probe points for multi-point calibration
        test_points = self.probe_points
        gcmd.respond_info("Multi-point calibration using %d points" % len(test_points))
        
        combined_data = {axis: None for axis in axes}
        point_data = {}
        
        for i, point in enumerate(test_points):
            gcmd.respond_info("=== Calibrating at point %d/%d: (%.1f, %.1f, %.1f) ===" % 
                             (i+1, len(test_points), point[0], point[1], point[2]))
            
            point_calibration = self._run_test(
                gcmd, axes, helper, accel_chips=accel_chips, test_point=point, 
                use_microphone=use_microphone)
            
            point_data[i] = point_calibration
            
            # Combine data from all points
            for axis in axes:
                if combined_data[axis] is None:
                    combined_data[axis] = point_calibration[axis]
                else:
                    combined_data[axis].add_data(point_calibration[axis])
        
        # Analyze spatial variation
        self._analyze_spatial_variation(gcmd, point_data, test_points)
        
    def _enhance_with_microphone_data(self, accel_data, microphone_data, gcmd):
        """Enhance accelerometer data with microphone analysis"""
        if not microphone_data or not microphone_data.get('peaks'):
            return accel_data
        
        # Extract peak frequencies from both sources
        accel_peaks = accel_data.find_peak_frequencies()
        audio_peaks = [p['frequency'] for p in microphone_data['peaks']]
        
        if len(audio_peaks) == 0:
            return accel_data
        
        # Cross-correlate audio and accelerometer peaks
        from . import microphone_resonance
        analyzer = microphone_resonance.AudioFrequencyAnalyzer(self.printer)
        correlations = analyzer.cross_correlate_with_accelerometer(
            microphone_data['peaks'], accel_peaks)
        
        # Report correlation findings
        gcmd.respond_info("=== Audio-Accelerometer Correlation Analysis ===")
        confirmed_peaks = 0
        new_peaks = 0
        
        for correlation in correlations:
            audio_freq = correlation['audio_peak']['frequency']
            accel_match = correlation['accelerometer_match']
            confidence = correlation['confidence']
            
            if accel_match is not None:
                if confidence > 0.8:
                    gcmd.respond_info("CONFIRMED: %.1f Hz (audio) matches %.1f Hz (accel), confidence %.1f%%" % 
                                    (audio_freq, accel_match, confidence * 100))
                    confirmed_peaks += 1
                else:
                    gcmd.respond_info("WEAK MATCH: %.1f Hz (audio) ~ %.1f Hz (accel), confidence %.1f%%" % 
                                    (audio_freq, accel_match, confidence * 100))
            else:
                gcmd.respond_info("NEW PEAK: %.1f Hz detected in audio only (%.1f dB)" % 
                                (audio_freq, correlation['audio_peak']['amplitude_db']))
                new_peaks += 1
        
        gcmd.respond_info("Summary: %d confirmed peaks, %d audio-only peaks" % 
                         (confirmed_peaks, new_peaks))
        
        # Store microphone analysis in the calibration data for later use
        if hasattr(accel_data, '_microphone_analysis'):
            accel_data._microphone_analysis = microphone_data
        else:
            # Add as new attribute
            accel_data._microphone_analysis = microphone_data
        
        return accel_data
    
    def _analyze_spatial_variation(self, gcmd, point_data, test_points):
        """Analyze how resonance characteristics vary across the bed"""
        gcmd.respond_info("=== Spatial Variation Analysis ===")
        
        # Analyze frequency variation across points
        for axis_idx, axis_name in enumerate(['x', 'y']):
            if axis_idx >= len(list(point_data[0].keys())):
                continue
                
            axis_key = list(point_data[0].keys())[axis_idx]
            frequencies = []
            amplitudes = []
            
            for point_idx, point_cal in point_data.items():
                analysis = point_cal[axis_key].get_comprehensive_analysis()
                if analysis['dominant_frequency']:
                    frequencies.append(analysis['dominant_frequency'])
                    amplitudes.append(analysis['max_amplitude'])
            
            if frequencies:
                import statistics
                freq_mean = statistics.mean(frequencies)
                freq_std = statistics.stdev(frequencies) if len(frequencies) > 1 else 0
                amp_mean = statistics.mean(amplitudes)
                amp_std = statistics.stdev(amplitudes) if len(amplitudes) > 1 else 0
                
                gcmd.respond_info("%s-axis spatial analysis:" % axis_name.upper())
                gcmd.respond_info("  Frequency: %.1f ± %.1f Hz (%.1f%% variation)" % 
                                 (freq_mean, freq_std, (freq_std/freq_mean)*100 if freq_mean > 0 else 0))
                gcmd.respond_info("  Amplitude: %.3f ± %.3f (%.1f%% variation)" % 
                                 (amp_mean, amp_std, (amp_std/amp_mean)*100 if amp_mean > 0 else 0))
                
                if freq_std / freq_mean > 0.1:  # More than 10% variation
                    gcmd.respond_info("  WARNING: High frequency variation detected!")
                    gcmd.respond_info("  Consider checking mechanical integrity across the bed")

    def _save_analysis_data(self, filename, analysis):
        """Save detailed analysis data to a file"""
        try:
            import json
            with open(filename.replace('.csv', '.json'), 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_analysis = {}
                for key, value in analysis.items():
                    if hasattr(value, 'tolist'):
                        serializable_analysis[key] = value.tolist()
                    elif isinstance(value, dict):
                        serializable_analysis[key] = {}
                        for subkey, subvalue in value.items():
                            if hasattr(subvalue, 'tolist'):
                                serializable_analysis[key][subkey] = subvalue.tolist()
                            else:
                                serializable_analysis[key][subkey] = subvalue
                    else:
                        serializable_analysis[key] = value
                
                json.dump(serializable_analysis, f, indent=2)
        except Exception as e:
            # Fallback to simple text format
            with open(filename.replace('.csv', '.txt'), 'w') as f:
                f.write("Comprehensive Resonance Analysis\n")
                f.write("================================\n\n")
                f.write("Peak Frequencies: %s Hz\n" % 
                       ', '.join(['%.1f' % f for f in analysis['peak_frequencies']]))
                if analysis['dominant_frequency']:
                    f.write("Dominant Frequency: %.1f Hz\n" % analysis['dominant_frequency'])
                f.write("Cross-coupling Strength: %.2f\n" % 
                       analysis['cross_coupling']['strength'])
                f.write("Frequency Centroid: %.1f Hz\n" % analysis['frequency_centroid'])
                f.write("\nQuality Metrics:\n")
                for key, value in analysis['quality_metrics'].items():
                    f.write("  %s: %.3f\n" % (key, value))

    cmd_COMPREHENSIVE_RESONANCE_TEST_help = (
        "Run comprehensive resonance analysis with all advanced features")
    def cmd_COMPREHENSIVE_RESONANCE_TEST(self, gcmd):
        """Comprehensive resonance testing with all advanced features"""
        # Parse parameters
        axis = gcmd.get("AXIS", None)
        if not axis:
            test_axes = [TestAxis('x'), TestAxis('y')]
        elif axis.lower() not in 'xy':
            raise gcmd.error("Unsupported axis '%s'" % (axis,))
        else:
            test_axes = [TestAxis(axis.lower())]
        
        chips_str = gcmd.get("CHIPS", None)
        accel_chips = self._parse_chips(chips_str) if chips_str else None
        use_microphone = gcmd.get_int("MICROPHONE", 0) and self.microphone_enabled
        
        name_suffix = gcmd.get("NAME", time.strftime("%Y%m%d_%H%M%S"))
        if not self.is_valid_name_suffix(name_suffix):
            raise gcmd.error("Invalid NAME parameter")
        
        gcmd.respond_info("=== COMPREHENSIVE RESONANCE ANALYSIS ===")
        gcmd.respond_info("This test will perform:")
        gcmd.respond_info("1. Multi-point bed mapping")
        gcmd.respond_info("2. Advanced frequency analysis")
        gcmd.respond_info("3. Cross-axis coupling analysis")
        gcmd.respond_info("4. Harmonic content analysis")
        gcmd.respond_info("5. Intelligent shaper recommendations")
        if use_microphone:
            gcmd.respond_info("6. Microphone-based audio analysis")
            gcmd.respond_info("7. Audio-accelerometer cross-correlation")
        
        # Setup comprehensive analysis
        helper = shaper_calibrate.ShaperCalibrate(self.printer)
        
        # Run multi-point calibration
        calibration_data = self._run_multi_point_calibration(
            gcmd, test_axes, helper, accel_chips, use_microphone)
        
        # Perform comprehensive analysis for each axis
        for axis in test_axes:
            axis_name = axis.get_name()
            gcmd.respond_info("\n=== COMPREHENSIVE ANALYSIS FOR %s AXIS ===" % axis_name.upper())
            
            calibration_data[axis].normalize_to_frequencies()
            
            # Get intelligent recommendations
            systime = self.printer.get_reactor().monotonic()
            toolhead = self.printer.lookup_object('toolhead')
            toolhead_info = toolhead.get_status(systime)
            scv = toolhead_info['square_corner_velocity']
            
            best_shaper, all_shapers, analysis = helper.get_intelligent_recommendations(
                calibration_data[axis], scv=scv, logger=gcmd.respond_info, 
                use_microphone=use_microphone)
            
            # Save comprehensive results
            analysis_name = self.get_filename(
                'comprehensive_analysis', name_suffix, axis, None)
            self._save_analysis_data(analysis_name, analysis)
            
            csv_name = self.save_calibration_data(
                'comprehensive_data', name_suffix, helper, axis,
                calibration_data[axis], all_shapers, 
                max_freq=self._get_max_calibration_freq())
            
            gcmd.respond_info("\n=== RESULTS FOR %s AXIS ===" % axis_name.upper())
            gcmd.respond_info("Best recommended shaper: %s @ %.1f Hz" % 
                             (best_shaper.name, best_shaper.freq))
            gcmd.respond_info("Vibration reduction: %.1f%%" % 
                             ((1 - best_shaper.vibrs) * 100))
            gcmd.respond_info("Smoothing factor: %.3f" % best_shaper.smoothing)
            gcmd.respond_info("Max recommended acceleration: %.0f mm/s²" % 
                             best_shaper.max_accel)
            gcmd.respond_info("Data saved to: %s" % csv_name)
            gcmd.respond_info("Analysis saved to: %s" % analysis_name)
            
            # Performance comparison
            gcmd.respond_info("\n=== SHAPER PERFORMANCE COMPARISON ===")
            sorted_shapers = sorted(all_shapers, key=lambda s: s.score)
            for i, shaper in enumerate(sorted_shapers[:5]):  # Top 5
                gcmd.respond_info("%d. %s: %.1f%% vibrations, %.3f smoothing, %.0f mm/s² max_accel" % 
                                 (i+1, shaper.name, shaper.vibrs * 100, 
                                  shaper.smoothing, shaper.max_accel))
        
        gcmd.respond_info("\n=== COMPREHENSIVE TEST COMPLETE ===")
        gcmd.respond_info("Use the recommended settings or review the detailed")
        gcmd.respond_info("analysis files for advanced tuning options.")
    cmd_MEASURE_AXES_NOISE_help = (
        "Measures noise of all enabled accelerometer chips")
    def cmd_MEASURE_AXES_NOISE(self, gcmd):
        meas_time = gcmd.get_float("MEAS_TIME", 2.)
        raw_values = [(chip_axis, chip.start_internal_client())
                      for chip_axis, chip in self.accel_chips]
        self.printer.lookup_object('toolhead').dwell(meas_time)
        for chip_axis, aclient in raw_values:
            aclient.finish_measurements()
        helper = shaper_calibrate.ShaperCalibrate(self.printer)
        for chip_axis, aclient in raw_values:
            if not aclient.has_valid_samples():
                raise gcmd.error(
                        "%s-axis accelerometer measured no data" % (
                            chip_axis,))
            data = helper.process_accelerometer_data(aclient)
            vx = data.psd_x.mean()
            vy = data.psd_y.mean()
            vz = data.psd_z.mean()
            gcmd.respond_info("Axes noise for %s-axis accelerometer: "
                              "%.6f (x), %.6f (y), %.6f (z)" % (
                                  chip_axis, vx, vy, vz))

    def is_valid_name_suffix(self, name_suffix):
        return name_suffix.replace('-', '').replace('_', '').isalnum()

    def get_filename(self, base, name_suffix, axis=None,
                     point=None, chip_name=None):
        name = base
        if axis:
            name += '_' + axis.get_name()
        if chip_name:
            name += '_' + chip_name.replace(" ", "_")
        if point:
            name += "_%.3f_%.3f_%.3f" % (point[0], point[1], point[2])
        name += '_' + name_suffix
        return os.path.join("/tmp", name + ".csv")

    def save_calibration_data(self, base_name, name_suffix, shaper_calibrate,
                              axis, calibration_data,
                              all_shapers=None, point=None, max_freq=None):
        output = self.get_filename(base_name, name_suffix, axis, point)
        shaper_calibrate.save_calibration_data(output, calibration_data,
                                               all_shapers, max_freq)
        return output

def load_config(config):
    return ResonanceTester(config)
