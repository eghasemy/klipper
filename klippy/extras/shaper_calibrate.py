# Automatic calibration of input shapers
#
# Copyright (C) 2020-2024  Dmitry Butyugin <dmbutyugin@google.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import collections, importlib, logging, math, multiprocessing, traceback
shaper_defs = importlib.import_module('.shaper_defs', 'extras')

MIN_FREQ = 5.
MAX_FREQ = 200.
WINDOW_T_SEC = 0.5
MAX_SHAPER_FREQ = 150.

TEST_DAMPING_RATIOS=[0.075, 0.1, 0.15]

AUTOTUNE_SHAPERS = ['zv', 'mzv', 'ei', '2hump_ei', '3hump_ei']

# Extended autotune list for comprehensive analysis
COMPREHENSIVE_AUTOTUNE_SHAPERS = ['zv', 'mzv', 'zvd', 'ei', '2hump_ei', '3hump_ei', 
                                  'smooth', 'adaptive_ei', 'multi_freq', 'ulv']

######################################################################
# Frequency response calculation and shaper auto-tuning
######################################################################

class CalibrationData:
    def __init__(self, freq_bins, psd_sum, psd_x, psd_y, psd_z):
        self.freq_bins = freq_bins
        self.psd_sum = psd_sum
        self.psd_x = psd_x
        self.psd_y = psd_y
        self.psd_z = psd_z
        self._psd_list = [self.psd_sum, self.psd_x, self.psd_y, self.psd_z]
        self._psd_map = {'x': self.psd_x, 'y': self.psd_y, 'z': self.psd_z,
                         'all': self.psd_sum}
        self.data_sets = 1
        # Enhanced analysis data
        self._peak_frequencies = None
        self._cross_coupling = None
        self._harmonic_analysis = None
    def add_data(self, other):
        np = self.numpy
        joined_data_sets = self.data_sets + other.data_sets
        for psd, other_psd in zip(self._psd_list, other._psd_list):
            # `other` data may be defined at different frequency bins,
            # interpolating to fix that.
            other_normalized = other.data_sets * np.interp(
                    self.freq_bins, other.freq_bins, other_psd)
            psd *= self.data_sets
            psd[:] = (psd + other_normalized) * (1. / joined_data_sets)
        self.data_sets = joined_data_sets
        # Reset analysis cache when data changes
        self._peak_frequencies = None
        self._cross_coupling = None
        self._harmonic_analysis = None
    def set_numpy(self, numpy):
        self.numpy = numpy
    def normalize_to_frequencies(self):
        for psd in self._psd_list:
            # Avoid division by zero errors
            psd /= self.freq_bins + .1
            # Remove low-frequency noise
            low_freqs = self.freq_bins < 2. * MIN_FREQ
            psd[low_freqs] *= self.numpy.exp(
                    -(2. * MIN_FREQ / (self.freq_bins[low_freqs] + .1))**2 + 1.)
    def get_psd(self, axis='all'):
        return self._psd_map[axis]
    
    def find_peak_frequencies(self, min_prominence=None):
        """Find dominant resonance frequencies using peak detection"""
        if self._peak_frequencies is not None:
            return self._peak_frequencies
            
        np = self.numpy
        try:
            from scipy.signal import find_peaks
        except ImportError:
            # Fallback simple peak detection
            psd = self.psd_sum
            peaks = []
            for i in range(1, len(psd)-1):
                if psd[i] > psd[i-1] and psd[i] > psd[i+1]:
                    peaks.append(i)
            peak_freqs = self.freq_bins[peaks] if peaks else []
        else:
            # Use scipy for more sophisticated peak detection
            psd = self.psd_sum
            if min_prominence is None:
                min_prominence = np.max(psd) * 0.1
            peaks, properties = find_peaks(psd, prominence=min_prominence)
            peak_freqs = self.freq_bins[peaks]
            
        self._peak_frequencies = peak_freqs
        return peak_freqs
    
    def analyze_cross_coupling(self):
        """Analyze cross-axis coupling between X and Y axes"""
        if self._cross_coupling is not None:
            return self._cross_coupling
            
        np = self.numpy
        # Calculate coherence between X and Y axes
        try:
            from scipy.signal import coherence
            # Simple correlation analysis as fallback
            correlation = np.corrcoef(self.psd_x, self.psd_y)[0, 1]
            coupling_strength = abs(correlation)
        except ImportError:
            # Simple correlation analysis
            correlation = np.corrcoef(self.psd_x, self.psd_y)[0, 1]
            coupling_strength = abs(correlation)
            
        # Find frequencies with strong coupling
        ratio = np.minimum(self.psd_x, self.psd_y) / np.maximum(self.psd_x, self.psd_y)
        high_coupling_mask = ratio > 0.5
        coupled_freqs = self.freq_bins[high_coupling_mask]
        
        self._cross_coupling = {
            'strength': coupling_strength,
            'correlation': correlation,
            'coupled_frequencies': coupled_freqs
        }
        return self._cross_coupling
    
    def analyze_harmonics(self, fundamental_freqs=None):
        """Analyze harmonic content in the frequency response"""
        if self._harmonic_analysis is not None:
            return self._harmonic_analysis
            
        if fundamental_freqs is None:
            fundamental_freqs = self.find_peak_frequencies()
            
        np = self.numpy
        harmonics = {}
        
        for fundamental in fundamental_freqs:
            if fundamental < MIN_FREQ:
                continue
                
            harmonic_data = []
            for harmonic_order in [2, 3, 4, 5]:
                harmonic_freq = fundamental * harmonic_order
                if harmonic_freq > np.max(self.freq_bins):
                    break
                    
                # Find closest frequency bin
                freq_idx = np.argmin(np.abs(self.freq_bins - harmonic_freq))
                actual_freq = self.freq_bins[freq_idx]
                
                if abs(actual_freq - harmonic_freq) < 2.0:  # Within 2 Hz
                    harmonic_amplitude = self.psd_sum[freq_idx]
                    harmonic_data.append({
                        'order': harmonic_order,
                        'frequency': actual_freq,
                        'amplitude': harmonic_amplitude
                    })
            
            if harmonic_data:
                harmonics[fundamental] = harmonic_data
        
        self._harmonic_analysis = harmonics
        return harmonics
    
    def get_comprehensive_analysis(self):
        """Get a comprehensive analysis of the resonance data"""
        peaks = self.find_peak_frequencies()
        coupling = self.analyze_cross_coupling()
        harmonics = self.analyze_harmonics()
        
        np = self.numpy
        # Statistical measures
        stats = {
            'peak_frequencies': peaks.tolist() if hasattr(peaks, 'tolist') else list(peaks),
            'dominant_frequency': peaks[np.argmax(self.psd_sum[np.searchsorted(self.freq_bins, peaks)])] if len(peaks) > 0 else None,
            'frequency_spread': np.std(peaks) if len(peaks) > 1 else 0,
            'max_amplitude': np.max(self.psd_sum),
            'frequency_centroid': np.sum(self.freq_bins * self.psd_sum) / np.sum(self.psd_sum),
            'cross_coupling': coupling,
            'harmonics': harmonics,
            'quality_metrics': {
                'noise_floor': np.percentile(self.psd_sum, 10),
                'dynamic_range': np.max(self.psd_sum) / np.percentile(self.psd_sum, 10),
                'frequency_resolution': self.freq_bins[1] - self.freq_bins[0]
            }
        }
        
        return stats


CalibrationResult = collections.namedtuple(
        'CalibrationResult',
        ('name', 'freq', 'vals', 'vibrs', 'smoothing', 'score', 'max_accel'))

class ShaperCalibrate:
    def __init__(self, printer):
        self.printer = printer
        self.error = printer.command_error if printer else Exception
        try:
            self.numpy = importlib.import_module('numpy')
        except ImportError:
            raise self.error(
                    "Failed to import `numpy` module, make sure it was "
                    "installed via `~/klippy-env/bin/pip install` (refer to "
                    "docs/Measuring_Resonances.md for more details).")

    def background_process_exec(self, method, args):
        if self.printer is None:
            return method(*args)
        import queuelogger
        parent_conn, child_conn = multiprocessing.Pipe()
        def wrapper():
            queuelogger.clear_bg_logging()
            try:
                res = method(*args)
            except:
                child_conn.send((True, traceback.format_exc()))
                child_conn.close()
                return
            child_conn.send((False, res))
            child_conn.close()
        # Start a process to perform the calculation
        calc_proc = multiprocessing.Process(target=wrapper)
        calc_proc.daemon = True
        calc_proc.start()
        # Wait for the process to finish
        reactor = self.printer.get_reactor()
        gcode = self.printer.lookup_object("gcode")
        eventtime = last_report_time = reactor.monotonic()
        while calc_proc.is_alive():
            if eventtime > last_report_time + 5.:
                last_report_time = eventtime
                gcode.respond_info("Wait for calculations..", log=False)
            eventtime = reactor.pause(eventtime + .1)
        # Return results
        is_err, res = parent_conn.recv()
        if is_err:
            raise self.error("Error in remote calculation: %s" % (res,))
        calc_proc.join()
        parent_conn.close()
        return res

    def _split_into_windows(self, x, window_size, overlap):
        # Memory-efficient algorithm to split an input 'x' into a series
        # of overlapping windows
        step_between_windows = window_size - overlap
        n_windows = (x.shape[-1] - overlap) // step_between_windows
        shape = (window_size, n_windows)
        strides = (x.strides[-1], step_between_windows * x.strides[-1])
        return self.numpy.lib.stride_tricks.as_strided(
                x, shape=shape, strides=strides, writeable=False)

    def _psd(self, x, fs, nfft):
        # Calculate power spectral density (PSD) using Welch's algorithm
        np = self.numpy
        window = np.kaiser(nfft, 6.)
        # Compensation for windowing loss
        scale = 1.0 / (window**2).sum()

        # Split into overlapping windows of size nfft
        overlap = nfft // 2
        x = self._split_into_windows(x, nfft, overlap)

        # First detrend, then apply windowing function
        x = window[:, None] * (x - np.mean(x, axis=0))

        # Calculate frequency response for each window using FFT
        result = np.fft.rfft(x, n=nfft, axis=0)
        result = np.conjugate(result) * result
        result *= scale / fs
        # For one-sided FFT output the response must be doubled, except
        # the last point for unpaired Nyquist frequency (assuming even nfft)
        # and the 'DC' term (0 Hz)
        result[1:-1,:] *= 2.

        # Welch's algorithm: average response over windows
        psd = result.real.mean(axis=-1)

        # Calculate the frequency bins
        freqs = np.fft.rfftfreq(nfft, 1. / fs)
        return freqs, psd

    def calc_freq_response(self, raw_values):
        np = self.numpy
        if raw_values is None:
            return None
        if isinstance(raw_values, np.ndarray):
            data = raw_values
        else:
            samples = raw_values.get_samples()
            if not samples:
                return None
            data = np.array(samples)

        N = data.shape[0]
        T = data[-1,0] - data[0,0]
        SAMPLING_FREQ = N / T
        # Round up to the nearest power of 2 for faster FFT
        M = 1 << int(SAMPLING_FREQ * WINDOW_T_SEC - 1).bit_length()
        if N <= M:
            return None

        # Calculate PSD (power spectral density) of vibrations per
        # frequency bins (the same bins for X, Y, and Z)
        fx, px = self._psd(data[:,1], SAMPLING_FREQ, M)
        fy, py = self._psd(data[:,2], SAMPLING_FREQ, M)
        fz, pz = self._psd(data[:,3], SAMPLING_FREQ, M)
        return CalibrationData(fx, px+py+pz, px, py, pz)

    def process_accelerometer_data(self, data):
        calibration_data = self.background_process_exec(
                self.calc_freq_response, (data,))
        if calibration_data is None:
            raise self.error(
                    "Internal error processing accelerometer data %s" % (data,))
        calibration_data.set_numpy(self.numpy)
        return calibration_data

    def _estimate_shaper(self, shaper, test_damping_ratio, test_freqs):
        np = self.numpy

        A, T = np.array(shaper[0]), np.array(shaper[1])
        inv_D = 1. / A.sum()

        omega = 2. * math.pi * test_freqs
        damping = test_damping_ratio * omega
        omega_d = omega * math.sqrt(1. - test_damping_ratio**2)
        W = A * np.exp(np.outer(-damping, (T[-1] - T)))
        S = W * np.sin(np.outer(omega_d, T))
        C = W * np.cos(np.outer(omega_d, T))
        return np.sqrt(S.sum(axis=1)**2 + C.sum(axis=1)**2) * inv_D

    def _estimate_remaining_vibrations(self, shaper, test_damping_ratio,
                                       freq_bins, psd):
        vals = self._estimate_shaper(shaper, test_damping_ratio, freq_bins)
        # The input shaper can only reduce the amplitude of vibrations by
        # SHAPER_VIBRATION_REDUCTION times, so all vibrations below that
        # threshold can be igonred
        vibr_threshold = psd.max() / shaper_defs.SHAPER_VIBRATION_REDUCTION
        remaining_vibrations = self.numpy.maximum(
                vals * psd - vibr_threshold, 0).sum()
        all_vibrations = self.numpy.maximum(psd - vibr_threshold, 0).sum()
        return (remaining_vibrations / all_vibrations, vals)

    def _get_shaper_smoothing(self, shaper, accel=5000, scv=5.):
        half_accel = accel * .5

        A, T = shaper
        inv_D = 1. / sum(A)
        n = len(T)
        # Calculate input shaper shift
        ts = sum([A[i] * T[i] for i in range(n)]) * inv_D

        # Calculate offset for 90 and 180 degrees turn
        offset_90 = offset_180 = 0.
        for i in range(n):
            if T[i] >= ts:
                # Calculate offset for one of the axes
                offset_90 += A[i] * (scv + half_accel * (T[i]-ts)) * (T[i]-ts)
            offset_180 += A[i] * half_accel * (T[i]-ts)**2
        offset_90 *= inv_D * math.sqrt(2.)
        offset_180 *= inv_D
        return max(offset_90, offset_180)

    def fit_shaper(self, shaper_cfg, calibration_data, shaper_freqs,
                   damping_ratio, scv, max_smoothing, test_damping_ratios,
                   max_freq):
        np = self.numpy

        damping_ratio = damping_ratio or shaper_defs.DEFAULT_DAMPING_RATIO
        test_damping_ratios = test_damping_ratios or TEST_DAMPING_RATIOS

        if not shaper_freqs:
            shaper_freqs = (None, None, None)
        if isinstance(shaper_freqs, tuple):
            freq_end = shaper_freqs[1] or MAX_SHAPER_FREQ
            freq_start = min(shaper_freqs[0] or shaper_cfg.min_freq,
                             freq_end - 1e-7)
            freq_step = shaper_freqs[2] or .2
            test_freqs = np.arange(freq_start, freq_end, freq_step)
        else:
            test_freqs = np.array(shaper_freqs)

        max_freq = max(max_freq or MAX_FREQ, test_freqs.max())

        freq_bins = calibration_data.freq_bins
        psd = calibration_data.psd_sum[freq_bins <= max_freq]
        freq_bins = freq_bins[freq_bins <= max_freq]

        best_res = None
        results = []
        for test_freq in test_freqs[::-1]:
            shaper_vibrations = 0.
            shaper_vals = np.zeros(shape=freq_bins.shape)
            shaper = shaper_cfg.init_func(test_freq, damping_ratio)
            shaper_smoothing = self._get_shaper_smoothing(shaper, scv=scv)
            if max_smoothing and shaper_smoothing > max_smoothing and best_res:
                return best_res
            # Exact damping ratio of the printer is unknown, pessimizing
            # remaining vibrations over possible damping values
            for dr in test_damping_ratios:
                vibrations, vals = self._estimate_remaining_vibrations(
                        shaper, dr, freq_bins, psd)
                shaper_vals = np.maximum(shaper_vals, vals)
                if vibrations > shaper_vibrations:
                    shaper_vibrations = vibrations
            max_accel = self.find_shaper_max_accel(shaper, scv)
            # The score trying to minimize vibrations, but also accounting
            # the growth of smoothing. The formula itself does not have any
            # special meaning, it simply shows good results on real user data
            shaper_score = shaper_smoothing * (shaper_vibrations**1.5 +
                                               shaper_vibrations * .2 + .01)
            results.append(
                    CalibrationResult(
                        name=shaper_cfg.name, freq=test_freq, vals=shaper_vals,
                        vibrs=shaper_vibrations, smoothing=shaper_smoothing,
                        score=shaper_score, max_accel=max_accel))
            if best_res is None or best_res.vibrs > results[-1].vibrs:
                # The current frequency is better for the shaper.
                best_res = results[-1]
        # Try to find an 'optimal' shapper configuration: the one that is not
        # much worse than the 'best' one, but gives much less smoothing
        selected = best_res
        for res in results[::-1]:
            if res.vibrs < best_res.vibrs * 1.1 and res.score < selected.score:
                selected = res
        return selected

    def _bisect(self, func):
        left = right = 1.
        if not func(1e-9):
            return 0.
        while not func(left):
            right = left
            left *= .5
        if right == left:
            while func(right):
                right *= 2.
        while right - left > 1e-8:
            middle = (left + right) * .5
            if func(middle):
                left = middle
            else:
                right = middle
        return left

    def find_shaper_max_accel(self, shaper, scv):
        # Just some empirically chosen value which produces good projections
        # for max_accel without much smoothing
        TARGET_SMOOTHING = 0.12
        max_accel = self._bisect(lambda test_accel: self._get_shaper_smoothing(
            shaper, test_accel, scv) <= TARGET_SMOOTHING)
        return max_accel

    def find_best_shaper(self, calibration_data, shapers=None,
                         damping_ratio=None, scv=None, shaper_freqs=None,
                         max_smoothing=None, test_damping_ratios=None,
                         max_freq=None, logger=None, comprehensive=False):
        best_shaper = None
        all_shapers = []
        
        # Use comprehensive shaper list if requested
        if comprehensive:
            shapers = shapers or COMPREHENSIVE_AUTOTUNE_SHAPERS
        else:
            shapers = shapers or AUTOTUNE_SHAPERS
            
        for shaper_cfg in shaper_defs.INPUT_SHAPERS:
            if shaper_cfg.name not in shapers:
                continue
            shaper = self.background_process_exec(self.fit_shaper, (
                shaper_cfg, calibration_data, shaper_freqs, damping_ratio,
                scv, max_smoothing, test_damping_ratios, max_freq))
            if logger is not None:
                logger("Fitted shaper '%s' frequency = %.1f Hz "
                       "(vibrations = %.1f%%, smoothing ~= %.3f)" % (
                           shaper.name, shaper.freq, shaper.vibrs * 100.,
                           shaper.smoothing))
                logger("To avoid too much smoothing with '%s', suggested "
                       "max_accel <= %.0f mm/sec^2" % (
                           shaper.name, round(shaper.max_accel / 100.) * 100.))
            all_shapers.append(shaper)
            if (best_shaper is None or shaper.score * 1.2 < best_shaper.score or
                    (shaper.score * 1.05 < best_shaper.score and
                        shaper.smoothing * 1.1 < best_shaper.smoothing)):
                # Either the shaper significantly improves the score (by 20%),
                # or it improves the score and smoothing (by 5% and 10% resp.)
                best_shaper = shaper
        return best_shaper, all_shapers

    def get_intelligent_recommendations(self, calibration_data, max_smoothing=None, 
                                       scv=None, logger=None, use_microphone=False):
        """Get intelligent shaper recommendations based on comprehensive analysis"""
        analysis = calibration_data.get_comprehensive_analysis()
        
        # Enhance analysis with microphone data if available
        if use_microphone and hasattr(calibration_data, '_microphone_analysis'):
            microphone_data = calibration_data._microphone_analysis
            if microphone_data and microphone_data.get('peaks'):
                analysis['microphone_peaks'] = [p['frequency'] for p in microphone_data['peaks']]
                analysis['microphone_peak_count'] = len(microphone_data['peaks'])
                
                # Cross-validate accelerometer peaks with microphone peaks
                audio_confirmed_peaks = []
                for audio_peak in microphone_data['peaks']:
                    for accel_peak in analysis['peak_frequencies']:
                        if abs(audio_peak['frequency'] - accel_peak) <= 2.0:  # Within 2 Hz
                            audio_confirmed_peaks.append(accel_peak)
                            break
                analysis['audio_confirmed_peaks'] = audio_confirmed_peaks
                analysis['audio_confirmation_ratio'] = (
                    len(audio_confirmed_peaks) / len(analysis['peak_frequencies']) 
                    if len(analysis['peak_frequencies']) > 0 else 0)
        
        if logger:
            logger("=== Comprehensive Resonance Analysis ===")
            logger("Peak frequencies detected: %s Hz" % 
                   ', '.join(['%.1f' % f for f in analysis['peak_frequencies']]))
            if analysis['dominant_frequency']:
                logger("Dominant frequency: %.1f Hz" % analysis['dominant_frequency'])
            logger("Cross-axis coupling strength: %.2f" % 
                   analysis['cross_coupling']['strength'])
            logger("Frequency centroid: %.1f Hz" % analysis['frequency_centroid'])
            
            # Microphone analysis
            if use_microphone and 'microphone_peaks' in analysis:
                logger("=== Microphone Analysis ===")
                logger("Audio peaks detected: %s Hz" % 
                       ', '.join(['%.1f' % f for f in analysis['microphone_peaks']]))
                logger("Audio-accelerometer confirmation: %.1f%% of peaks confirmed" % 
                       (analysis['audio_confirmation_ratio'] * 100))
                if analysis['audio_confirmed_peaks']:
                    logger("Confirmed peaks: %s Hz" % 
                           ', '.join(['%.1f' % f for f in analysis['audio_confirmed_peaks']]))
            
            # Harmonic analysis
            if analysis['harmonics']:
                logger("Detected harmonics:")
                for fundamental, harmonics in analysis['harmonics'].items():
                    harmonic_info = ', '.join(['%dx (%.1fHz)' % (h['order'], h['frequency']) 
                                             for h in harmonics])
                    logger("  %.1f Hz: %s" % (fundamental, harmonic_info))

        # Adaptive recommendations based on analysis
        recommendations = []
        
        # Check for complex resonance patterns
        num_peaks = len(analysis['peak_frequencies'])
        coupling_strength = analysis['cross_coupling']['strength']
        has_harmonics = len(analysis['harmonics']) > 0
        
        # Enhanced decision making with microphone data
        audio_confidence = analysis.get('audio_confirmation_ratio', 0) if use_microphone else 0
        if use_microphone and audio_confidence > 0.7:
            # High audio confirmation - trust the accelerometer data more
            confidence_boost = True
            if logger:
                logger("High audio-accelerometer correlation detected - boosting confidence in measurements")
        else:
            confidence_boost = False
        
        if num_peaks <= 1 and coupling_strength < 0.3 and not has_harmonics:
            # Simple resonance pattern - recommend efficient shapers
            if confidence_boost:
                recommendations.extend(['smooth', 'zv', 'mzv'])  # Prioritize advanced smooth shaper
            else:
                recommendations.extend(['zv', 'mzv', 'smooth'])
            if logger:
                logger("Simple resonance pattern detected - recommending efficient shapers")
        elif num_peaks > 2 or has_harmonics:
            # Complex resonance pattern - recommend advanced shapers
            if confidence_boost:
                recommendations.extend(['ulv', 'multi_freq', '3hump_ei'])  # Prioritize most advanced
            else:
                recommendations.extend(['multi_freq', 'ulv', '3hump_ei'])
            if logger:
                logger("Complex resonance pattern detected - recommending advanced shapers")
        elif coupling_strength > 0.6:
            # Strong cross-coupling - recommend robust shapers
            recommendations.extend(['ei', '2hump_ei', 'adaptive_ei'])
            if logger:
                logger("Strong cross-axis coupling detected - recommending robust shapers")
        else:
            # Moderate complexity - balanced approach
            if confidence_boost:
                recommendations.extend(['adaptive_ei', 'smooth', 'mzv', 'ei'])
            else:
                recommendations.extend(['mzv', 'ei', 'adaptive_ei', 'smooth'])
            if logger:
                logger("Moderate complexity detected - using balanced approach")

        # Get detailed analysis for each recommended shaper
        best_shaper, all_shapers = self.find_best_shaper(
            calibration_data, shapers=recommendations, max_smoothing=max_smoothing,
            scv=scv, logger=logger, comprehensive=True)
        
        return best_shaper, all_shapers, analysis

    def save_params(self, configfile, axis, shaper_name, shaper_freq):
        if axis == 'xy':
            self.save_params(configfile, 'x', shaper_name, shaper_freq)
            self.save_params(configfile, 'y', shaper_name, shaper_freq)
        else:
            configfile.set('input_shaper', 'shaper_type_'+axis, shaper_name)
            configfile.set('input_shaper', 'shaper_freq_'+axis,
                           '%.1f' % (shaper_freq,))

    def apply_params(self, input_shaper, axis, shaper_name, shaper_freq):
        if axis == 'xy':
            self.apply_params(input_shaper, 'x', shaper_name, shaper_freq)
            self.apply_params(input_shaper, 'y', shaper_name, shaper_freq)
            return
        gcode = self.printer.lookup_object("gcode")
        axis = axis.upper()
        input_shaper.cmd_SET_INPUT_SHAPER(gcode.create_gcode_command(
                "SET_INPUT_SHAPER", "SET_INPUT_SHAPER", {
                    "SHAPER_TYPE_" + axis: shaper_name,
                    "SHAPER_FREQ_" + axis: shaper_freq}))

    def save_calibration_data(self, output, calibration_data, shapers=None,
                              max_freq=None):
        try:
            max_freq = max_freq or MAX_FREQ
            with open(output, "w") as csvfile:
                csvfile.write("freq,psd_x,psd_y,psd_z,psd_xyz")
                if shapers:
                    for shaper in shapers:
                        csvfile.write(",%s(%.1f)" % (shaper.name, shaper.freq))
                csvfile.write("\n")
                num_freqs = calibration_data.freq_bins.shape[0]
                for i in range(num_freqs):
                    if calibration_data.freq_bins[i] >= max_freq:
                        break
                    csvfile.write("%.1f,%.3e,%.3e,%.3e,%.3e" % (
                        calibration_data.freq_bins[i],
                        calibration_data.psd_x[i],
                        calibration_data.psd_y[i],
                        calibration_data.psd_z[i],
                        calibration_data.psd_sum[i]))
                    if shapers:
                        for shaper in shapers:
                            csvfile.write(",%.3f" % (shaper.vals[i],))
                    csvfile.write("\n")
        except IOError as e:
            raise self.error("Error writing to file '%s': %s", output, str(e))
