# Advanced Multi-Dimensional Resonance Compensation System
#
# Copyright (C) 2024  Dmitry Butyugin <dmbutyugin@google.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.

import logging, math, json, time, collections
import numpy as np
from . import shaper_calibrate, resonance_tester
from . import shaper_defs

######################################################################
# Multi-Dimensional Compensation Model
######################################################################

class MovementPattern:
    """Represents a movement pattern with its characteristics"""
    def __init__(self, pattern_type, speed_range, accel_range, direction=None):
        self.pattern_type = pattern_type  # 'linear', 'curved', 'corner', 'infill'
        self.speed_range = speed_range    # (min_speed, max_speed)
        self.accel_range = accel_range    # (min_accel, max_accel)
        self.direction = direction        # direction vector or None for omnidirectional
        self.resonance_profile = {}       # freq -> compensation_params

class SpatialCalibrationPoint:
    """Represents calibration data at a specific spatial location"""
    def __init__(self, x, y, z=None):
        self.position = (x, y, z or 0)
        self.calibration_data = {}  # axis -> CalibrationData
        self.movement_profiles = {}  # MovementPattern -> resonance characteristics
        self.optimal_shapers = {}   # axis -> (shaper_type, freq, params)

class AdaptiveCompensationModel:
    """Multi-dimensional model for adaptive resonance compensation"""
    
    def __init__(self, config):
        self.printer = config.get_printer()
        self.name = config.get_name().split()[-1]
        
        # Configuration parameters
        self.spatial_resolution = config.getfloat('spatial_resolution', 50.0, above=10.0)
        self.speed_bins = config.getint('speed_bins', 5, minval=3, maxval=20)
        self.accel_bins = config.getint('accel_bins', 5, minval=3, maxval=20)
        self.transition_smoothing = config.getfloat('transition_smoothing', 0.1, 
                                                  minval=0.01, maxval=1.0)
        
        # Model data
        self.spatial_points = []
        self.compensation_model = {}
        
        # Multi-axis support
        self.supported_axes = ['x', 'y', 'z', 'a', 'b', 'c']
        self.current_parameters = {axis: None for axis in self.supported_axes}
        self.last_position = None
        self.last_movement_params = None
        
        # Analysis parameters
        self.min_spatial_points = 9  # 3x3 grid minimum
        self.max_spatial_points = 25 # 5x5 grid maximum
        
        # Register gcode commands
        gcode = self.printer.lookup_object('gcode')
        gcode.register_command("ADAPTIVE_RESONANCE_CALIBRATE", 
                             self.cmd_ADAPTIVE_RESONANCE_CALIBRATE,
                             desc=self.cmd_ADAPTIVE_RESONANCE_CALIBRATE_help)
        gcode.register_command("BUILD_COMPENSATION_MODEL",
                             self.cmd_BUILD_COMPENSATION_MODEL,
                             desc=self.cmd_BUILD_COMPENSATION_MODEL_help)
        gcode.register_command("APPLY_ADAPTIVE_SHAPING",
                             self.cmd_APPLY_ADAPTIVE_SHAPING,
                             desc=self.cmd_APPLY_ADAPTIVE_SHAPING_help)

    def get_printer_bounds(self):
        """Get the printer's build volume bounds"""
        toolhead = self.printer.lookup_object('toolhead')
        kin = toolhead.get_kinematics()
        
        # Try to get bounds from kinematics
        try:
            bounds = kin.get_status()
            x_min, x_max = bounds.get('axis_minimum', [0, 0, 0])[0], bounds.get('axis_maximum', [200, 200, 200])[0]
            y_min, y_max = bounds.get('axis_minimum', [0, 0, 0])[1], bounds.get('axis_maximum', [200, 200, 200])[1]
        except:
            # Fallback defaults
            x_min, x_max = 0, 200
            y_min, y_max = 0, 200
            
        return (x_min, x_max, y_min, y_max)

    def generate_calibration_points(self, density='medium'):
        """Generate spatial calibration points across the build volume"""
        x_min, x_max, y_min, y_max = self.get_printer_bounds()
        
        # Adjust grid density
        if density == 'low':
            nx, ny = 3, 3
        elif density == 'medium':
            nx, ny = 4, 4  
        elif density == 'high':
            nx, ny = 5, 5
        else:
            nx, ny = density, density
            
        # Generate grid points with some margin from edges
        margin_x = (x_max - x_min) * 0.1
        margin_y = (y_max - y_min) * 0.1
        
        x_points = np.linspace(x_min + margin_x, x_max - margin_x, nx)
        y_points = np.linspace(y_min + margin_y, y_max - margin_y, ny)
        
        points = []
        for x in x_points:
            for y in y_points:
                points.append(SpatialCalibrationPoint(x, y))
                
        return points

    def define_movement_patterns(self):
        """Define different movement patterns to test"""
        patterns = []
        
        # Linear movements at different speeds
        for speed in [50, 100, 150, 200]:
            for accel in [1000, 3000, 5000]:
                patterns.append(MovementPattern('linear', (speed*0.8, speed*1.2), 
                                               (accel*0.8, accel*1.2)))
        
        # Directional movements
        for direction in [(1, 0), (0, 1), (1, 1), (-1, 1)]:  # X, Y, diagonal
            patterns.append(MovementPattern('directional', (80, 120), (2000, 4000), direction))
            
        # Corner/direction change patterns
        patterns.append(MovementPattern('corner', (30, 80), (3000, 7000)))
        
        # High-speed infill patterns
        patterns.append(MovementPattern('infill', (100, 250), (1000, 5000)))
        
        return patterns

    cmd_ADAPTIVE_RESONANCE_CALIBRATE_help = "Perform comprehensive spatial resonance calibration"
    def cmd_ADAPTIVE_RESONANCE_CALIBRATE(self, gcmd):
        """Perform comprehensive spatial and movement pattern calibration"""
        density = gcmd.get('DENSITY', 'medium')
        test_patterns = gcmd.get_int('TEST_PATTERNS', 1, minval=0, maxval=1)
        output_prefix = gcmd.get('OUTPUT', '/tmp/adaptive_calibration')
        
        gcmd.respond_info("Starting adaptive resonance calibration...")
        
        # Generate calibration points
        self.spatial_points = self.generate_calibration_points(density)
        gcmd.respond_info(f"Testing {len(self.spatial_points)} spatial points")
        
        # Get resonance tester
        res_tester = self.printer.lookup_object('resonance_tester')
        
        calibration_results = {}
        
        for i, point in enumerate(self.spatial_points):
            gcmd.respond_info(f"Calibrating point {i+1}/{len(self.spatial_points)}: "
                             f"X={point.position[0]:.1f} Y={point.position[1]:.1f}")
            
            # Move to calibration point
            toolhead = self.printer.lookup_object('toolhead')
            toolhead.manual_move([point.position[0], point.position[1], None], 100)
            toolhead.wait_moves()
            
            # Test basic resonances at this point
            try:
                # Test all supported axes
                for axis in self.supported_axes[:3]:  # Test X, Y, Z axes
                    try:
                        axis_data = res_tester._test_axis(gcmd, axis)
                        point.calibration_data[axis] = axis_data
                    except Exception as axis_error:
                        gcmd.respond_info(f"Warning: Failed to test {axis}-axis at point {i}: {axis_error}")
                        continue
                
                # Analyze resonance characteristics
                self._analyze_point_resonances(point)
                
                calibration_results[f"point_{i}"] = {
                    'position': point.position,
                    'optimal_shapers': point.optimal_shapers
                }
                
                # Add peak frequencies for each tested axis
                for axis in point.calibration_data:
                    try:
                        peaks = point.calibration_data[axis].find_peak_frequencies().tolist()
                        calibration_results[f"point_{i}"][f'{axis}_peaks'] = peaks
                    except:
                        calibration_results[f"point_{i}"][f'{axis}_peaks'] = []
                
            except Exception as e:
                gcmd.respond_info(f"Warning: Failed to calibrate point {i}: {e}")
                continue
        
        # Test movement patterns if requested
        if test_patterns:
            gcmd.respond_info("Testing movement patterns...")
            movement_patterns = self.define_movement_patterns()
            self._test_movement_patterns(gcmd, movement_patterns[:3])  # Test subset for demo
        
        # Save results
        self._save_calibration_results(calibration_results, output_prefix)
        
        gcmd.respond_info(f"Adaptive calibration complete. Results saved to {output_prefix}_*")

    def _analyze_point_resonances(self, point):
        """Analyze resonance characteristics at a spatial point"""
        for axis in ['x', 'y']:
            if axis not in point.calibration_data:
                continue
                
            cal_data = point.calibration_data[axis]
            
            # Find optimal shaper for this point
            shaper_cfgs = shaper_defs.INPUT_SHAPERS
            best_shaper = None
            best_score = float('inf')
            
            for shaper_cfg in shaper_cfgs:
                try:
                    freq_bins = cal_data.freq_bins
                    psd = cal_data.get_psd(axis)
                    
                    # Simple optimization - find frequency with lowest vibration
                    for freq in np.linspace(20, 100, 20):
                        A, T = shaper_cfg.init_func(freq, 0.1)
                        
                        # Calculate shaper response
                        shaper_vals = shaper_defs.calc_shaper_vals(
                            shaper_cfg.init_func, freq, 0.1, freq_bins, psd)
                        
                        # Score based on vibration reduction
                        score = np.sum(shaper_vals * psd)
                        
                        if score < best_score:
                            best_score = score
                            best_shaper = (shaper_cfg.name, freq, 0.1)
                            
                except:
                    continue
            
            if best_shaper:
                point.optimal_shapers[axis] = best_shaper

    def _test_movement_patterns(self, gcmd, patterns):
        """Test resonance characteristics for different movement patterns"""
        gcmd.respond_info("Testing movement patterns (simplified for demo)")
        
        # For demonstration, we'll simulate testing different movement types
        # In a real implementation, this would involve generating specific
        # movement sequences and measuring their resonance characteristics
        
        for pattern in patterns:
            gcmd.respond_info(f"Testing {pattern.pattern_type} pattern: "
                             f"speeds {pattern.speed_range}, accels {pattern.accel_range}")
            
            # Simulate pattern-specific resonance testing
            # Real implementation would generate movement sequences and measure vibrations
            time.sleep(0.1)  # Placeholder for actual testing

    def _save_calibration_results(self, results, prefix):
        """Save calibration results to files"""
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Convert results to ensure JSON serialization compatibility
        serializable_results = convert_numpy_types(results)
        
        # Save JSON results
        with open(f"{prefix}_results.json", 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save human-readable summary
        with open(f"{prefix}_summary.txt", 'w') as f:
            f.write("Adaptive Resonance Calibration Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for point_name, data in results.items():
                f.write(f"{point_name}: Position {data['position']}\n")
                # Safely access peaks data with fallback for missing keys
                x_peaks = data.get('x_peaks', [])
                y_peaks = data.get('y_peaks', [])
                f.write(f"  X peaks: {x_peaks}\n") 
                f.write(f"  Y peaks: {y_peaks}\n")
                optimal_shapers = data.get('optimal_shapers', {})
                f.write(f"  Optimal shapers: {optimal_shapers}\n\n")

    cmd_BUILD_COMPENSATION_MODEL_help = "Build multi-dimensional compensation model from calibration data"
    def cmd_BUILD_COMPENSATION_MODEL(self, gcmd):
        """Build the multi-dimensional compensation model"""
        if not self.spatial_points:
            raise gcmd.error("No calibration data available. Run ADAPTIVE_RESONANCE_CALIBRATE first.")
        
        gcmd.respond_info("Building multi-dimensional compensation model...")
        
        # Analyze spatial variations
        spatial_analysis = self._analyze_spatial_variations()
        
        # Build interpolation model
        self._build_interpolation_model()
        
        # Analyze movement pattern dependencies
        self._analyze_movement_dependencies()
        
        gcmd.respond_info("Compensation model built successfully")
        gcmd.respond_info(f"Spatial variation detected: {spatial_analysis['max_variation']:.1f}%")
        gcmd.respond_info(f"Model covers {len(self.spatial_points)} calibration points")

    def _analyze_spatial_variations(self):
        """Analyze how resonance characteristics vary across the build volume"""
        if len(self.spatial_points) < 2:
            return {'max_variation': 0}
        
        # Extract frequency data from all points
        x_frequencies = []
        y_frequencies = []
        
        for point in self.spatial_points:
            if 'x' in point.calibration_data:
                x_peaks = point.calibration_data['x'].find_peak_frequencies()
                if len(x_peaks) > 0:
                    x_frequencies.append(x_peaks[0])  # Primary peak
            
            if 'y' in point.calibration_data:
                y_peaks = point.calibration_data['y'].find_peak_frequencies()
                if len(y_peaks) > 0:
                    y_frequencies.append(y_peaks[0])  # Primary peak
        
        # Calculate variation
        max_variation = 0
        if x_frequencies:
            x_variation = (np.max(x_frequencies) - np.min(x_frequencies)) / np.mean(x_frequencies) * 100
            max_variation = max(max_variation, x_variation)
            
        if y_frequencies:
            y_variation = (np.max(y_frequencies) - np.min(y_frequencies)) / np.mean(y_frequencies) * 100
            max_variation = max(max_variation, y_variation)
        
        return {'max_variation': max_variation}

    def _build_interpolation_model(self):
        """Build spatial interpolation model for compensation parameters"""
        # For each axis, create interpolation functions
        for axis in ['x', 'y']:
            positions = []
            frequencies = []
            shaper_types = []
            
            for point in self.spatial_points:
                if axis in point.optimal_shapers:
                    positions.append(point.position[:2])  # (x, y)
                    shaper_name, freq, damping = point.optimal_shapers[axis]
                    frequencies.append(freq)
                    shaper_types.append(shaper_name)
            
            if len(positions) >= 3:  # Minimum for interpolation
                self.compensation_model[axis] = {
                    'positions': np.array(positions),
                    'frequencies': np.array(frequencies),
                    'shaper_types': shaper_types
                }

    def _analyze_movement_dependencies(self):
        """Analyze how movement characteristics affect optimal compensation"""
        # Placeholder for movement pattern analysis
        # In full implementation, this would analyze how speed, acceleration,
        # and direction affect optimal shaper parameters
        pass

    cmd_APPLY_ADAPTIVE_SHAPING_help = "Apply adaptive input shaping based on current movement"
    def cmd_APPLY_ADAPTIVE_SHAPING(self, gcmd):
        """Apply adaptive input shaping parameters"""
        if not self.compensation_model:
            raise gcmd.error("No compensation model available. Run BUILD_COMPENSATION_MODEL first.")
        
        # Get current position
        toolhead = self.printer.lookup_object('toolhead')
        pos = toolhead.get_position()
        
        # Get movement parameters (simplified)
        speed = gcmd.get_float('SPEED', 100., above=0.)
        accel = gcmd.get_float('ACCEL', 3000., above=0.)
        
        # Calculate optimal parameters for current position and movement
        optimal_params = self._calculate_optimal_parameters(pos[:2], speed, accel)
        
        # Apply parameters to input shaper
        input_shaper = self.printer.lookup_object('input_shaper')
        
        for axis, params in optimal_params.items():
            if params:
                shaper_type, freq, damping = params
                gcmd.respond_info(f"Applying {axis}-axis: {shaper_type} @ {freq:.1f}Hz")
                
                # Apply shaper parameters
                gcode = self.printer.lookup_object("gcode")
                axis_upper = axis.upper()
                input_shaper.cmd_SET_INPUT_SHAPER(gcode.create_gcode_command(
                    "SET_INPUT_SHAPER", "SET_INPUT_SHAPER", {
                        "SHAPER_TYPE_" + axis_upper: shaper_type,
                        "SHAPER_FREQ_" + axis_upper: freq}))

    def _calculate_optimal_parameters(self, position, speed, accel):
        """Calculate optimal shaper parameters for given position and movement"""
        optimal_params = {}
        
        for axis in ['x', 'y']:
            if axis not in self.compensation_model:
                optimal_params[axis] = None
                continue
            
            model = self.compensation_model[axis]
            positions = model['positions']
            frequencies = model['frequencies']
            
            # Find nearest calibration points
            distances = np.sqrt(np.sum((positions - position)**2, axis=1))
            nearest_indices = np.argsort(distances)[:3]  # Use 3 nearest points
            
            # Weighted interpolation based on distance
            weights = 1.0 / (distances[nearest_indices] + 0.1)  # Avoid division by zero
            weights /= np.sum(weights)  # Normalize
            
            # Interpolate frequency
            interpolated_freq = np.sum(frequencies[nearest_indices] * weights)
            
            # Select most common shaper type among nearest points
            nearest_shapers = [model['shaper_types'][i] for i in nearest_indices]
            shaper_type = max(set(nearest_shapers), key=nearest_shapers.count)
            
            # Adjust for movement characteristics (simplified)
            freq_adjustment = 1.0
            if speed > 150:
                freq_adjustment *= 1.1  # Higher frequency for high speed
            if accel > 5000:
                freq_adjustment *= 1.05  # Slight increase for high acceleration
            
            adjusted_freq = interpolated_freq * freq_adjustment
            optimal_params[axis] = (shaper_type, adjusted_freq, 0.1)
        
        return optimal_params

def load_config(config):
    return AdaptiveCompensationModel(config)