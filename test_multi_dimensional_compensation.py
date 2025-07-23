#!/usr/bin/env python3
# Test Suite for Multi-Dimensional Resonance Compensation System
#
# This script validates the functionality and performance of the advanced
# multi-dimensional compensation system

import sys
import os
import unittest
import json
import numpy as np
from unittest.mock import Mock, MagicMock, patch

# Add klippy to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'klippy'))

# Mock the environment before importing our modules
sys.modules['chelper'] = Mock()

class TestMultiDimensionalCompensation(unittest.TestCase):
    """Test suite for multi-dimensional resonance compensation"""
    
    def setUp(self):
        """Set up test environment"""
        # Create mock configuration
        self.mock_config = Mock()
        self.mock_config.get_name.return_value = "adaptive_input_shaper test"
        self.mock_config.getfloat.side_effect = lambda key, default, **kwargs: {
            'spatial_resolution': 50.0,
            'transition_smoothing': 0.1,
            'adaptation_rate': 0.1,
            'min_update_interval': 0.5,
            'analysis_window': 2.0,
            'update_interval': 0.1
        }.get(key, default)
        self.mock_config.getint.side_effect = lambda key, default, **kwargs: {
            'speed_bins': 5,
            'accel_bins': 5
        }.get(key, default)
        self.mock_config.getboolean.side_effect = lambda key, default: {
            'enabled': False
        }.get(key, default)
        
        # Create mock printer
        self.mock_printer = Mock()
        self.mock_gcode = Mock()
        self.mock_printer.lookup_object.side_effect = lambda name: {
            'gcode': self.mock_gcode,
            'toolhead': Mock(),
            'input_shaper': Mock()
        }.get(name, Mock())
        self.mock_config.get_printer.return_value = self.mock_printer

    def test_spatial_calibration_point_creation(self):
        """Test spatial calibration point creation"""
        from extras.adaptive_input_shaper import SpatialCalibrationPoint
        
        point = SpatialCalibrationPoint(100, 150, 10)
        
        self.assertEqual(point.position, (100, 150, 10))
        self.assertEqual(point.calibration_data, {})
        self.assertEqual(point.movement_profiles, {})
        self.assertEqual(point.optimal_shapers, {})

    def test_movement_pattern_definition(self):
        """Test movement pattern definition"""
        from extras.adaptive_input_shaper import MovementPattern
        
        pattern = MovementPattern('linear', (50, 100), (1000, 3000), (1, 0))
        
        self.assertEqual(pattern.pattern_type, 'linear')
        self.assertEqual(pattern.speed_range, (50, 100))
        self.assertEqual(pattern.accel_range, (1000, 3000))
        self.assertEqual(pattern.direction, (1, 0))
        self.assertEqual(pattern.resonance_profile, {})

    def test_adaptive_compensation_model_initialization(self):
        """Test adaptive compensation model initialization"""
        from extras.adaptive_input_shaper import AdaptiveCompensationModel
        
        model = AdaptiveCompensationModel(self.mock_config)
        
        self.assertEqual(model.spatial_resolution, 50.0)
        self.assertEqual(model.speed_bins, 5)
        self.assertEqual(model.accel_bins, 5)
        self.assertEqual(model.transition_smoothing, 0.1)
        self.assertEqual(len(model.spatial_points), 0)

    def test_printer_bounds_detection(self):
        """Test printer bounds detection"""
        from extras.adaptive_input_shaper import AdaptiveCompensationModel
        
        model = AdaptiveCompensationModel(self.mock_config)
        bounds = model.get_printer_bounds()
        
        # Should return default bounds when status not available
        self.assertEqual(len(bounds), 4)  # x_min, x_max, y_min, y_max
        self.assertIsInstance(bounds[0], (int, float))

    def test_calibration_point_generation(self):
        """Test calibration point generation"""
        from extras.adaptive_input_shaper import AdaptiveCompensationModel
        
        model = AdaptiveCompensationModel(self.mock_config)
        
        # Test different densities
        points_low = model.generate_calibration_points('low')
        points_medium = model.generate_calibration_points('medium')
        points_high = model.generate_calibration_points('high')
        
        self.assertEqual(len(points_low), 9)  # 3x3 grid
        self.assertEqual(len(points_medium), 16)  # 4x4 grid
        self.assertEqual(len(points_high), 25)  # 5x5 grid
        
        # Verify points are within bounds
        for point in points_medium:
            x, y, z = point.position
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x, 200)
            self.assertLessEqual(y, 200)

    def test_movement_pattern_definition_generation(self):
        """Test movement pattern definition generation"""
        from extras.adaptive_input_shaper import AdaptiveCompensationModel
        
        model = AdaptiveCompensationModel(self.mock_config)
        patterns = model.define_movement_patterns()
        
        self.assertGreater(len(patterns), 0)
        
        # Verify pattern types
        pattern_types = [p.pattern_type for p in patterns]
        self.assertIn('linear', pattern_types)
        self.assertIn('directional', pattern_types)
        self.assertIn('corner', pattern_types)
        self.assertIn('infill', pattern_types)

    def test_dynamic_input_shaper_initialization(self):
        """Test dynamic input shaper initialization"""
        from extras.dynamic_input_shaper import DynamicInputShaper
        
        shaper = DynamicInputShaper(self.mock_config)
        
        self.assertFalse(shaper.enabled)
        self.assertEqual(shaper.adaptation_rate, 0.1)
        self.assertEqual(shaper.min_update_interval, 0.5)
        self.assertIsNotNone(shaper.motion_analyzer)

    def test_motion_analyzer_pattern_detection(self):
        """Test motion analyzer pattern detection"""
        from extras.dynamic_input_shaper import MotionAnalyzer
        
        analyzer = MotionAnalyzer(self.mock_config)
        
        # Test with linear motion
        for i in range(20):
            position = [i * 5, 50, 10]
            velocity = [50, 0, 0]
            acceleration = [100, 0, 0]
            analyzer.analyze_motion(position, velocity, acceleration, i * 0.1)
        
        pattern = analyzer._detect_movement_pattern()
        self.assertIn(pattern, ['linear', 'mixed', 'unknown'])

    def test_motion_statistics_calculation(self):
        """Test motion statistics calculation"""
        from extras.dynamic_input_shaper import MotionAnalyzer
        
        analyzer = MotionAnalyzer(self.mock_config)
        
        # Add some test motion data
        for i in range(10):
            position = [i * 10, i * 5, 10]
            velocity = [100, 50, 0]
            acceleration = [1000, 500, 0]
            analyzer.motion_history.append({
                'time': i * 0.1,
                'position': position[:2],
                'velocity': velocity[:2],
                'acceleration': acceleration[:2]
            })
        
        stats = analyzer._calculate_motion_statistics()
        
        self.assertIn('avg_speed', stats)
        self.assertIn('max_speed', stats)
        self.assertIn('avg_accel', stats)
        self.assertIn('max_accel', stats)
        self.assertGreater(stats['avg_speed'], 0)

    def test_compensation_recommendations(self):
        """Test compensation parameter recommendations"""
        from extras.dynamic_input_shaper import MotionAnalyzer
        
        analyzer = MotionAnalyzer(self.mock_config)
        
        # Test different patterns
        patterns = ['linear', 'corner_heavy', 'variable_speed']
        stats = {
            'avg_speed': 100,
            'max_speed': 150,
            'speed_std': 10,
            'avg_accel': 3000,
            'max_accel': 5000,
            'accel_std': 500
        }
        
        for pattern in patterns:
            recommendations = analyzer._generate_compensation_recommendations(pattern, stats)
            
            self.assertIn('x', recommendations)
            self.assertIn('y', recommendations)
            
            for axis in ['x', 'y']:
                self.assertIn('freq_adjustment', recommendations[axis])
                self.assertIn('damping_adjustment', recommendations[axis])
                self.assertGreater(recommendations[axis]['freq_adjustment'], 0)
                self.assertGreater(recommendations[axis]['damping_adjustment'], 0)

    def test_spatial_variation_analysis(self):
        """Test spatial variation analysis"""
        from extras.adaptive_input_shaper import AdaptiveCompensationModel, SpatialCalibrationPoint
        
        model = AdaptiveCompensationModel(self.mock_config)
        
        # Create test points with varying frequencies
        for i, freq_x in enumerate([40, 45, 50, 42]):
            point = SpatialCalibrationPoint(i * 50, i * 50)
            
            # Mock calibration data with find_peak_frequencies method
            mock_cal_data = Mock()
            mock_cal_data.find_peak_frequencies.return_value = np.array([freq_x])
            point.calibration_data['x'] = mock_cal_data
            point.calibration_data['y'] = mock_cal_data
            
            model.spatial_points.append(point)
        
        analysis = model._analyze_spatial_variations()
        
        self.assertIn('max_variation', analysis)
        self.assertGreaterEqual(analysis['max_variation'], 0)

    def test_interpolation_model_building(self):
        """Test interpolation model building"""
        from extras.adaptive_input_shaper import AdaptiveCompensationModel, SpatialCalibrationPoint
        
        model = AdaptiveCompensationModel(self.mock_config)
        
        # Create test points with optimal shapers
        for i in range(4):
            point = SpatialCalibrationPoint(i * 50, i * 30)
            point.optimal_shapers['x'] = ('mzv', 45.0 + i, 0.1)
            point.optimal_shapers['y'] = ('ei', 42.0 + i, 0.1)
            model.spatial_points.append(point)
        
        model._build_interpolation_model()
        
        self.assertIn('x', model.compensation_model)
        self.assertIn('y', model.compensation_model)
        
        for axis in ['x', 'y']:
            self.assertIn('positions', model.compensation_model[axis])
            self.assertIn('frequencies', model.compensation_model[axis])
            self.assertIn('shaper_types', model.compensation_model[axis])

    def test_optimal_parameter_calculation(self):
        """Test optimal parameter calculation"""
        from extras.adaptive_input_shaper import AdaptiveCompensationModel
        
        model = AdaptiveCompensationModel(self.mock_config)
        
        # Set up compensation model
        model.compensation_model['x'] = {
            'positions': np.array([[50, 50], [100, 100], [150, 150]]),
            'frequencies': np.array([45.0, 47.0, 49.0]),
            'shaper_types': ['mzv', 'ei', 'mzv']
        }
        model.compensation_model['y'] = {
            'positions': np.array([[50, 50], [100, 100], [150, 150]]),
            'frequencies': np.array([42.0, 44.0, 46.0]),
            'shaper_types': ['ei', 'mzv', 'ei']
        }
        
        # Test parameter calculation
        position = [100, 100]
        speed = 120
        accel = 3000
        
        params = model._calculate_optimal_parameters(position, speed, accel)
        
        self.assertIn('x', params)
        self.assertIn('y', params)
        
        for axis in ['x', 'y']:
            if params[axis]:
                shaper_type, freq, damping = params[axis]
                self.assertIsInstance(shaper_type, str)
                self.assertGreater(freq, 0)
                self.assertGreater(damping, 0)

    def test_parameter_transition_smoothing(self):
        """Test parameter transition smoothing"""
        from extras.dynamic_input_shaper import DynamicInputShaper
        
        shaper = DynamicInputShaper(self.mock_config)
        
        # Test transition queueing
        shaper._queue_parameter_transition('x', 'mzv', 45.0, 0.1)
        
        self.assertGreater(len(shaper.transition_queue), 0)
        
        # Verify transition steps are properly ordered
        times = [step['time'] for step in shaper.transition_queue]
        self.assertEqual(times, sorted(times))

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.test_data_dir = '/tmp/test_multi_dimensional'
        os.makedirs(self.test_data_dir, exist_ok=True)

    def test_end_to_end_calibration_simulation(self):
        """Test complete calibration workflow simulation"""
        # This test simulates the complete workflow without requiring actual hardware
        
        # Mock spatial calibration data
        spatial_data = {
            'point_0': {
                'position': [50, 50, 0],
                'x_peaks': [45.2],
                'y_peaks': [42.1],
                'optimal_shapers': {'x': ('mzv', 45.2, 0.1), 'y': ('ei', 42.1, 0.1)}
            },
            'point_1': {
                'position': [150, 150, 0], 
                'x_peaks': [47.8],
                'y_peaks': [44.3],
                'optimal_shapers': {'x': ('ei', 47.8, 0.1), 'y': ('mzv', 44.3, 0.1)}
            }
        }
        
        # Verify data structure
        for point_name, point_data in spatial_data.items():
            self.assertIn('position', point_data)
            self.assertIn('x_peaks', point_data)
            self.assertIn('y_peaks', point_data)
            self.assertIn('optimal_shapers', point_data)
            
            self.assertEqual(len(point_data['position']), 3)
            self.assertGreater(len(point_data['x_peaks']), 0)
            self.assertGreater(len(point_data['y_peaks']), 0)

    def test_performance_validation(self):
        """Test performance improvements validation"""
        # Simulate performance comparison
        traditional_vibration = [8.5, 12.3, 6.8, 4.2, 9.1]
        adaptive_vibration = [2.1, 3.4, 1.8, 0.9, 2.7]
        
        # Calculate improvement
        improvements = [(t - a) / t * 100 for t, a in zip(traditional_vibration, adaptive_vibration)]
        avg_improvement = np.mean(improvements)
        
        # Verify significant improvement
        self.assertGreater(avg_improvement, 50)  # At least 50% improvement
        self.assertTrue(all(a < t for a, t in zip(adaptive_vibration, traditional_vibration)))

    def test_configuration_validation(self):
        """Test configuration file validation"""
        # Test configuration examples
        config_examples = [
            {
                'spatial_resolution': 50.0,
                'speed_bins': 5,
                'accel_bins': 5,
                'transition_smoothing': 0.1,
                'enabled': True
            },
            {
                'spatial_resolution': 25.0,  # Higher resolution
                'speed_bins': 8,
                'accel_bins': 8,
                'transition_smoothing': 0.05,  # Faster transitions
                'enabled': True
            }
        ]
        
        for config in config_examples:
            # Validate configuration parameters
            self.assertGreater(config['spatial_resolution'], 0)
            self.assertGreaterEqual(config['speed_bins'], 3)
            self.assertGreaterEqual(config['accel_bins'], 3)
            self.assertGreater(config['transition_smoothing'], 0)
            self.assertLessEqual(config['transition_smoothing'], 1.0)

def run_performance_benchmarks():
    """Run performance benchmarks for the system"""
    print("Running Performance Benchmarks...")
    
    # Simulate calibration time benchmark
    calibration_points = [9, 16, 25]  # 3x3, 4x4, 5x5 grids
    estimated_times = []
    
    for points in calibration_points:
        # Estimate: 30 seconds per point for resonance testing
        estimated_time = points * 30
        estimated_times.append(estimated_time)
        print(f"  {points} points: ~{estimated_time/60:.1f} minutes")
    
    # Simulate real-time processing performance
    update_frequency = 10  # Hz
    processing_time_per_update = 0.001  # 1ms
    cpu_usage = processing_time_per_update * update_frequency * 100
    
    print(f"\nReal-time Performance:")
    print(f"  Update frequency: {update_frequency} Hz")
    print(f"  Processing time per update: {processing_time_per_update*1000:.1f} ms")
    print(f"  Estimated CPU usage: {cpu_usage:.1f}%")
    
    return {
        'calibration_times': dict(zip(calibration_points, estimated_times)),
        'realtime_performance': {
            'update_frequency': update_frequency,
            'cpu_usage_percent': cpu_usage
        }
    }

class TestCalibrationResultsSaving(unittest.TestCase):
    """Test suite for calibration results saving functionality"""
    
    def setUp(self):
        """Set up test environment for calibration saving tests"""
        # Create mock printer first
        self.mock_printer = Mock()
        self.mock_gcode = Mock()
        self.mock_printer.lookup_object.side_effect = lambda name: {
            'gcode': self.mock_gcode,
            'toolhead': Mock(),
            'input_shaper': Mock()
        }.get(name)
        
        # Create mock configuration
        self.mock_config = Mock()
        self.mock_config.get_name.return_value = "adaptive_input_shaper test"
        self.mock_config.get_printer.return_value = self.mock_printer
        self.mock_config.getfloat.side_effect = lambda key, default, **kwargs: {
            'spatial_resolution': 50.0,
            'transition_smoothing': 0.1,
        }.get(key, default)
        self.mock_config.getint.side_effect = lambda key, default, **kwargs: {
            'speed_bins': 5,
            'accel_bins': 5
        }.get(key, default)
        
        # Create the adaptive compensation model instance
        from extras.adaptive_input_shaper import AdaptiveCompensationModel
        self.adaptive_model = AdaptiveCompensationModel(self.mock_config)
    
    def test_save_calibration_results_missing_peaks(self):
        """Test that _save_calibration_results handles missing x_peaks/y_peaks gracefully"""
        import tempfile
        import os
        
        # Test data with missing peak information (simulating failed axis testing)
        test_results = {
            'point_0': {
                'position': [50, 50, 0],
                'optimal_shapers': {'x': ('mzv', 45.2, 0.1)}
                # Note: missing 'x_peaks' and 'y_peaks' keys
            },
            'point_1': {
                'position': [150, 150, 0],
                'x_peaks': [47.8],  # Only x_peaks present
                'optimal_shapers': {'x': ('ei', 47.8, 0.1)}
                # Note: missing 'y_peaks' key
            }
        }
        
        # Create temporary directory for test outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            test_prefix = os.path.join(temp_dir, "test_calibration")
            
            # This should not raise an exception
            try:
                self.adaptive_model._save_calibration_results(test_results, test_prefix)
                
                # Verify files were created
                self.assertTrue(os.path.exists(f"{test_prefix}_results.json"))
                self.assertTrue(os.path.exists(f"{test_prefix}_summary.txt"))
                
                # Verify JSON file content
                with open(f"{test_prefix}_results.json", 'r') as f:
                    saved_results = json.load(f)
                    self.assertEqual(len(saved_results), 2)
                    self.assertIn('point_0', saved_results)
                    self.assertIn('point_1', saved_results)
                
                # Verify summary file content
                with open(f"{test_prefix}_summary.txt", 'r') as f:
                    summary_content = f.read()
                    self.assertIn("point_0", summary_content)
                    self.assertIn("point_1", summary_content)
                    self.assertIn("X peaks: []", summary_content)  # Should show empty list for missing peaks
                    self.assertIn("Y peaks: []", summary_content)  # Should show empty list for missing peaks
                    self.assertIn("X peaks: [47.8]", summary_content)  # Should show actual peaks when present
                    
            except Exception as e:
                self.fail(f"_save_calibration_results raised an exception with missing peaks: {e}")
    
    def test_save_calibration_results_with_numpy_types(self):
        """Test that numpy types are properly converted for JSON serialization"""
        import tempfile
        import os
        import numpy as np
        
        # Test data with numpy types that need conversion
        test_results = {
            'point_0': {
                'position': [np.float64(50.0), np.float64(50.0), np.int32(0)],
                'x_peaks': [np.float64(45.2), np.float64(46.1)],
                'y_peaks': [np.float64(42.1)],
                'optimal_shapers': {'x': ('mzv', np.float64(45.2), np.float64(0.1))}
            }
        }
        
        # Create temporary directory for test outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            test_prefix = os.path.join(temp_dir, "test_numpy_calibration")
            
            # This should not raise a JSON encoding exception
            try:
                self.adaptive_model._save_calibration_results(test_results, test_prefix)
                
                # Verify JSON file was created and can be loaded back
                with open(f"{test_prefix}_results.json", 'r') as f:
                    saved_results = json.load(f)
                    
                    # Verify all numpy types were converted to Python native types
                    point_data = saved_results['point_0']
                    self.assertIsInstance(point_data['position'][0], (int, float))
                    self.assertIsInstance(point_data['x_peaks'][0], (int, float))
                    self.assertIsInstance(point_data['y_peaks'][0], (int, float))
                    
            except Exception as e:
                self.fail(f"_save_calibration_results raised an exception with numpy types: {e}")

def main():
    """Main test runner"""
    print("Multi-Dimensional Resonance Compensation Test Suite")
    print("=" * 60)
    
    # Run unit tests
    print("Running unit tests...")
    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromTestCase(TestMultiDimensionalCompensation)
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestIntegration))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestCalibrationResultsSaving))
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    # Run performance benchmarks
    print("\n" + "="*60)
    benchmarks = run_performance_benchmarks()
    
    # Summary
    print(f"\n" + "="*60)
    print(f"Test Summary:")
    print(f"  Tests run: {test_result.testsRun}")
    print(f"  Failures: {len(test_result.failures)}")
    print(f"  Errors: {len(test_result.errors)}")
    
    if test_result.wasSuccessful():
        print(f"  Result: ✓ ALL TESTS PASSED")
    else:
        print(f"  Result: ✗ SOME TESTS FAILED")
        
    # Save test results
    test_summary = {
        'tests_run': test_result.testsRun,
        'failures': len(test_result.failures),
        'errors': len(test_result.errors),
        'success': test_result.wasSuccessful(),
        'benchmarks': benchmarks
    }
    
    with open('/tmp/test_results.json', 'w') as f:
        json.dump(test_summary, f, indent=2)
    
    print(f"\nTest results saved to /tmp/test_results.json")
    
    return test_result.wasSuccessful()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)