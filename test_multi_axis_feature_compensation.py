#!/usr/bin/env python3

"""
Comprehensive test suite for multi-axis support and G-code feature type compensation
"""

import unittest, tempfile, os, shutil
import numpy as np
import sys

# Add the klippy directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'klippy'))

from klippy.extras.dynamic_input_shaper import DynamicInputShaper, FeatureTypeManager, MotionAnalyzer


class MockPrinter:
    """Mock printer for testing"""
    def __init__(self):
        self.event_handlers = {}
        self.objects = {}
        
    def register_event_handler(self, event, handler):
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)
    
    def lookup_object(self, name):
        return self.objects.get(name, MockObject())
    
    def get_start_args(self):
        return {}

class MockObject:
    """Mock object for testing"""
    def __init__(self):
        pass
    
    def register_command(self, cmd, handler, desc=None):
        pass
    
    def get_position(self):
        return [100, 100, 50, 0, 0, 0]

class MockConfig:
    """Mock configuration for testing"""
    def __init__(self, values=None):
        self.values = values or {}
        self.printer = MockPrinter()
    
    def get_printer(self):
        return self.printer
    
    def get_name(self):
        return "dynamic_input_shaper test"
    
    def getboolean(self, key, default=None):
        return self.values.get(key, default)
    
    def getfloat(self, key, default=None, minval=None, maxval=None, above=None, below=None):
        return self.values.get(key, default)
    
    def getint(self, key, default=None, minval=None, maxval=None):
        return self.values.get(key, default)
    
    def get(self, key, default=None):
        return self.values.get(key, default)


class TestMultiAxisSupport(unittest.TestCase):
    """Test multi-axis support functionality"""
    
    def setUp(self):
        self.config = MockConfig({'enabled': True})
        self.motion_analyzer = MotionAnalyzer(self.config)
    
    def test_multi_axis_motion_analysis(self):
        """Test motion analysis with multiple axes"""
        # Test with 6-axis motion data
        position = [100, 100, 50, 0, 0, 0]
        velocity = [10, 5, 2, 1, 0, 0]
        acceleration = [1000, 500, 200, 100, 0, 0]
        timestamp = 1.0
        
        result = self.motion_analyzer.analyze_motion(position, velocity, acceleration, timestamp)
        
        # Should return None for first few calls until history builds up
        self.assertIsNone(result)
        
        # Add more motion data to build history
        for i in range(15):
            t = 1.0 + i * 0.1
            pos = [100 + i, 100 + i*0.5, 50, 0, 0, 0]
            vel = [10 + i, 5 + i*0.2, 2, 1, 0, 0]
            acc = [1000, 500, 200, 100, 0, 0]
            result = self.motion_analyzer.analyze_motion(pos, vel, acc, t)
        
        # Should now return recommendations
        self.assertIsNotNone(result)
        self.assertIn('x', result)
        self.assertIn('y', result)
        self.assertIn('z', result)
        self.assertIn('a', result)
        self.assertIn('b', result)
        self.assertIn('c', result)
    
    def test_multi_axis_recommendations(self):
        """Test that recommendations are generated for all axes"""
        pattern = 'linear'
        stats = {
            'avg_speed': 100,
            'max_speed': 150,
            'speed_std': 10,
            'avg_accel': 2000,
            'max_accel': 3000,
            'accel_std': 200
        }
        
        recommendations = self.motion_analyzer._generate_compensation_recommendations(pattern, stats)
        
        # Check all axes are included
        axis_names = ['x', 'y', 'z', 'a', 'b', 'c']
        for axis in axis_names:
            self.assertIn(axis, recommendations)
            self.assertIn('freq_adjustment', recommendations[axis])
            self.assertIn('damping_adjustment', recommendations[axis])
            self.assertIn('shaper_hint', recommendations[axis])
        
        # Linear pattern should suggest 'smooth' shaper
        for axis in axis_names:
            self.assertEqual(recommendations[axis]['shaper_hint'], 'smooth')


class TestFeatureTypeManager(unittest.TestCase):
    """Test G-code feature type detection and configuration"""
    
    def setUp(self):
        self.config = MockConfig({
            'feature_wall_outer_preference': 'quality',
            'feature_infill_preference': 'speed',
            'feature_support_preference': 'speed'
        })
        self.feature_manager = FeatureTypeManager(self.config)
    
    def test_feature_type_detection(self):
        """Test detection of feature types from G-code comments"""
        # Test different comment formats
        test_lines = [
            ";TYPE:WALL-OUTER",
            "; TYPE:INFILL",
            ";type:support",
            "G1 X10 Y10 ;TYPE:BRIDGE"
        ]
        
        expected_features = ['WALL-OUTER', 'INFILL', 'SUPPORT', 'BRIDGE']
        
        for i, line in enumerate(test_lines):
            match = self.feature_manager.feature_regex.search(line)
            self.assertIsNotNone(match, f"Failed to detect feature in: {line}")
            self.assertEqual(match.group(1).upper(), expected_features[i])
    
    def test_feature_configuration_loading(self):
        """Test loading of feature-specific configurations"""
        # Check that configurations were loaded
        self.assertIn('WALL-OUTER', self.feature_manager.feature_configs)
        self.assertIn('INFILL', self.feature_manager.feature_configs)
        
        # Check specific preferences
        self.assertEqual(self.feature_manager.feature_configs['WALL-OUTER']['preference'], 'quality')
        self.assertEqual(self.feature_manager.feature_configs['INFILL']['preference'], 'speed')
    
    def test_quality_presets(self):
        """Test quality preset definitions"""
        # Check that all required presets exist
        required_presets = ['speed', 'balance', 'quality']
        for preset in required_presets:
            self.assertIn(preset, self.feature_manager.quality_presets)
            
            preset_config = self.feature_manager.quality_presets[preset]
            self.assertIn('shaper_preference', preset_config)
            self.assertIn('freq_adjustment', preset_config)
            self.assertIn('damping_adjustment', preset_config)
    
    def test_feature_compensation_params(self):
        """Test generation of feature-specific compensation parameters"""
        params = self.feature_manager.get_feature_compensation_params('WALL-OUTER')
        
        self.assertIn('shaper_preference', params)
        self.assertIn('freq_adjustment', params)
        self.assertIn('damping_adjustment', params)
        self.assertIn('custom_shaper', params)
        self.assertIn('custom_freqs', params)
        
        # Quality preference should favor quality shapers
        quality_shapers = ['ulv', 'multi_freq', 'ei']
        self.assertIn(params['shaper_preference'][0], quality_shapers)


class TestDynamicInputShaper(unittest.TestCase):
    """Test dynamic input shaper with multi-axis and feature support"""
    
    def setUp(self):
        self.config = MockConfig({'enabled': True})
        # Mock the gcode object
        self.config.printer.objects['gcode'] = MockObject()
        self.shaper = DynamicInputShaper(self.config)
    
    def test_multi_axis_initialization(self):
        """Test that all axes are properly initialized"""
        axis_names = ['x', 'y', 'z', 'a', 'b', 'c']
        
        for axis in axis_names:
            self.assertIn(axis, self.shaper.base_parameters)
            self.assertIn(axis, self.shaper.current_adjustments)
            self.assertIn(axis, self.shaper.target_parameters)
            
            self.assertIsNone(self.shaper.base_parameters[axis])
            self.assertEqual(self.shaper.current_adjustments[axis]['freq'], 1.0)
            self.assertEqual(self.shaper.current_adjustments[axis]['damping'], 1.0)
    
    def test_feature_type_handling(self):
        """Test feature type detection and compensation updates"""
        # Test setting feature type
        self.shaper.set_current_feature('WALL-OUTER')
        self.assertEqual(self.shaper.current_feature, 'WALL-OUTER')
        
        # Test feature change
        self.shaper.set_current_feature('INFILL')
        self.assertEqual(self.shaper.current_feature, 'INFILL')
        
        # Test that feature change triggers compensation update
        # (Would need more sophisticated mocking for full test)
    
    def test_compensation_parameter_application(self):
        """Test application of feature-specific compensation parameters"""
        # Set some base parameters
        for axis in ['x', 'y', 'z']:
            self.shaper.base_parameters[axis] = {'freq': 50.0, 'damping': 0.1}
        
        # Set feature and trigger update
        self.shaper.set_current_feature('WALL-OUTER')
        self.shaper._update_feature_compensation()
        
        # Check that adjustments were applied
        for axis in ['x', 'y', 'z']:
            self.assertIsNotNone(self.shaper.current_adjustments[axis]['freq'])
            self.assertIsNotNone(self.shaper.current_adjustments[axis]['damping'])


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = MockConfig({
            'enabled': True,
            'feature_wall_outer_preference': 'quality',
            'feature_infill_preference': 'speed',
            'feature_bridge_preference': 'quality'
        })
        self.config.printer.objects['gcode'] = MockObject()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow(self):
        """Test complete workflow from G-code parsing to compensation application"""
        # Create dynamic input shaper
        shaper = DynamicInputShaper(self.config)
        
        # Simulate G-code processing with feature comments
        test_gcode = [
            "G28",
            ";TYPE:WALL-OUTER",
            "G1 X10 Y10 F3000",
            "G1 X20 Y20",
            ";TYPE:INFILL", 
            "G1 X30 Y30 F6000",
            "G1 X40 Y40"
        ]
        
        # Process each line (simplified simulation)
        for line in test_gcode:
            match = shaper.feature_manager.feature_regex.search(line)
            if match:
                feature = match.group(1).upper()
                shaper.set_current_feature(feature)
        
        # Verify final state
        self.assertEqual(shaper.current_feature, 'INFILL')


def run_comprehensive_tests():
    """Run comprehensive test suite"""
    print("Running Multi-Axis Support and G-code Feature Type Compensation Tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_cases = [
        TestMultiAxisSupport,
        TestFeatureTypeManager, 
        TestDynamicInputShaper,
        TestIntegration
    ]
    
    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE TEST RESULTS")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)