#!/usr/bin/env python3

"""
Test suite for ML-based pattern recognition and predictive compensation features.

This test validates the new machine learning and predictive compensation
capabilities added to the dynamic input shaper system.
"""

import sys
import os
import tempfile
import unittest
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add the path to klippy modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'klippy'))

# Import the modules to test
from klippy.extras.dynamic_input_shaper import (
    MLPatternRecognizer, PredictiveCompensator, MotionAnalyzer, DynamicInputShaper
)

class TestMLPatternRecognizer(unittest.TestCase):
    """Test ML-based pattern recognition"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_printer = Mock()
        self.mock_config = Mock()
        
        # Set up config defaults
        self.mock_config.getboolean.return_value = True
        self.mock_config.get.return_value = '/tmp/test_model.pkl'
        self.mock_config.getint.return_value = 100
        self.mock_config.getfloat.return_value = 1.0
        
        # Create recognizer
        self.recognizer = MLPatternRecognizer(self.mock_config, self.mock_printer)
    
    def test_feature_extraction(self):
        """Test motion feature extraction"""
        # Create sample motion history
        motion_history = []
        for i in range(20):
            motion_history.append({
                'time': i * 0.1,
                'position': [i * 1.0, i * 0.5, 0],
                'velocity': [10.0, 5.0, 0],
                'acceleration': [100.0, 50.0, 0]
            })
        
        features = self.recognizer._extract_motion_features(motion_history)
        
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 15)  # Should have multiple features
    
    def test_fallback_pattern_detection(self):
        """Test fallback pattern detection when ML is unavailable"""
        # Disable ML for this test
        self.recognizer.enabled = False
        
        # Create motion history with corner-heavy pattern
        motion_history = []
        for i in range(10):
            # Alternating directions to simulate corners
            x_vel = 10.0 if i % 2 == 0 else -10.0
            y_vel = 5.0 if i % 2 == 0 else -5.0
            
            motion_history.append({
                'time': i * 0.1,
                'position': [i, i],
                'velocity': [x_vel, y_vel],
                'acceleration': [100.0, 50.0]
            })
        
        pattern = self.recognizer._fallback_pattern_detection(motion_history)
        
        self.assertIsInstance(pattern, str)
        self.assertIn(pattern, ['corner_heavy', 'variable_speed', 'linear', 'mixed'])
    
    def test_ground_truth_determination(self):
        """Test ground truth label determination from G-code context"""
        # Test with TYPE comment
        gcode_context = "; TYPE:WALL-OUTER"
        predicted = "linear"
        
        truth = self.recognizer._determine_ground_truth(gcode_context, predicted)
        self.assertEqual(truth, "perimeter")
        
        # Test without TYPE comment
        gcode_context = "; Some other comment"
        truth = self.recognizer._determine_ground_truth(gcode_context, predicted)
        self.assertEqual(truth, predicted)
    
    @patch('klippy.extras.dynamic_input_shaper.ML_AVAILABLE', True)
    def test_training_data_collection(self):
        """Test training data collection"""
        initial_count = len(self.recognizer.training_data['features'])
        
        # Create sample features
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gcode_context = "; TYPE:INFILL"
        predicted = "linear"
        
        self.recognizer._collect_training_sample(features, gcode_context, predicted)
        
        # Check that data was collected
        self.assertEqual(len(self.recognizer.training_data['features']), initial_count + 1)
        self.assertEqual(len(self.recognizer.training_data['labels']), initial_count + 1)
        
        # Check the collected data
        self.assertTrue(np.array_equal(self.recognizer.training_data['features'][-1], features))
        self.assertEqual(self.recognizer.training_data['labels'][-1], "infill")

class TestPredictiveCompensator(unittest.TestCase):
    """Test predictive compensation system"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_printer = Mock()
        self.mock_config = Mock()
        
        # Set up config defaults
        self.mock_config.getboolean.return_value = True
        self.mock_config.getfloat.return_value = 2.0
        self.mock_config.getint.return_value = 50
        
        # Mock G-code object
        self.mock_gcode = Mock()
        self.mock_printer.lookup_object.return_value = self.mock_gcode
        
        self.compensator = PredictiveCompensator(self.mock_config, self.mock_printer)
    
    def test_gcode_parsing(self):
        """Test G-code command parsing"""
        # Test motion command
        command = "G1 X10.5 Y20.3 F1500"
        parsed = self.compensator._parse_gcode_command(command)
        
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed['type'], 'G1')
        self.assertEqual(parsed['coordinates']['X'], 10.5)
        self.assertEqual(parsed['coordinates']['Y'], 20.3)
        self.assertEqual(parsed['feedrate'], 1500)
        
        # Test non-motion command
        command = "M104 S200"
        parsed = self.compensator._parse_gcode_command(command)
        self.assertIsNone(parsed)
    
    def test_sequence_identification(self):
        """Test motion sequence identification"""
        # Create sample commands
        commands = []
        for i in range(10):
            commands.append({
                'time': i * 0.1,
                'command': f"G1 X{i} Y{i} F1500",
                'parsed': {
                    'type': 'G1',
                    'coordinates': {'X': float(i), 'Y': float(i)},
                    'feedrate': 1500,
                    'is_extrusion': True
                }
            })
        
        sequences = self.compensator._identify_motion_sequences(commands)
        
        self.assertGreater(len(sequences), 0)
        # Should identify one main sequence
        self.assertGreaterEqual(len(sequences[0]), 2)
    
    def test_pattern_classification(self):
        """Test motion pattern classification"""
        # Create linear sequence
        sequence = []
        for i in range(5):
            sequence.append({
                'parsed': {
                    'type': 'G1',
                    'coordinates': {'X': float(i), 'Y': 0.0},
                    'feedrate': 1500
                }
            })
        
        pattern = self.compensator._classify_sequence_pattern(sequence)
        self.assertIsInstance(pattern, str)
        
        # Create corner-heavy sequence
        sequence = []
        for i in range(6):
            x = 10.0 if i % 2 == 0 else 0.0
            y = 0.0 if i % 2 == 0 else 10.0
            sequence.append({
                'parsed': {
                    'type': 'G1',
                    'coordinates': {'X': x, 'Y': y},
                    'feedrate': 1500
                }
            })
        
        pattern = self.compensator._classify_sequence_pattern(sequence)
        # Should detect as corner-heavy or detailed
        self.assertIn(pattern, ['corner_heavy', 'detailed', 'mixed'])
    
    def test_resonance_risk_estimation(self):
        """Test resonance risk estimation"""
        # Create high-risk sequence (high speed, direction changes)
        sequence = []
        for i in range(5):
            x = 10.0 if i % 2 == 0 else -10.0
            sequence.append({
                'parsed': {
                    'type': 'G1',
                    'coordinates': {'X': x, 'Y': 0.0},
                    'feedrate': 5000  # High feedrate
                }
            })
        
        risk = self.compensator._estimate_resonance_risk(sequence)
        self.assertIsInstance(risk, float)
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)
    
    def test_compensation_recommendation(self):
        """Test compensation parameter recommendation"""
        # Create test sequence
        sequence = [{
            'parsed': {
                'type': 'G1',
                'coordinates': {'X': 10.0, 'Y': 10.0},
                'feedrate': 3000
            }
        }] * 3
        
        compensation = self.compensator._recommend_sequence_compensation(sequence)
        
        self.assertIsInstance(compensation, dict)
        self.assertIn('freq_adjustment', compensation)
        self.assertIn('damping_adjustment', compensation)
        self.assertIn('priority', compensation)

class TestIntegratedSystem(unittest.TestCase):
    """Test integrated ML and predictive system"""
    
    def setUp(self):
        """Set up integrated test environment"""
        self.mock_printer = Mock()
        self.mock_config = Mock()
        
        # Set up config defaults
        self.mock_config.getboolean.return_value = True
        self.mock_config.get.return_value = '/tmp/test_model.pkl'
        self.mock_config.getint.return_value = 100
        self.mock_config.getfloat.return_value = 1.0
        self.mock_config.get_name.return_value = "dynamic_input_shaper"
        
        # Mock other objects
        self.mock_gcode = Mock()
        self.mock_printer.lookup_object.return_value = self.mock_gcode
        self.mock_printer.get_reactor.return_value = Mock()
        
        self.shaper = DynamicInputShaper(self.mock_config)
    
    def test_ml_integration(self):
        """Test ML recognizer integration"""
        self.assertIsNotNone(self.shaper.ml_recognizer)
        self.assertIsNotNone(self.shaper.motion_analyzer.ml_recognizer)
    
    def test_predictive_integration(self):
        """Test predictive compensator integration"""
        self.assertIsNotNone(self.shaper.predictive_compensator)
    
    def test_effectiveness_tracking(self):
        """Test compensation effectiveness tracking"""
        # Simulate motion analysis with pattern detection
        self.shaper.motion_analyzer.current_pattern = "test_pattern"
        self.shaper.motion_analyzer.pattern_confidence = 0.8
        
        recommendations = {
            'x': {'freq_adjustment': 1.1, 'damping_adjustment': 1.0},
            'y': {'freq_adjustment': 1.0, 'damping_adjustment': 1.1}
        }
        
        # Track effectiveness
        self.shaper._track_compensation_effectiveness(recommendations, time.time())
        
        # Check that data was recorded
        self.assertGreater(len(self.shaper.pattern_performance_history), 0)
    
    def test_gcode_commands(self):
        """Test new G-code commands"""
        # Mock G-code response
        mock_gcmd = Mock()
        mock_gcmd.get_int.return_value = 0
        mock_gcmd.get.return_value = "test_pattern"
        mock_gcmd.get_float.return_value = 0.8
        mock_gcmd.respond_info = Mock()
        
        # Test ML training command
        self.shaper.cmd_TRAIN_ML_MODEL(mock_gcmd)
        
        # Test prediction status command
        self.shaper.cmd_GET_PREDICTION_STATUS(mock_gcmd)
        
        # Test effectiveness update command
        self.shaper.cmd_UPDATE_PATTERN_EFFECTIVENESS(mock_gcmd)
        
        # Verify commands executed without errors
        self.assertTrue(mock_gcmd.respond_info.called)

class TestPerformanceAndReliability(unittest.TestCase):
    """Test system performance and reliability"""
    
    def test_large_motion_history(self):
        """Test with large motion history"""
        mock_config = Mock()
        mock_config.getboolean.return_value = True
        mock_config.get.return_value = '/tmp/test.pkl'
        mock_config.getint.return_value = 100
        mock_config.getfloat.return_value = 1.0
        
        recognizer = MLPatternRecognizer(mock_config, Mock())
        
        # Create large motion history
        motion_history = []
        for i in range(1000):
            motion_history.append({
                'time': i * 0.1,
                'position': [i * 0.1, i * 0.05, 0],
                'velocity': [1.0, 0.5, 0],
                'acceleration': [10.0, 5.0, 0]
            })
        
        # Should handle large history without issues
        features = recognizer._extract_motion_features(motion_history)
        self.assertIsNotNone(features)
    
    def test_error_handling(self):
        """Test error handling in critical paths"""
        mock_config = Mock()
        mock_config.getboolean.return_value = True
        mock_config.get.return_value = '/tmp/nonexistent/path/test.pkl'
        mock_config.getint.return_value = 100
        mock_config.getfloat.return_value = 1.0
        
        # Should handle file path errors gracefully
        recognizer = MLPatternRecognizer(mock_config, Mock())
        self.assertIsNotNone(recognizer)
        
        # Test with invalid motion data
        result = recognizer._extract_motion_features([])
        self.assertIsNone(result)
    
    def test_memory_usage(self):
        """Test memory usage with continuous operation"""
        mock_config = Mock()
        mock_config.getboolean.return_value = True
        mock_config.get.return_value = '/tmp/test.pkl'
        mock_config.getint.return_value = 100
        mock_config.getfloat.return_value = 1.0
        
        compensator = PredictiveCompensator(mock_config, Mock())
        
        # Simulate continuous G-code buffering
        for i in range(1000):
            compensator._buffer_gcode_command(f"G1 X{i} Y{i} F1500")
        
        # Buffer should be limited in size
        self.assertLessEqual(len(compensator.gcode_buffer), compensator.lookahead_commands)

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("=" * 60)
    print("ML and Predictive Compensation Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestMLPatternRecognizer))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictiveCompensator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegratedSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceAndReliability))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, trace in result.failures:
            print(f"- {test}: {trace}")
    
    if result.errors:
        print("\nERRORS:")
        for test, trace in result.errors:
            print(f"- {test}: {trace}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_test()
    
    if success:
        print("\n✅ All tests passed! ML and predictive features are working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)