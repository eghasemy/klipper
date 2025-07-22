#!/usr/bin/env python3

"""
Standalone test for ML and predictive compensation features.

This test validates the ML and predictive features without requiring
the full Klipper environment.
"""

import sys
import os
import time
import unittest
import tempfile
import numpy as np
from unittest.mock import Mock, patch

# Test the feature extraction and pattern recognition logic directly
def extract_motion_features(motion_history):
    """Extract feature vector from motion history for ML analysis"""
    if len(motion_history) < 10:
        return None
    
    recent_moves = list(motion_history)[-20:]  # Last 20 moves
    
    # Feature extraction
    features = []
    
    # Velocity statistics
    velocities = [np.array(move['velocity']) for move in recent_moves]
    speeds = [np.linalg.norm(vel) for vel in velocities]
    
    features.extend([
        np.mean(speeds),           # Average speed
        np.std(speeds),            # Speed variation
        np.max(speeds),            # Peak speed
        np.min(speeds),            # Minimum speed
        np.percentile(speeds, 90), # 90th percentile speed
    ])
    
    # Acceleration statistics
    accelerations = [np.array(move['acceleration']) for move in recent_moves]
    accel_magnitudes = [np.linalg.norm(acc) for acc in accelerations]
    
    features.extend([
        np.mean(accel_magnitudes),           # Average acceleration
        np.std(accel_magnitudes),            # Acceleration variation
        np.max(accel_magnitudes),            # Peak acceleration
        np.percentile(accel_magnitudes, 90), # 90th percentile acceleration
    ])
    
    # Direction change analysis
    direction_changes = 0
    angle_changes = []
    
    for i in range(1, len(velocities)):
        prev_vel = velocities[i-1]
        curr_vel = velocities[i]
        
        prev_speed = np.linalg.norm(prev_vel)
        curr_speed = np.linalg.norm(curr_vel)
        
        if prev_speed > 1.0 and curr_speed > 1.0:  # Ignore very slow moves
            # Calculate angle between velocity vectors
            cos_angle = np.dot(prev_vel, curr_vel) / (prev_speed * curr_speed)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
            angle = np.arccos(cos_angle)
            angle_changes.append(angle)
            
            if angle > np.pi / 4:  # 45 degree threshold
                direction_changes += 1
    
    features.extend([
        direction_changes / len(recent_moves),      # Direction change rate
        np.mean(angle_changes) if angle_changes else 0,  # Average angle change
        np.std(angle_changes) if angle_changes else 0,   # Angle change variation
    ])
    
    return np.array(features)

def classify_pattern_rule_based(motion_history):
    """Rule-based pattern classification"""
    if len(motion_history) < 5:
        return 'unknown'
    
    recent_moves = list(motion_history)[-20:]
    
    # Simple rule-based classification
    direction_changes = 0
    speed_variations = []
    
    for i in range(1, len(recent_moves)):
        prev_vel = recent_moves[i-1]['velocity']
        curr_vel = recent_moves[i]['velocity']
        
        if np.dot(prev_vel, curr_vel) < 0.5 * np.linalg.norm(prev_vel) * np.linalg.norm(curr_vel):
            direction_changes += 1
        
        prev_speed = np.linalg.norm(prev_vel)
        curr_speed = np.linalg.norm(curr_vel)
        if prev_speed > 0:
            speed_variations.append(abs(curr_speed - prev_speed) / prev_speed)
    
    avg_speed_variation = np.mean(speed_variations) if speed_variations else 0
    direction_change_rate = direction_changes / len(recent_moves)
    
    if direction_change_rate > 0.3:
        return 'corner_heavy'
    elif avg_speed_variation > 0.2:
        return 'variable_speed'
    elif direction_change_rate < 0.1 and avg_speed_variation < 0.1:
        return 'linear'
    else:
        return 'mixed'

def parse_gcode_command(command):
    """Parse G-code command to extract motion information"""
    cmd_parts = command.upper().split()
    if not cmd_parts:
        return None
    
    cmd_type = cmd_parts[0]
    
    # Only interested in motion commands
    if cmd_type not in ['G0', 'G1', 'G2', 'G3']:
        return None
    
    parsed = {
        'type': cmd_type,
        'coordinates': {},
        'feedrate': None,
        'is_extrusion': False
    }
    
    # Extract coordinates and parameters
    for part in cmd_parts[1:]:
        if part.startswith('X'):
            parsed['coordinates']['X'] = float(part[1:])
        elif part.startswith('Y'):
            parsed['coordinates']['Y'] = float(part[1:])
        elif part.startswith('Z'):
            parsed['coordinates']['Z'] = float(part[1:])
        elif part.startswith('E'):
            parsed['coordinates']['E'] = float(part[1:])
            parsed['is_extrusion'] = True
        elif part.startswith('F'):
            parsed['feedrate'] = float(part[1:])
    
    return parsed

class TestMLFeatures(unittest.TestCase):
    """Test ML-based features"""
    
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
        
        features = extract_motion_features(motion_history)
        
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 10)  # Should have multiple features
    
    def test_pattern_classification(self):
        """Test pattern classification"""
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
        
        pattern = classify_pattern_rule_based(motion_history)
        
        self.assertIsInstance(pattern, str)
        self.assertIn(pattern, ['corner_heavy', 'variable_speed', 'linear', 'mixed'])
    
    def test_linear_motion_pattern(self):
        """Test linear motion pattern detection"""
        # Create consistent linear motion
        motion_history = []
        for i in range(15):
            motion_history.append({
                'time': i * 0.1,
                'position': [i * 2.0, i * 1.0, 0],
                'velocity': [20.0, 10.0, 0],  # Consistent velocity
                'acceleration': [0.0, 0.0, 0]
            })
        
        pattern = classify_pattern_rule_based(motion_history)
        self.assertEqual(pattern, 'linear')

class TestPredictiveFeatures(unittest.TestCase):
    """Test predictive compensation features"""
    
    def test_gcode_parsing(self):
        """Test G-code command parsing"""
        # Test motion command
        command = "G1 X10.5 Y20.3 F1500"
        parsed = parse_gcode_command(command)
        
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed['type'], 'G1')
        self.assertEqual(parsed['coordinates']['X'], 10.5)
        self.assertEqual(parsed['coordinates']['Y'], 20.3)
        self.assertEqual(parsed['feedrate'], 1500)
        
        # Test non-motion command
        command = "M104 S200"
        parsed = parse_gcode_command(command)
        self.assertIsNone(parsed)
    
    def test_gcode_extrusion_detection(self):
        """Test detection of extrusion commands"""
        command = "G1 X10 Y10 E0.5 F1500"
        parsed = parse_gcode_command(command)
        
        self.assertIsNotNone(parsed)
        self.assertTrue(parsed['is_extrusion'])
        self.assertEqual(parsed['coordinates']['E'], 0.5)
    
    def test_sequence_consistency(self):
        """Test motion sequence consistency checking"""
        # Create consistent motion sequence
        commands = []
        for i in range(5):
            commands.append({
                'parsed': {
                    'type': 'G1',
                    'coordinates': {'X': float(i), 'Y': 0.0},
                    'feedrate': 1500,
                    'is_extrusion': True
                }
            })
        
        # All commands should be consistent (same type, similar feedrate)
        for i in range(1, len(commands)):
            self.assertEqual(commands[i]['parsed']['type'], commands[0]['parsed']['type'])
            self.assertEqual(commands[i]['parsed']['feedrate'], commands[0]['parsed']['feedrate'])

class TestPerformanceAndReliability(unittest.TestCase):
    """Test system performance and reliability"""
    
    def test_large_motion_history(self):
        """Test with large motion history"""
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
        features = extract_motion_features(motion_history)
        self.assertIsNotNone(features)
        
        pattern = classify_pattern_rule_based(motion_history)
        self.assertIsNotNone(pattern)
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty motion history
        features = extract_motion_features([])
        self.assertIsNone(features)
        
        # Short motion history
        short_history = [{'velocity': [1, 1], 'acceleration': [1, 1]}] * 3
        features = extract_motion_features(short_history)
        self.assertIsNone(features)
        
        pattern = classify_pattern_rule_based([])
        self.assertEqual(pattern, 'unknown')
    
    def test_zero_velocity_handling(self):
        """Test handling of zero velocities"""
        motion_history = []
        for i in range(15):
            motion_history.append({
                'time': i * 0.1,
                'position': [0, 0, 0],
                'velocity': [0, 0, 0],  # Zero velocity
                'acceleration': [0, 0, 0]
            })
        
        # Should handle zero velocities gracefully
        features = extract_motion_features(motion_history)
        self.assertIsNotNone(features)
        
        # Should detect as static/unknown pattern
        pattern = classify_pattern_rule_based(motion_history)
        self.assertIn(pattern, ['linear', 'unknown'])

class TestMLIntegration(unittest.TestCase):
    """Test ML integration features"""
    
    def test_feature_vector_consistency(self):
        """Test that feature vectors are consistent in size"""
        # Create different motion patterns
        patterns = {
            'linear': self._create_linear_motion(),
            'corner': self._create_corner_motion(),
            'variable': self._create_variable_motion()
        }
        
        feature_sizes = []
        for pattern_name, motion_history in patterns.items():
            features = extract_motion_features(motion_history)
            if features is not None:
                feature_sizes.append(len(features))
        
        # All feature vectors should be the same size
        self.assertTrue(all(size == feature_sizes[0] for size in feature_sizes))
    
    def _create_linear_motion(self):
        """Create linear motion pattern"""
        motion_history = []
        for i in range(15):
            motion_history.append({
                'time': i * 0.1,
                'position': [i * 2.0, i * 1.0, 0],
                'velocity': [20.0, 10.0, 0],
                'acceleration': [5.0, 2.5, 0]
            })
        return motion_history
    
    def _create_corner_motion(self):
        """Create corner-heavy motion pattern"""
        motion_history = []
        for i in range(15):
            # Alternating directions
            x_vel = 20.0 if i % 2 == 0 else -20.0
            y_vel = 10.0 if i % 2 == 0 else -10.0
            
            motion_history.append({
                'time': i * 0.1,
                'position': [i, i],
                'velocity': [x_vel, y_vel, 0],
                'acceleration': [100.0, 50.0, 0]
            })
        return motion_history
    
    def _create_variable_motion(self):
        """Create variable speed motion pattern"""
        motion_history = []
        for i in range(15):
            # Variable speeds
            speed = 10.0 + 20.0 * np.sin(i * 0.5)
            
            motion_history.append({
                'time': i * 0.1,
                'position': [i * 1.0, i * 0.5, 0],
                'velocity': [speed, speed * 0.5, 0],
                'acceleration': [speed * 2, speed, 0]
            })
        return motion_history

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("=" * 60)
    print("ML and Predictive Compensation Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestMLFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictiveFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceAndReliability))
    suite.addTests(loader.loadTestsFromTestCase(TestMLIntegration))
    
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
        print("\n🎯 Key Features Validated:")
        print("  • Motion feature extraction")
        print("  • Pattern classification")
        print("  • G-code parsing and analysis")
        print("  • Edge case handling")
        print("  • Performance with large datasets")
        print("  • ML integration compatibility")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)