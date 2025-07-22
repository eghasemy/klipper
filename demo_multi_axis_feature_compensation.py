#!/usr/bin/env python3

"""
Demo: Multi-Axis Support and G-code Feature Type Compensation

This demo showcases the enhanced resonance compensation system with:
1. Multi-axis support (X, Y, Z, A, B, C)
2. G-code feature type detection and compensation
3. Quality/balance/speed presets for each feature type
"""

import os, sys, tempfile, time
import numpy as np

# Add the klippy directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'klippy'))

from klippy.extras.dynamic_input_shaper import DynamicInputShaper, FeatureTypeManager, MotionAnalyzer


class DemoEnvironment:
    """Demo environment with simulated printer"""
    
    def __init__(self):
        self.printer = self._create_mock_printer()
        self.gcode_handler = self._create_mock_gcode()
        self.printer.objects['gcode'] = self.gcode_handler
        
    def _create_mock_printer(self):
        """Create a mock printer for demo"""
        class MockPrinter:
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
            def __init__(self):
                self.commands = {}
                
            def register_command(self, cmd, handler, desc=None):
                self.commands[cmd] = handler
                
            def get_position(self):
                return [120, 100, 50, 0, 15, -5]  # 6-axis position
        
        return MockPrinter()
    
    def _create_mock_gcode(self):
        """Create a mock G-code handler"""
        class MockGCode:
            def __init__(self):
                self.commands = {}
                
            def register_command(self, cmd, handler, desc=None):
                self.commands[cmd] = handler
        
        return MockGCode()
    
    def create_config(self, values=None):
        """Create a configuration object"""
        class MockConfig:
            def __init__(self, printer, values=None):
                self.printer = printer
                self.values = values or {}
            
            def get_printer(self):
                return self.printer
            
            def get_name(self):
                return "dynamic_input_shaper demo"
            
            def getboolean(self, key, default=None):
                return self.values.get(key, default)
            
            def getfloat(self, key, default=None, minval=None, maxval=None, above=None, below=None):
                return self.values.get(key, default)
            
            def getint(self, key, default=None, minval=None, maxval=None):
                return self.values.get(key, default)
            
            def get(self, key, default=None):
                return self.values.get(key, default)
        
        return MockConfig(self.printer, values)


def demo_multi_axis_support():
    """Demonstrate multi-axis motion analysis and compensation"""
    print("\n" + "="*60)
    print("DEMO 1: Multi-Axis Support")
    print("="*60)
    
    env = DemoEnvironment()
    config = env.create_config({'enabled': True})
    
    print("Creating motion analyzer with multi-axis support...")
    motion_analyzer = MotionAnalyzer(config)
    
    print("\nTesting 6-axis motion analysis:")
    print("Axes: X, Y, Z, A, B, C")
    
    # Simulate multi-axis motion
    motion_data = [
        # pos_x, pos_y, pos_z, pos_a, pos_b, pos_c
        ([100, 100, 50, 0, 0, 0], [20, 15, 5, 2, 1, 0], [2000, 1500, 500, 200, 100, 50]),
        ([102, 101, 50, 1, 0, 0], [22, 16, 5, 2, 1, 0], [2100, 1600, 500, 200, 100, 50]),
        ([105, 103, 51, 2, 1, 0], [25, 18, 6, 3, 1, 0], [2200, 1700, 600, 250, 110, 50]),
        ([109, 106, 52, 3, 2, 1], [28, 20, 7, 3, 2, 1], [2300, 1800, 700, 300, 120, 60]),
        ([114, 110, 53, 4, 3, 2], [30, 22, 8, 4, 2, 1], [2400, 1900, 800, 350, 130, 70]),
    ]
    
    for i, (position, velocity, acceleration) in enumerate(motion_data):
        timestamp = time.time() + i * 0.1
        
        print(f"\nMotion frame {i+1}:")
        print(f"  Position: {[f'{p:.1f}' for p in position]}")
        print(f"  Velocity: {[f'{v:.1f}' for v in velocity]}")
        print(f"  Acceleration: {[f'{a:.0f}' for a in acceleration]}")
        
        result = motion_analyzer.analyze_motion(position, velocity, acceleration, timestamp)
        
        if result:
            print(f"  Recommendations generated for {len(result)} axes:")
            for axis, rec in result.items():
                if any(rec.values()):  # Only show if there are actual recommendations
                    print(f"    {axis.upper()}: freq×{rec['freq_adjustment']:.2f}, "
                          f"damping×{rec['damping_adjustment']:.2f}, "
                          f"shaper: {rec['shaper_hint'] or 'default'}")
        else:
            print(f"  Building motion history... ({len(motion_analyzer.motion_history)} frames)")
    
    print(f"\nMotion analysis complete. Pattern detected: {motion_analyzer._detect_movement_pattern()}")


def demo_feature_type_detection():
    """Demonstrate G-code feature type detection and configuration"""
    print("\n" + "="*60)
    print("DEMO 2: G-code Feature Type Detection")
    print("="*60)
    
    env = DemoEnvironment()
    config = env.create_config({
        'enabled': True,
        'feature_wall_outer_preference': 'quality',
        'feature_infill_preference': 'speed',
        'feature_support_preference': 'speed',
        'feature_bridge_preference': 'quality',
        'feature_top_surface_preference': 'quality'
    })
    
    print("Creating feature type manager...")
    feature_manager = FeatureTypeManager(config)
    
    print("\nConfigured feature preferences:")
    for feature, config_data in feature_manager.feature_configs.items():
        print(f"  {feature}: {config_data['preference']}")
    
    print("\nTesting G-code feature type detection:")
    
    # Sample G-code with feature type comments
    sample_gcode = [
        "G28 ; home all axes",
        ";TYPE:WALL-OUTER",
        "G1 X10 Y10 F3000",
        "G1 X20 Y20 E0.5",
        ";TYPE:INFILL",
        "G1 X30 Y30 F6000",
        "G1 X40 Y40 E1.0",
        ";TYPE:BRIDGE",
        "G1 X50 Y50 F2000",
        "G1 X60 Y60 E1.5",
        ";TYPE:SUPPORT",
        "G1 X70 Y70 F4000",
        "G1 X80 Y80 E2.0"
    ]
    
    current_feature = None
    for line in sample_gcode:
        print(f"\nProcessing: {line}")
        
        # Check for feature type comment
        match = feature_manager.feature_regex.search(line)
        if match:
            new_feature = match.group(1).upper()
            if new_feature != current_feature:
                current_feature = new_feature
                print(f"  → Feature changed to: {current_feature}")
                
                # Get compensation parameters for this feature
                params = feature_manager.get_feature_compensation_params(current_feature)
                print(f"  → Compensation settings:")
                print(f"    Preferred shapers: {params['shaper_preference']}")
                print(f"    Frequency adjustment: ×{params['freq_adjustment']}")
                print(f"    Damping adjustment: ×{params['damping_adjustment']}")
        else:
            print(f"  → G-code command (feature: {current_feature or 'none'})")


def demo_quality_presets():
    """Demonstrate quality presets for different feature types"""
    print("\n" + "="*60)
    print("DEMO 3: Quality Presets")
    print("="*60)
    
    env = DemoEnvironment()
    config = env.create_config({'enabled': True})
    feature_manager = FeatureTypeManager(config)
    
    print("Available quality presets:")
    for preset_name, preset_config in feature_manager.quality_presets.items():
        print(f"\n{preset_name.upper()} preset:")
        print(f"  Preferred shapers: {preset_config['shaper_preference']}")
        print(f"  Frequency adjustment: ×{preset_config['freq_adjustment']}")
        print(f"  Damping adjustment: ×{preset_config['damping_adjustment']}")
        
        if preset_name == 'speed':
            print("  → Optimized for high-speed printing with minimal smoothing")
        elif preset_name == 'balance':
            print("  → Balanced approach between speed and quality")
        elif preset_name == 'quality':
            print("  → Maximum vibration reduction for best surface finish")
    
    print("\nFeature type recommendations:")
    feature_examples = {
        'WALL-OUTER': 'quality',  # External surfaces need best finish
        'WALL-INNER': 'balance',  # Internal walls can be faster
        'INFILL': 'speed',        # Infill can be printed fast
        'SUPPORT': 'speed',       # Support structures don't need high quality
        'BRIDGE': 'quality',      # Bridges need careful printing
        'TOP-SURFACE': 'quality'  # Top surfaces need good finish
    }
    
    for feature, recommended in feature_examples.items():
        params = feature_manager.get_feature_compensation_params(feature)
        print(f"\n{feature}:")
        print(f"  Recommended preset: {recommended}")
        print(f"  Applied shapers: {params['shaper_preference']}")
        print(f"  Frequency adjustment: ×{params['freq_adjustment']}")


def demo_dynamic_compensation():
    """Demonstrate dynamic compensation with feature types"""
    print("\n" + "="*60)
    print("DEMO 4: Dynamic Compensation Integration")
    print("="*60)
    
    env = DemoEnvironment()
    config = env.create_config({
        'enabled': True,
        'feature_wall_outer_preference': 'quality',
        'feature_infill_preference': 'speed'
    })
    
    print("Creating dynamic input shaper with feature support...")
    shaper = DynamicInputShaper(config)
    
    # Simulate base parameters for multi-axis
    base_params = {
        'x': {'freq': 52.3, 'damping': 0.08},
        'y': {'freq': 48.7, 'damping': 0.09},
        'z': {'freq': 35.2, 'damping': 0.12},
        'a': {'freq': 25.8, 'damping': 0.15},
        'b': {'freq': 22.1, 'damping': 0.18},
        'c': {'freq': 18.9, 'damping': 0.20}
    }
    
    print("\nSetting base calibration parameters:")
    for axis, params in base_params.items():
        shaper.base_parameters[axis] = params
        print(f"  {axis.upper()}-axis: {params['freq']:.1f} Hz, damping {params['damping']:.3f}")
    
    print("\nSimulating feature type changes during printing:")
    
    # Simulate different feature types
    feature_sequence = [
        ('WALL-OUTER', "Starting outer wall - quality mode"),
        ('WALL-INNER', "Switching to inner wall - balanced mode"), 
        ('INFILL', "Starting infill - speed mode"),
        ('BRIDGE', "Printing bridge - quality mode"),
        ('SUPPORT', "Printing supports - speed mode")
    ]
    
    for feature, description in feature_sequence:
        print(f"\n{description}")
        print(f"Feature type: {feature}")
        
        # Apply feature change
        shaper.set_current_feature(feature)
        shaper._update_feature_compensation()
        
        print("Applied compensation adjustments:")
        for axis in ['x', 'y', 'z']:
            if shaper.base_parameters[axis]:
                base_freq = shaper.base_parameters[axis]['freq']
                adjustment = shaper.current_adjustments[axis]
                new_freq = base_freq * adjustment['freq']
                
                print(f"  {axis.upper()}-axis: {base_freq:.1f}Hz → {new_freq:.1f}Hz "
                      f"(×{adjustment['freq']:.2f}), damping ×{adjustment['damping']:.2f}")


def demo_configuration_examples():
    """Show configuration examples for users"""
    print("\n" + "="*60)
    print("DEMO 5: Configuration Examples")
    print("="*60)
    
    print("Example configuration for printer.cfg:")
    print()
    
    config_example = """
[dynamic_input_shaper]
enabled: True
adaptation_rate: 0.1
min_update_interval: 0.5

# Feature type preferences (speed/balance/quality)
feature_wall_outer_preference: quality
feature_wall_inner_preference: balance
feature_infill_preference: speed
feature_support_preference: speed
feature_bridge_preference: quality
feature_top_surface_preference: quality
feature_bottom_surface_preference: balance

# Custom shaper overrides (optional)
feature_wall_outer_shaper: ulv
feature_bridge_shaper: multi_freq

# Custom frequency overrides (optional)
feature_wall_outer_freq_x: 45.2
feature_wall_outer_freq_y: 41.8
feature_infill_freq_x: 58.5
feature_infill_freq_y: 55.2
"""
    
    print(config_example)
    
    print("\nG-code commands for runtime control:")
    print()
    
    commands_example = """
# Enable dynamic shaping
ENABLE_DYNAMIC_SHAPING ENABLE=1

# Configure feature-specific compensation
SET_FEATURE_COMPENSATION FEATURE=WALL-OUTER PREFERENCE=quality
SET_FEATURE_COMPENSATION FEATURE=INFILL PREFERENCE=speed CUSTOM_FREQ_X=55.0

# Check current status
GET_DYNAMIC_SHAPER_STATUS

# Multi-axis calibration
ADAPTIVE_RESONANCE_CALIBRATE DENSITY=medium
"""
    
    print(commands_example)


def main():
    """Run all demos"""
    print("Multi-Axis Support and G-code Feature Type Compensation Demo")
    print("=" * 70)
    print("This demo showcases the enhanced resonance compensation system")
    print("with multi-axis support and intelligent G-code feature detection.")
    
    try:
        demo_multi_axis_support()
        demo_feature_type_detection()
        demo_quality_presets()
        demo_dynamic_compensation()
        demo_configuration_examples()
        
        print("\n" + "="*70)
        print("DEMO COMPLETE")
        print("="*70)
        print("Key improvements demonstrated:")
        print("✓ Multi-axis support (X, Y, Z, A, B, C)")
        print("✓ G-code feature type detection (;TYPE: comments)")
        print("✓ Quality/balance/speed presets for each feature")
        print("✓ Dynamic compensation parameter adjustment")
        print("✓ Comprehensive configuration options")
        print("\nThe system provides intelligent, feature-aware compensation")
        print("that automatically optimizes settings based on what's being printed.")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)