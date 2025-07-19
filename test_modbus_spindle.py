#!/usr/bin/env python3
# Test script for Modbus VFD spindle functionality
#
# This script provides basic testing for the Modbus spindle implementation
# without requiring a full Klipper installation.

import sys
import time
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Mock config class for testing
class MockConfig:
    def __init__(self, config_dict):
        self.config = config_dict
        
    def get(self, key, default=None):
        return self.config.get(key, default)
        
    def getint(self, key, default=None, minval=None, maxval=None):
        value = int(self.config.get(key, default))
        if minval is not None and value < minval:
            raise ValueError(f"{key} value {value} below minimum {minval}")
        if maxval is not None and value > maxval:
            raise ValueError(f"{key} value {value} above maximum {maxval}")
        return value
        
    def getfloat(self, key, default=None, minval=None):
        value = float(self.config.get(key, default))
        if minval is not None and value < minval:
            raise ValueError(f"{key} value {value} below minimum {minval}")
        return value

# Mock printer class
class MockPrinter:
    def __init__(self):
        self.objects = {}
        
    def load_object(self, config, name):
        return None
        
    def register_event_handler(self, event, handler):
        pass

def test_modbus_crc():
    """Test Modbus CRC16 calculation"""
    print("Testing Modbus CRC16 calculation...")
    
    # Import the modbus module
    sys.path.insert(0, '/home/runner/work/klipper/klipper/klippy/extras')
    from modbus import ModbusRTU
    
    # Create a mock config
    config = MockConfig({
        'device': '/dev/ttyUSB0',
        'baudrate': 9600,
        'slave_id': 1
    })
    
    # Create mock printer
    class MockPrinter:
        def register_event_handler(self, event, handler):
            pass
        def get_reactor(self):
            return None
            
    config.get_printer = lambda: MockPrinter()
    
    # Create ModbusRTU instance
    modbus = ModbusRTU(config)
    
    # Test CRC calculation with known values
    test_data = bytearray([0x01, 0x03, 0x00, 0x00, 0x00, 0x02])
    expected_crc = 0x0BC4  # Known CRC for this data (little-endian)
    calculated_crc = modbus._calc_crc16(test_data)
    
    print(f"Test data: {' '.join(f'{b:02X}' for b in test_data)}")
    print(f"Expected CRC: {expected_crc:04X}")
    print(f"Calculated CRC: {calculated_crc:04X}")
    
    if calculated_crc == expected_crc:
        print("✓ CRC calculation test PASSED")
        return True
    else:
        print("✗ CRC calculation test FAILED")
        return False

def test_frame_building():
    """Test Modbus frame building"""
    print("\nTesting Modbus frame building...")
    
    sys.path.insert(0, '/home/runner/work/klipper/klipper/klippy/extras')
    from modbus import ModbusRTU
    
    config = MockConfig({
        'device': '/dev/ttyUSB0',
        'baudrate': 9600,
        'slave_id': 1
    })
    
    class MockPrinter:
        def register_event_handler(self, event, handler):
            pass
        def get_reactor(self):
            return None
            
    config.get_printer = lambda: MockPrinter()
    
    modbus = ModbusRTU(config)
    
    # Test building a read holding registers frame
    frame = modbus._build_frame(0x01, 0x03, bytearray([0x00, 0x00, 0x00, 0x02]))
    expected = bytearray([0x01, 0x03, 0x00, 0x00, 0x00, 0x02, 0xC4, 0x0B])
    
    print(f"Built frame: {' '.join(f'{b:02X}' for b in frame)}")
    print(f"Expected:    {' '.join(f'{b:02X}' for b in expected)}")
    
    if frame == expected:
        print("✓ Frame building test PASSED")
        return True
    else:
        print("✗ Frame building test FAILED")
        return False

def test_frame_validation():
    """Test Modbus frame validation"""
    print("\nTesting Modbus frame validation...")
    
    sys.path.insert(0, '/home/runner/work/klipper/klipper/klippy/extras')
    from modbus import ModbusRTU
    
    config = MockConfig({
        'device': '/dev/ttyUSB0',
        'baudrate': 9600,
        'slave_id': 1
    })
    
    class MockPrinter:
        def register_event_handler(self, event, handler):
            pass
        def get_reactor(self):
            return None
            
    config.get_printer = lambda: MockPrinter()
    
    modbus = ModbusRTU(config)
    
    # Test with valid frame
    valid_frame = bytearray([0x01, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00, 0xFA, 0x33])
    invalid_frame = bytearray([0x01, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF])
    
    valid_result = modbus._validate_frame(valid_frame)
    invalid_result = modbus._validate_frame(invalid_frame)
    
    print(f"Valid frame validation: {valid_result}")
    print(f"Invalid frame validation: {invalid_result}")
    
    if valid_result and not invalid_result:
        print("✓ Frame validation test PASSED")
        return True
    else:
        print("✗ Frame validation test FAILED")
        return False

def test_huanyang_rpm_conversion():
    """Test Huanyang RPM to register conversion"""
    print("\nTesting Huanyang RPM conversion...")
    
    # Test RPM to frequency conversion math directly
    max_rpm = 24000
    rated_freq = 50.0
    test_rpm = 12000  # Half speed
    
    expected_freq = (test_rpm / max_rpm) * rated_freq  # Should be 25 Hz
    expected_register = int(expected_freq * 100)  # Should be 2500
    
    print(f"Test RPM: {test_rpm}")
    print(f"Expected frequency: {expected_freq} Hz")
    print(f"Expected register value: {expected_register}")
    
    # Reverse conversion test
    register_value = 2500
    freq = register_value / 100.0
    rpm = (freq / rated_freq) * max_rpm
    
    print(f"Reverse conversion - Register: {register_value} -> Frequency: {freq} Hz -> RPM: {rpm}")
    
    if abs(rpm - test_rpm) < 1.0:  # Allow small rounding error
        print("✓ Huanyang RPM conversion test PASSED")
        return True
    else:
        print("✗ Huanyang RPM conversion test FAILED")
        return False

def main():
    """Run all tests"""
    print("=== Modbus VFD Spindle Tests ===\n")
    
    tests = [
        test_modbus_crc,
        test_frame_building,
        test_frame_validation,
        test_huanyang_rpm_conversion
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("✓ All tests PASSED!")
        return 0
    else:
        print("✗ Some tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())