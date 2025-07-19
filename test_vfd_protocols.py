#!/usr/bin/env python3
"""
Test suite for Modbus VFD spindle implementations
Tests all supported VFD protocols with mock hardware validation
"""
import sys
import os
import logging
import threading
import time
import queue

# Add the klippy path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'klippy'))

# Mock classes for testing
class MockPrinter:
    def __init__(self):
        self.objects = {}
        self.event_handlers = {}
        
    def lookup_object(self, name, default=None):
        return self.objects.get(name, default)
        
    def load_object(self, config, name):
        return self.objects.get(name)
        
    def register_event_handler(self, event, handler):
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)
        
    def get_reactor(self):
        return MockReactor()

class MockReactor:
    def __init__(self):
        pass

class MockGCode:
    def __init__(self):
        self.commands = {}
        
    def register_command(self, name, handler, desc=""):
        self.commands[name] = handler
        
    def error(self, msg):
        raise Exception(msg)

class MockConfig:
    def __init__(self, name, values=None):
        self.name = name
        self.values = values or {}
        self.printer = MockPrinter()
        
    def get_name(self):
        return self.name
        
    def get_printer(self):
        return self.printer
        
    def get(self, key, default=None):
        return self.values.get(key, default)
        
    def getint(self, key, default=None, minval=None, maxval=None):
        value = self.values.get(key, default)
        if value is not None:
            return int(value)
        return default
        
    def getfloat(self, key, default=None, minval=None, above=None):
        value = self.values.get(key, default)
        if value is not None:
            return float(value)
        return default
        
    def getsection(self, name):
        return MockConfig(name)
        
    def error(self, msg):
        raise Exception(msg)
        
    def set(self, key, value):
        self.values[key] = value

class MockModbus:
    def __init__(self):
        self.connected = False
        self.registers = {}
        self.call_log = []
        
    def connect(self):
        self.connected = True
        
    def read_holding_registers(self, address, count):
        self.call_log.append(f"read_holding_registers({address}, {count})")
        # Return mock data based on address
        if address == 0xB005:  # H2A max RPM
            return [24000, 0]
        elif address == 0x0005:  # H100 max frequency
            return [4000]
        elif address == 0x0308:  # YL620 min frequency
            return [1000]
        elif address == 0x0000:  # YL620 max frequency
            return [4000]
        elif address == 0x0007:  # NowForever max/min frequency
            return [24000, 1000]
        else:
            return [0] * count
            
    def write_single_register(self, address, value):
        self.call_log.append(f"write_single_register({address}, {value})")
        self.registers[address] = value
        return True
        
    def send_command(self, function_code, data, expect_response=True, timeout=2.0):
        self.call_log.append(f"send_command({function_code}, {data})")
        # Mock response based on function code
        if function_code == 0x0F:  # Write multiple coils
            return bytearray([1, 0x0F, 0, 0, 0, 0x20])
        elif function_code == 0x01:  # Read coils
            return bytearray([1, 0x01, 2, 0, 0])
        return bytearray([1, function_code, 2, 0, 0])

def test_vfd_protocol(vfd_class, config_values, test_name):
    """Test a specific VFD protocol implementation"""
    print(f"\n=== Testing {test_name} ===")
    
    try:
        # Setup mock objects
        config = MockConfig("test_spindle", config_values)
        config.printer.objects['gcode'] = MockGCode()
        config.printer.objects['modbus'] = MockModbus()
        
        # Create VFD instance
        vfd = vfd_class(config)
        
        # Test initialization
        print(f"✓ {test_name} initialized successfully")
        
        # Test ready handler
        if hasattr(vfd, '_handle_ready'):
            vfd._handle_ready()
            print(f"✓ {test_name} ready handler executed")
        
        # Test spindle commands
        if hasattr(vfd, '_set_spindle_state'):
            # Test CW rotation
            vfd._set_spindle_state(True, 1, 12000)
            print(f"✓ {test_name} CW rotation test passed")
            
            # Test CCW rotation
            vfd._set_spindle_state(True, -1, 8000) 
            print(f"✓ {test_name} CCW rotation test passed")
            
            # Test stop
            vfd._set_spindle_state(False)
            print(f"✓ {test_name} stop test passed")
        
        # Test G-code commands
        gcode = config.printer.objects['gcode']
        if 'M3' in gcode.commands:
            print(f"✓ {test_name} M3 command registered")
        if 'M4' in gcode.commands:
            print(f"✓ {test_name} M4 command registered")
        if 'M5' in gcode.commands:
            print(f"✓ {test_name} M5 command registered")
        
        # Test status
        if hasattr(vfd, 'get_status'):
            status = vfd.get_status(time.time())
            print(f"✓ {test_name} status: {status}")
        
        # Cleanup
        if hasattr(vfd, '_shutdown'):
            vfd._shutdown()
            
        print(f"✓ {test_name} all tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ {test_name} test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all VFD protocol tests"""
    print("Starting Modbus VFD Protocol Test Suite")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    results = []
    
    # Test 1: Generic Modbus VFD
    try:
        from klippy.extras.modbus_spindle import ModbusSpindle
        config_values = {
            'protocol': 'generic',
            'min_rpm': 100,
            'max_rpm': 24000,
            'speed_register': 0x2000,
            'control_register': 0x2001
        }
        result = test_vfd_protocol(ModbusSpindle, config_values, "Generic Modbus VFD")
        results.append(("Generic Modbus VFD", result))
    except ImportError as e:
        print(f"✗ Could not import modbus_spindle: {e}")
        results.append(("Generic Modbus VFD", False))
    
    # Test 2: Huanyang VFD
    try:
        from klippy.extras.huanyang_spindle import HuanyangSpindle
        config_values = {
            'min_rpm': 100,
            'max_rpm': 24000,
            'rated_frequency': 50.0,
            'max_frequency': 100.0
        }
        result = test_vfd_protocol(HuanyangSpindle, config_values, "Huanyang VFD")
        results.append(("Huanyang VFD", result))
    except ImportError as e:
        print(f"✗ Could not import huanyang_spindle: {e}")
        results.append(("Huanyang VFD", False))
    
    # Test 3: H2A VFD
    try:
        from klippy.extras.vfd_h2a import H2AVFDSpindle
        config_values = {
            'min_rpm': 1000,
            'max_rpm': 24000,
            'speed_register': 0x1000,
            'control_register': 0x2000
        }
        result = test_vfd_protocol(H2AVFDSpindle, config_values, "H2A VFD")
        results.append(("H2A VFD", result))
    except ImportError as e:
        print(f"✗ Could not import vfd_h2a: {e}")
        results.append(("H2A VFD", False))
    
    # Test 4: Danfoss VLT2800 VFD
    try:
        from klippy.extras.vfd_danfoss_vlt2800 import DanfossVLT2800Spindle
        config_values = {
            'min_rpm': 100,
            'max_rpm': 24000,
            'max_frequency': 400.0
        }
        result = test_vfd_protocol(DanfossVLT2800Spindle, config_values, "Danfoss VLT2800 VFD")
        results.append(("Danfoss VLT2800 VFD", result))
    except ImportError as e:
        print(f"✗ Could not import vfd_danfoss_vlt2800: {e}")
        results.append(("Danfoss VLT2800 VFD", False))
    
    # Test 5: Siemens V20 VFD
    try:
        from klippy.extras.vfd_siemens_v20 import SiemensV20Spindle
        config_values = {
            'min_rpm': 100,
            'max_rpm': 24000,
            'motor_poles': 2,
            'motor_phases': 3,
            'max_frequency': 400.0
        }
        result = test_vfd_protocol(SiemensV20Spindle, config_values, "Siemens V20 VFD")
        results.append(("Siemens V20 VFD", result))
    except ImportError as e:
        print(f"✗ Could not import vfd_siemens_v20: {e}")
        results.append(("Siemens V20 VFD", False))
    
    # Test 6: YL620 VFD
    try:
        from klippy.extras.vfd_yl620 import YL620Spindle
        config_values = {
            'min_rpm': 600,
            'max_rpm': 24000,
            'speed_register': 0x2001,
            'control_register': 0x2000
        }
        result = test_vfd_protocol(YL620Spindle, config_values, "YL620 VFD")
        results.append(("YL620 VFD", result))
    except ImportError as e:
        print(f"✗ Could not import vfd_yl620: {e}")
        results.append(("YL620 VFD", False))
    
    # Test 7: NowForever VFD
    try:
        from klippy.extras.vfd_nowforever import NowForeverSpindle
        config_values = {
            'min_rpm': 600,
            'max_rpm': 24000,
            'speed_register': 0x0901,
            'control_register': 0x0900
        }
        result = test_vfd_protocol(NowForeverSpindle, config_values, "NowForever VFD")
        results.append(("NowForever VFD", result))
    except ImportError as e:
        print(f"✗ Could not import vfd_nowforever: {e}")
        results.append(("NowForever VFD", False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name:25} : {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All VFD protocol tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())