# Modbus RTU communication support for Klipper
#
# Copyright (C) 2024  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging, threading, time, queue
import serial

class ModbusError(Exception):
    pass

class ModbusRTU:
    """Modbus RTU communication handler"""
    
    def __init__(self, config):
        self.printer = config.get_printer()
        self.reactor = self.printer.get_reactor()
        
        # Serial port configuration
        self.device = config.get('device', '/dev/ttyUSB0')
        self.baudrate = config.getint('baudrate', 9600)
        self.bytesize = config.getint('bytesize', 8)
        self.parity = config.get('parity', 'N')
        self.stopbits = config.getint('stopbits', 1)
        self.timeout = config.getfloat('timeout', 1.0)
        
        # Modbus settings
        self.slave_id = config.getint('slave_id', 1, minval=1, maxval=247)
        self.retries = config.getint('retries', 3, minval=1)
        self.retry_delay = config.getfloat('retry_delay', 0.1)
        
        # Communication state
        self.serial_port = None
        self.command_queue = queue.Queue(maxsize=20)
        self.response_queue = queue.Queue()
        self.comm_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # Register for shutdown
        self.printer.register_event_handler('klippy:shutdown', self._shutdown)
        
    def _shutdown(self):
        """Cleanup on shutdown"""
        self.running = False
        if self.comm_thread and self.comm_thread.is_alive():
            self.comm_thread.join(timeout=2.0)
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            
    def connect(self):
        """Establish serial connection and start communication thread"""
        try:
            self.serial_port = serial.Serial(
                port=self.device,
                baudrate=self.baudrate,
                bytesize=self.bytesize,
                parity=self.parity,
                stopbits=self.stopbits,
                timeout=self.timeout
            )
            
            # Enable RS485 mode if supported
            if hasattr(self.serial_port, 'rs485_mode'):
                from serial.rs485 import RS485Settings
                self.serial_port.rs485_mode = RS485Settings()
                
            self.running = True
            self.comm_thread = threading.Thread(target=self._comm_worker)
            self.comm_thread.daemon = True
            self.comm_thread.start()
            
            logging.info(f"Modbus RTU connected to {self.device} at {self.baudrate} baud")
            
        except Exception as e:
            raise ModbusError(f"Failed to connect to Modbus device: {e}")
            
    def _calc_crc16(self, data):
        """Calculate Modbus RTU CRC16"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        # Return CRC in little-endian format for Modbus RTU
        return crc
        
    def _build_frame(self, slave_id, function_code, data):
        """Build Modbus RTU frame with CRC"""
        frame = bytearray([slave_id, function_code]) + data
        crc = self._calc_crc16(frame)
        # Append CRC in little-endian format (low byte first)
        frame.extend([crc & 0xFF, (crc >> 8) & 0xFF])
        return frame
        
    def _validate_frame(self, frame):
        """Validate received Modbus RTU frame"""
        if len(frame) < 4:
            return False
            
        # Check CRC
        data = frame[:-2]
        # CRC is in little-endian format (low byte first)
        received_crc = frame[-2] | (frame[-1] << 8)
        calculated_crc = self._calc_crc16(data)
        
        return received_crc == calculated_crc
        
    def _comm_worker(self):
        """Communication thread worker"""
        while self.running:
            try:
                # Get command from queue with timeout
                try:
                    command = self.command_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                    
                success = False
                response = None
                
                # Retry loop
                for attempt in range(self.retries):
                    try:
                        # Send command
                        self.serial_port.reset_input_buffer()
                        self.serial_port.reset_output_buffer()
                        self.serial_port.write(command['frame'])
                        
                        # Read response
                        if command.get('expect_response', True):
                            # Read at least 4 bytes (slave_id, function, crc)
                            response_data = self.serial_port.read(4)
                            if len(response_data) >= 4:
                                # Check if this is an error response
                                if response_data[1] & 0x80:
                                    # Error response - read one more byte for error code
                                    error_code = self.serial_port.read(1)
                                    if error_code:
                                        response_data += error_code
                                else:
                                    # Normal response - read additional data based on function code
                                    if response_data[1] == 0x03:  # Read holding registers
                                        data_length = response_data[2] if len(response_data) > 2 else 0
                                        if data_length > 0:
                                            additional_data = self.serial_port.read(data_length)
                                            response_data += additional_data
                                            
                                # Validate frame
                                if self._validate_frame(response_data):
                                    response = response_data
                                    success = True
                                    break
                                    
                    except Exception as e:
                        logging.debug(f"Modbus communication attempt {attempt + 1} failed: {e}")
                        
                    if attempt < self.retries - 1:
                        time.sleep(self.retry_delay)
                        
                # Put result in response queue
                self.response_queue.put({
                    'command_id': command['id'],
                    'success': success,
                    'response': response,
                    'error': None if success else 'Communication failed after retries'
                })
                
                self.command_queue.task_done()
                
            except Exception as e:
                logging.error(f"Modbus communication thread error: {e}")
                time.sleep(0.1)
                
    def send_command(self, function_code, data, expect_response=True, timeout=2.0):
        """Send Modbus command and wait for response"""
        if not self.running or not self.serial_port:
            raise ModbusError("Modbus not connected")
            
        # Build frame
        frame = self._build_frame(self.slave_id, function_code, data)
        
        # Generate unique command ID
        command_id = time.time()
        
        # Queue command
        command = {
            'id': command_id,
            'frame': frame,
            'expect_response': expect_response
        }
        
        try:
            self.command_queue.put(command, timeout=1.0)
        except queue.Full:
            raise ModbusError("Command queue full")
            
        if not expect_response:
            return None
            
        # Wait for response
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result = self.response_queue.get(timeout=0.1)
                if result['command_id'] == command_id:
                    if result['success']:
                        return result['response']
                    else:
                        raise ModbusError(result['error'])
            except queue.Empty:
                continue
                
        raise ModbusError("Command timeout")
        
    def read_holding_registers(self, address, count):
        """Read holding registers (function code 0x03)"""
        data = bytearray()
        data.extend([(address >> 8) & 0xFF, address & 0xFF])
        data.extend([(count >> 8) & 0xFF, count & 0xFF])
        
        response = self.send_command(0x03, data)
        
        if response and len(response) >= 3:
            # Check if error response
            if response[1] & 0x80:
                error_code = response[2] if len(response) > 2 else 0
                raise ModbusError(f"Modbus error code: {error_code}")
                
            # Parse data
            data_length = response[2]
            if len(response) >= 3 + data_length:
                register_data = response[3:3+data_length]
                # Convert bytes to 16-bit registers
                registers = []
                for i in range(0, len(register_data), 2):
                    if i + 1 < len(register_data):
                        reg_value = (register_data[i] << 8) | register_data[i + 1]
                        registers.append(reg_value)
                return registers
                
        raise ModbusError("Invalid response format")
        
    def write_single_register(self, address, value):
        """Write single register (function code 0x06)"""
        data = bytearray()
        data.extend([(address >> 8) & 0xFF, address & 0xFF])
        data.extend([(value >> 8) & 0xFF, value & 0xFF])
        
        response = self.send_command(0x06, data)
        
        if response and len(response) >= 6:
            # Check if error response
            if response[1] & 0x80:
                error_code = response[2] if len(response) > 2 else 0
                raise ModbusError(f"Modbus error code: {error_code}")
            return True
            
        raise ModbusError("Invalid response format")

def load_config(config):
    return ModbusRTU(config)