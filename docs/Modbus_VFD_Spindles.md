# Modbus VFD Spindle Control

This document describes the comprehensive Modbus VFD (Variable Frequency Drive) support in Klipper, enabling precise spindle control for CNC machines using RS485/Modbus RTU communication.

## Overview

Klipper supports multiple VFD protocols through Modbus RTU communication, allowing control of various industrial spindle motors. This implementation provides:

- **Real-time speed control** with RPM feedback
- **Bidirectional operation** (CW/CCW rotation)
- **Status monitoring** with error detection
- **Multiple VFD protocols** for different manufacturers
- **RS485 communication** via USB adapters

## Supported VFD Types

### 1. Generic Modbus VFD
Basic support for standard Modbus-compatible VFDs using common register layouts.

**Configuration:**
```ini
[modbus_spindle]
protocol: generic
speed_register: 0x2000
control_register: 0x2001
```

### 2. Huanyang VFD
Comprehensive support for popular Chinese Huanyang VFDs with proper frequency scaling.

**Configuration:**
```ini
[huanyang_spindle]
rated_frequency: 50.0
max_frequency: 100.0
```

**Features:**
- Frequency-based speed control
- Proper Hz to RPM conversion
- Status monitoring with error detection
- Direction change handling

### 3. H100 Series VFD
Support for H100 series VFDs with coil-based control.

**Configuration:**
```ini
[modbus_spindle h100]
protocol: h100
speed_register: 0x0201
control_register: 0x0049
```

**Features:**
- Reads min/max frequency from VFD
- Automatic speed limit configuration
- Output frequency monitoring

### 4. H2A Series VFD
Advanced support for H2A VFDs with percentage-based speed control.

**Configuration:**
```ini
[vfd_h2a]
speed_register: 0x1000    # Percentage (0-10000 = 0-100%)
control_register: 0x2000
```

**Features:**
- Percentage-based speed setting
- Direct RPM feedback
- Automatic max RPM detection

### 5. Danfoss VLT2800 Series
Industrial-grade support for Danfoss VLT2800 drives with complex control words.

**Configuration:**
```ini
[vfd_danfoss_vlt2800]
max_frequency: 400.0
```

**Features:**
- Complex control word management
- Multiple coil control
- Industrial fault monitoring
- Cached state management

### 6. Siemens V20 Series
Professional support for Siemens V20 drives with scaled frequency control.

**Configuration:**
```ini
[vfd_siemens_v20]
motor_poles: 2
motor_phases: 3
max_frequency: 400.0
```

**Features:**
- Scaled frequency control (16384 resolution)
- Motor parameter-based RPM calculation
- Comprehensive status monitoring
- Fault detection and reporting

### 7. YL620 Series VFD
Support for Yalang YL620/YL620-A VFDs with deciHz format.

**Configuration:**
```ini
[vfd_yl620]
speed_register: 0x2001    # Frequency in 0.1Hz
control_register: 0x2000
```

**Features:**
- DeciHz frequency format (0.1Hz resolution)
- Automatic frequency limit detection
- Command register control

### 8. NowForever Series VFD
Support for NowForever VFDs with Hz*100 frequency format.

**Configuration:**
```ini
[vfd_nowforever]
speed_register: 0x0901    # Frequency in Hz*100
fault_register: 0x0300
```

**Features:**
- Hz*100 frequency format
- Multiple register write operations
- Fault status monitoring
- Direction status feedback

## Hardware Setup

### USB to RS485 Adapter
Connect your VFD to the Klipper host using a USB to RS485 adapter:

1. **USB to RS485 Converter**: Use a quality USB-RS485 adapter
2. **Wiring**: Connect A/B terminals to VFD's RS485 terminals
3. **Termination**: Add 120Ω resistors at cable ends for long runs
4. **Shielding**: Use shielded twisted pair cable

### VFD Configuration
Each VFD must be configured for Modbus RTU operation:

#### Common Parameters:
- **Protocol**: Set to Modbus RTU
- **Slave Address**: Match the `slave_id` in config
- **Baud Rate**: Typically 9600 or 19200 bps
- **Parity**: Usually None or Even
- **Control Source**: Set to RS485/Communication

#### VFD-Specific Setup:

**Huanyang VFD:**
```
PD163: Communication address (1-247)
PD164: Baud rate (9600, 19200, etc.)
PD165: Communication protocol (1=Modbus RTU)
```

**Siemens V20:**
```
P0700[0]: Command source = 5 (RS485)
P1000[0]: Frequency setpoint = 5 (RS485)
P2023[0]: Protocol = 2 (Modbus RTU)
P2021[0]: Modbus address
```

**YL620:**
```
P00.01: Command source = 3
P03.00: RS485 Baud rate = 3 (9600)
P03.01: RS485 address = 1
P03.02: RS485 protocol = 2
```

## Configuration

### Basic Setup

1. **Configure Modbus Interface:**
```ini
[modbus]
device: /dev/ttyUSB0
baudrate: 9600
slave_id: 1
```

2. **Choose VFD Type:**
```ini
# For Huanyang VFDs
[huanyang_spindle]
min_rpm: 100
max_rpm: 24000

# For Siemens V20
[vfd_siemens_v20]
motor_poles: 2
max_frequency: 400.0

# For generic VFDs
[modbus_spindle]
protocol: generic
```

### Advanced Configuration

**Custom Register Addresses:**
```ini
[modbus_spindle custom]
protocol: generic
speed_register: 0x1000      # Custom speed register
control_register: 0x1001    # Custom control register
status_register: 0x2000     # Custom status register
frequency_register: 0x2001  # Custom frequency register
```

**Status Monitoring:**
```ini
[vfd_siemens_v20]
status_interval: 0.5        # Poll every 500ms
```

**Communication Parameters:**
```ini
[modbus]
timeout: 2.0               # Increase for slow VFDs
retries: 5                 # More retries for reliability
retry_delay: 0.2           # Delay between retries
```

## G-code Commands

All VFD spindles support standard spindle G-codes:

### M3 - Spindle On Clockwise
```gcode
M3 S12000          ; Start spindle at 12000 RPM clockwise
```

### M4 - Spindle On Counter-Clockwise
```gcode
M4 S8000           ; Start spindle at 8000 RPM counter-clockwise
```

### M5 - Spindle Off
```gcode
M5                 ; Stop spindle
```

### Usage Examples

**Basic Operation:**
```gcode
M3 S15000          ; Start spindle at 15000 RPM
G4 P2              ; Wait 2 seconds for spin-up
G0 X10 Y10         ; Move to position
G1 Z-2 F200        ; Feed into material
; ... machining operations ...
M5                 ; Stop spindle
```

**With Tool Changes:**
```gcode
T1                 ; Select tool 1
M3 S12000          ; Start spindle
; ... operations ...
M5                 ; Stop for tool change
T2                 ; Select tool 2
M3 S18000          ; Different speed for new tool
```

## Status and Monitoring

### Real-time Feedback
VFD spindles provide real-time status information:

- **Current Speed**: Actual RPM from VFD feedback
- **Target Speed**: Commanded RPM
- **Direction**: CW/CCW rotation
- **Status**: Running/stopped state
- **Errors**: VFD fault conditions

### Status Queries
Check spindle status via console:
```
QUERY_SPINDLE
```

Output includes:
```
spindle: enabled=True speed=12000.0 target_speed=12000.0 direction=CW protocol=siemens_v20
```

### Error Handling
VFD errors are logged and reported:
- Communication timeouts
- VFD fault conditions
- Speed deviation warnings
- Protocol-specific errors

## Troubleshooting

### Communication Issues

**No Response from VFD:**
1. Check USB-RS485 adapter connection
2. Verify VFD slave address matches config
3. Ensure VFD is in Modbus RTU mode
4. Check cable wiring (A/B terminals)

**Timeout Errors:**
1. Increase `timeout` value in [modbus] section
2. Reduce `status_interval` frequency
3. Check for electromagnetic interference
4. Add termination resistors

**CRC Errors:**
1. Verify cable quality and shielding
2. Check for loose connections
3. Reduce baud rate
4. Add ferrite cores on cable

### Speed Control Issues

**Incorrect Speed:**
1. Verify max_rpm setting matches VFD
2. Check VFD frequency parameters
3. Ensure proper motor pole count (for some VFDs)
4. Verify speed register address

**No Speed Change:**
1. Check if VFD is in remote control mode
2. Verify speed register address
3. Ensure VFD accepts frequency commands
4. Check for VFD parameter locks

### VFD-Specific Issues

**Huanyang VFD:**
- Ensure PD165 = 1 (Modbus RTU mode)
- Check rated frequency setting
- Verify frequency scaling parameters

**Siemens V20:**
- Confirm P0700[0] and P1000[0] = 5 (RS485)
- Check parity setting (usually Even)
- Verify P2023[0] = 2 (Modbus RTU)

**YL620:**
- Ensure P00.01 = 3 (communication control)
- Check P03.02 = 2 (Modbus RTU protocol)
- Verify frequency range settings

## Advanced Features

### Multiple Spindles
Configure multiple spindles with different VFD types:

```ini
[modbus modbus1]
device: /dev/ttyUSB0
slave_id: 1

[modbus modbus2]  
device: /dev/ttyUSB1
slave_id: 1

[huanyang_spindle main_spindle]
; Uses modbus1

[vfd_siemens_v20 aux_spindle]
; Uses modbus2
```

### Custom Protocols
Extend support for new VFD types by creating custom protocol implementations based on existing examples.

### Integration with Macros
Use VFD spindles in Klipper macros for automated operations:

```ini
[gcode_macro START_JOB]
gcode:
    M3 S{params.SPEED|default(12000)}
    G4 P3                    ; Wait for spin-up
    ; Continue with job...

[gcode_macro END_JOB]
gcode:
    M5                       ; Stop spindle
    G4 P5                    ; Wait for spin-down
```

## Performance Considerations

### Communication Frequency
- Default status polling: 1 second
- Reduce for real-time applications
- Increase for slower systems

### Response Time
- Typical command response: 50-200ms
- Varies by VFD type and settings
- Factor into motion planning

### Reliability
- All protocols include retry logic
- CRC validation ensures data integrity
- Error recovery and logging

## Safety Considerations

1. **Emergency Stop**: Ensure E-stop cuts VFD power
2. **Spindle Protection**: Configure proper limits
3. **Direction Changes**: Allow spin-down time
4. **Error Handling**: Monitor VFD fault conditions
5. **Electrical Safety**: Follow VFD installation guidelines

## Conclusion

Klipper's comprehensive Modbus VFD support enables professional CNC spindle control with industrial-grade reliability. The modular architecture supports multiple VFD types while maintaining consistent G-code compatibility and ease of configuration.