# CNC Machine Support in Klipper

Klipper now includes comprehensive support for CNC machines, enabling users to control milling machines, routers, lathes, and other CNC equipment. This document describes the available features and configuration options.

## Overview

The CNC support in Klipper provides:

- **Spindle Control**: M3/M4/M5 commands with PWM speed control
- **Coolant Control**: M7/M8/M9 commands for mist and flood coolant
- **Tool Change**: T-commands with automated tool change sequences  
- **CNC Probing**: G38.x commands for touch-plate and edge-finding
- **Work Coordinate Systems**: G54-G59 for multiple part origins
- **Feed Hold/Pause**: M0/M1 with safe pause/resume sequences
- **Canned Cycles**: G81/G82/G83 drilling cycles
- **Multi-Axis Support**: A/B rotary axes with acceleration limiting
- **Handwheel Control**: Manual jog wheels for precise positioning

All CNC features coexist with existing 3D printer functionality, allowing dual-purpose machines.

## Configuration

### Spindle Control

The `[spindle]` section configures spindle control:

```ini
[spindle]
enable_pin: ar8             # Relay/SSR for spindle enable
direction_pin: ar9          # CW/CCW direction control (optional)
pwm_pin: ar10              # PWM output for VFD speed control
cycle_time: 0.001          # PWM frequency (1kHz)
min_rpm: 100               # Minimum spindle speed
max_rpm: 24000             # Maximum spindle speed  
min_power: 0.1             # Minimum PWM duty cycle
max_power: 1.0             # Maximum PWM duty cycle
```

#### Commands

- `M3 S[rpm]` - Start spindle clockwise at specified RPM
- `M4 S[rpm]` - Start spindle counter-clockwise at specified RPM
- `M5` - Stop spindle

### Coolant Control

The `[coolant]` section configures coolant systems:

```ini
[coolant]
mist_pin: ar11             # Mist coolant control
flood_pin: ar12            # Flood coolant control
```

#### Commands

- `M7` - Turn on mist coolant
- `M8` - Turn on flood coolant
- `M9` - Turn off all coolant

### Tool Change

The `[tool_change]` section enables tool change support:

```ini
[tool_change]
max_tool: 20               # Maximum tool number
tool_change_macro: TOOL_CHANGE_SEQUENCE
park_x: 0                  # Tool change parking position
park_y: 0
park_z: 25
```

#### Commands

- `T0`, `T1`, etc. - Select specified tool
- `TOOL_CHANGE TOOL=N` - Manual tool change
- `GET_CURRENT_TOOL` - Report current tool

### CNC Probing

The `[cnc_probing]` section configures touch-plate probing:

```ini
[cnc_probing]
probe_pin: ^ar19           # Touch probe input
speed: 5                   # Probing speed
samples: 3                 # Number of samples
sample_retract_dist: 2.0   # Retract distance
samples_tolerance: 0.05    # Sample tolerance
```

#### Commands

- `G38.2` - Probe toward workpiece (error if no contact)
- `G38.3` - Probe toward workpiece (no error)
- `G38.4` - Probe away from workpiece (error if contact remains)
- `G38.5` - Probe away from workpiece (no error)

### Work Coordinate Systems

The `[work_coordinate_systems]` section enables G54-G59 support:

```ini
[work_coordinate_systems]
# No additional configuration required
```

#### Commands

- `G54` through `G59` - Select work coordinate system
- `G10 L2 P[1-6] X[val] Y[val] Z[val]` - Set coordinate system offsets
- `G92 X[val] Y[val] Z[val]` - Set current position
- `G53` - Move in machine coordinates
- `WCS_INFO` - Display coordinate system information

### Feed Hold and Pause

The `[feed_hold]` section configures pause functionality:

```ini
[feed_hold]
feed_hold_pin: ^ar20       # External feed hold button (optional)
retract_length: 1.0        # Retract during pause
lift_z: 5.0               # Z lift during pause
move_speed: 100           # Resume move speed
```

#### Commands

- `M0 [P<seconds>]` - Program stop (optional auto-resume)
- `M1` - Optional stop
- `FEED_HOLD` - Immediate motion stop
- `CYCLE_START` / `RESUME_CNC` - Resume operation

### Canned Drilling Cycles

The `[canned_cycles]` section enables drilling cycles:

```ini
[canned_cycles]
default_feed_rate: 200     # Default drilling feed rate
default_rapid_rate: 1500   # Default rapid rate
default_dwell_time: 0.5    # Default dwell time
```

#### Commands

- `G81 X Y Z R F` - Simple drilling cycle
- `G82 X Y Z R F P` - Drilling with dwell
- `G83 X Y Z R F Q` - Peck drilling cycle
- `G80` - Cancel canned cycles
- `G98` - Return to initial Z level
- `G99` - Return to R plane

### Multi-Axis Support

The `[multi_axis]` section adds rotary axes:

```ini
[multi_axis]
# A axis (4th axis)
a_step_pin: ar36
a_dir_pin: ar34
a_enable_pin: !ar30
a_rotation_distance: 360.0    # Degrees per revolution
a_max_velocity: 45            # Degrees/second
a_max_accel: 300             # Degrees/second²

# Acceleration limiting
accel_limited_axes: z,a      # Limit specific axes
z_max_accel: 200            # Override Z acceleration
a_max_accel: 150            # Override A acceleration
```

#### Commands

- `G28.A` - Home A axis
- `G28.B` - Home B axis (if configured)
- `MULTI_AXIS_STATUS` - Report axis configuration
- `SET_AXIS_LIMITS AXIS=a MIN=-180 MAX=180` - Set axis limits

### Handwheel Control

The `[handwheel]` section enables manual jog wheels:

```ini
[handwheel]
encoder_pin_a: ^ar21       # Encoder A phase
encoder_pin_b: ^ar22       # Encoder B phase
axis_x_pin: ^ar23          # X axis selection
axis_y_pin: ^ar24          # Y axis selection
axis_z_pin: ^ar25          # Z axis selection
multiplier_0_pin: ^ar26    # 0.001mm step
multiplier_1_pin: ^ar27    # 0.01mm step
multiplier_2_pin: ^ar28    # 0.1mm step
multiplier_3_pin: ^ar29    # 1.0mm step
step_multipliers: 0.001,0.01,0.1,1.0
max_velocity: 25
```

#### Commands

- `HANDWHEEL_ENABLE` - Enable handwheel control
- `HANDWHEEL_DISABLE` - Disable handwheel control
- `HANDWHEEL_JOG AXIS=x DISTANCE=1.0` - Manual jog command
- `HANDWHEEL_SET_AXIS AXIS=x` - Set active axis
- `HANDWHEEL_SET_STEP SIZE=0.1` - Set step size

## Example Workflows

### Basic CNC Setup

1. **Initialize machine:**
   ```gcode
   CNC_START          ; Initialize CNC mode
   ```

2. **Set work coordinates:**
   ```gcode
   G54               ; Select coordinate system
   G38.2 X-10 F100   ; Probe X edge
   G38.2 Y-10 F100   ; Probe Y edge  
   G38.2 Z-10 F50    ; Probe Z surface
   G92 X0 Y0 Z0      ; Set origin
   ```

3. **Start machining:**
   ```gcode
   M3 S12000         ; Start spindle at 12000 RPM
   M8                ; Turn on flood coolant
   G0 X10 Y10        ; Rapid to start position
   G1 Z-2 F200       ; Plunge cut
   ```

### Tool Change Sequence

```gcode
T1                  ; Select tool 1
; (Automatic tool change sequence runs)
G43 H1              ; Apply tool length offset (if configured)
```

### Drilling Operations

```gcode
G98                 ; Return to initial Z
G81 X10 Y10 Z-5 R2 F200  ; Drill hole at X10 Y10
G81 X20 Y10         ; Drill another hole (reuse parameters)
G80                 ; Cancel canned cycle
```

### Emergency Procedures

```gcode
FEED_HOLD          ; Immediate stop
; Or use physical feed hold button

CYCLE_START        ; Resume when ready
```

## Safety Considerations

1. **Always verify spindle direction** before starting operations
2. **Use appropriate speeds and feeds** for your tooling and material
3. **Enable feed hold button** for immediate stop capability
4. **Test tool changes** in safe positions before production
5. **Verify work coordinate systems** before machining
6. **Use proper workholding** and spindle balancing

## Integration with CAM Software

Popular CAM packages can generate G-code compatible with Klipper's CNC support:

- **Fusion 360**: Use "Generic CNC" post-processor
- **LinuxCNC Compatible**: Most LinuxCNC G-code works directly
- **Mach3/4 Compatible**: Basic compatibility for standard G-codes
- **Custom Post-Processors**: Can be created for specific requirements

## Troubleshooting

### Spindle Issues
- Check PWM signal with oscilloscope
- Verify VFD parameter settings
- Ensure proper grounding and shielding

### Probing Problems
- Verify probe wiring and pull-up resistors
- Check probe travel distances
- Calibrate probe accuracy

### Tool Change Failures
- Check macro syntax and logic
- Verify parking positions are reachable
- Test manual tool change sequence

For additional support, consult the Klipper documentation and community forums.