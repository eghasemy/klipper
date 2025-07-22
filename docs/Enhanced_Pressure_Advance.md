# Enhanced Pressure Advance Documentation

## Overview

This enhancement extends Klipper's pressure advance system with variable pressure advance based on extrusion rate changes, support for non-linear pressure advance models, and automatic calibration routines.

## Features

### Variable Pressure Advance

Variable pressure advance adjusts the pressure advance value dynamically based on the current extrusion rate, providing better printing quality across different speeds and flow rates.

### Configuration

Add these parameters to your extruder configuration:

```
[extruder]
# ... existing extruder config ...

# Enable variable pressure advance
pressure_advance_variable: True
pressure_advance_rate_min: 1.0          # Minimum extrusion rate (mm/s)
pressure_advance_rate_max: 8.0          # Maximum extrusion rate (mm/s)  
pressure_advance_value_min: 0.02        # PA value at minimum rate
pressure_advance_value_max: 0.15        # PA value at maximum rate
pressure_advance_rate_power: 1.5        # Power factor for non-linear interpolation
```

### G-code Commands

#### SET_VARIABLE_PRESSURE_ADVANCE

Configure variable pressure advance parameters:

```
SET_VARIABLE_PRESSURE_ADVANCE EXTRUDER=extruder RATE_MIN=1.0 RATE_MAX=8.0 PA_MIN=0.02 PA_MAX=0.15 POWER=1.5 [SMOOTH_TIME=0.040]
```

Parameters:
- `RATE_MIN`: Minimum extrusion rate in mm/s
- `RATE_MAX`: Maximum extrusion rate in mm/s
- `PA_MIN`: Pressure advance value at minimum rate
- `PA_MAX`: Pressure advance value at maximum rate
- `POWER`: Power factor for interpolation (1.0 = linear, >1.0 = exponential)
- `SMOOTH_TIME`: Smoothing time (optional, defaults to current value)

#### CALIBRATE_PRESSURE_ADVANCE

Automatically calibrate pressure advance across different rates:

```
CALIBRATE_PRESSURE_ADVANCE EXTRUDER=extruder [START_RATE=1.0] [END_RATE=8.0] [STEPS=5] [START_PA=0.01] [END_PA=0.1] [DISTANCE=50]
```

Parameters:
- `START_RATE`: Starting extrusion rate for calibration
- `END_RATE`: Ending extrusion rate for calibration  
- `STEPS`: Number of test steps to perform
- `START_PA`: Starting pressure advance value
- `END_PA`: Ending pressure advance value
- `DISTANCE`: Test extrusion distance in mm

## How It Works

### Non-Linear Interpolation

The pressure advance value is calculated using:

```
rate_factor = (current_rate - rate_min) / (rate_max - rate_min)
rate_factor = rate_factor^power
pa_value = pa_min + rate_factor * (pa_max - pa_min)
```

The power factor allows for non-linear relationships:
- `power = 1.0`: Linear interpolation
- `power > 1.0`: More aggressive PA at higher rates
- `power < 1.0`: More aggressive PA at lower rates

### Backward Compatibility

The traditional pressure advance system remains fully functional. Existing configurations will work unchanged. Variable pressure advance is only enabled when explicitly configured.

## Examples

### Basic Variable Setup

```
[extruder]
# Standard extruder config...
pressure_advance_variable: True
pressure_advance_rate_min: 2.0
pressure_advance_rate_max: 10.0  
pressure_advance_value_min: 0.03
pressure_advance_value_max: 0.12
pressure_advance_rate_power: 1.2
```

### Non-Linear Response

```
# More aggressive PA at higher speeds
SET_VARIABLE_PRESSURE_ADVANCE EXTRUDER=extruder RATE_MIN=1.0 RATE_MAX=8.0 PA_MIN=0.02 PA_MAX=0.20 POWER=2.0

# More conservative PA at higher speeds  
SET_VARIABLE_PRESSURE_ADVANCE EXTRUDER=extruder RATE_MIN=1.0 RATE_MAX=8.0 PA_MIN=0.08 PA_MAX=0.12 POWER=0.5
```

### Calibration Workflow

```
# Home and heat up
G28
M104 S200
M190 S60

# Run calibration
CALIBRATE_PRESSURE_ADVANCE EXTRUDER=extruder START_RATE=2.0 END_RATE=12.0 STEPS=6 START_PA=0.02 END_PA=0.15

# Test results with actual prints
G1 X50 Y50 E10 F300   # Slow print
G1 X100 Y100 E15 F1800 # Fast print
```