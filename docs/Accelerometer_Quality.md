# Accelerometer Quality Control

This document describes Klipper's real-time accelerometer-based print quality control feature. This feature continuously monitors vibration patterns during printing using an ADXL345 accelerometer and makes real-time speed adjustments to improve print quality.

## Overview

Unlike the existing accelerometer support which is mainly used for pre-print calibration (resonance testing and input shaping), this feature provides real-time monitoring during actual printing operations. The system:

- Continuously monitors vibration patterns during printing
- Automatically reduces speed when high vibration is detected
- Increases speed during stable printing conditions
- Provides G-code commands for runtime control

## Configuration

To use this feature, you must first have a working ADXL345 accelerometer configured. Then add the accelerometer quality control section to your configuration:

```ini
[accelerometer_quality]
accel_chip: adxl345
vibration_threshold: 5000.0
speed_reduction_factor: 0.8
speed_increase_factor: 1.1
sample_time: 1.0
stable_threshold: 1000.0
```

### Configuration Parameters

- `accel_chip`: Name of the accelerometer chip to use (default: "adxl345")
- `vibration_threshold`: Vibration magnitude threshold for triggering speed reduction in mm/s² (default: 5000.0)
- `speed_reduction_factor`: Factor to multiply current speed when high vibration is detected (default: 0.8 = 80%)
- `speed_increase_factor`: Factor to multiply current speed when conditions are stable (default: 1.1 = 110%)
- `sample_time`: Interval between vibration measurements in seconds (default: 1.0)
- `stable_threshold`: Vibration magnitude threshold for "stable" conditions in mm/s² (default: 1000.0)

## G-code Commands

### ACCELEROMETER_QUALITY_ENABLE
Enables real-time quality control monitoring. The accelerometer must be available and functional.

### ACCELEROMETER_QUALITY_DISABLE  
Disables real-time quality control and restores the original speed factor.

### ACCELEROMETER_QUALITY_STATUS
Reports the current status of the quality control system, including:
- Enable/monitoring state
- Current and baseline speed factors
- Configuration thresholds

## Usage

1. Ensure your ADXL345 accelerometer is properly configured and attached to the toolhead
2. Start your print normally
3. During printing, use `ACCELEROMETER_QUALITY_ENABLE` to activate real-time control
4. Monitor the system status with `ACCELEROMETER_QUALITY_STATUS`
5. Use `ACCELEROMETER_QUALITY_DISABLE` to deactivate when desired

## How It Works

The system works by:

1. **Continuous Monitoring**: A background thread samples accelerometer data at regular intervals
2. **Vibration Analysis**: Calculates RMS vibration magnitude by removing the DC component (gravity/static acceleration)
3. **Adaptive Speed Control**: Compares vibration levels to thresholds and adjusts print speed accordingly
4. **Speed Adjustment**: Uses M220 commands to modify the speed factor

### Vibration Detection

The system measures vibration by:
- Collecting accelerometer samples over the configured time window
- Calculating the mean acceleration (DC component)
- Computing RMS of variations from the mean (vibrations)
- Comparing to configured thresholds

### Speed Adjustment Logic

- **High Vibration**: When vibration exceeds `vibration_threshold`, speed is reduced by `speed_reduction_factor`
- **Stable Conditions**: When vibration is below `stable_threshold`, speed can be increased by `speed_increase_factor`
- **Safety Limits**: Speed will not be reduced below 30% of the baseline
- **Restoration**: Original speed is restored when the feature is disabled

## Safety Considerations

- The system will not reduce print speed below 30% of the original speed
- Speed changes are gradual and controlled to avoid print defects
- The feature can be disabled at any time to restore normal operation
- Memory usage is managed by periodic cleanup of accelerometer data

## Tuning Guidelines

### vibration_threshold
- Start with the default value (5000.0)
- Increase if the system is too sensitive (frequent speed reductions)
- Decrease if vibrations are not being detected appropriately

### stable_threshold  
- Should be significantly lower than vibration_threshold
- Controls when speed can be increased
- Lower values = more conservative speed increases

### sample_time
- Longer times = more stable measurements but slower response
- Shorter times = faster response but potentially more noise
- Recommended range: 0.5 to 2.0 seconds

### speed_reduction_factor
- Conservative starting point: 0.8 (20% reduction)
- For severe vibration issues: 0.5-0.7
- For minor adjustments: 0.85-0.95

## Troubleshooting

**"Accelerometer chip not found"**: Ensure the ADXL345 is properly configured and the chip name matches

**"Failed to start accelerometer client"**: Check ADXL345 wiring and configuration

**No speed changes observed**: Check that vibration thresholds are appropriate for your printer and print conditions

**Excessive speed reductions**: Increase vibration_threshold or check for mechanical issues causing vibrations