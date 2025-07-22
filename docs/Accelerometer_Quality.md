# Accelerometer Quality Control

This document describes Klipper's real-time accelerometer-based print quality control feature. This feature continuously monitors vibration patterns during printing using an ADXL345 accelerometer and makes real-time speed adjustments to improve print quality.

## Overview

Unlike the existing accelerometer support which is mainly used for pre-print calibration (resonance testing and input shaping), this feature provides real-time monitoring during actual printing operations. The system:

- Continuously monitors vibration patterns during printing
- Applies advanced signal processing including filtering and frequency analysis
- Uses enhanced decision-making based on vibration patterns and frequency content
- Automatically reduces speed when high vibration is detected
- Increases speed during stable printing conditions
- Provides simulation mode to analyze decisions without affecting prints
- Offers detailed decision logging and analysis capabilities
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
# Enhanced features
simulation_mode: False
sample_frequency: 100.0
enable_frequency_analysis: True
resonance_frequency_threshold: 50.0
filter_cutoff_frequency: 20.0
```

### Basic Configuration Parameters

- `accel_chip`: Name of the accelerometer chip to use (default: "adxl345")
- `vibration_threshold`: Vibration magnitude threshold for triggering speed reduction in mm/s² (default: 5000.0)
- `speed_reduction_factor`: Factor to multiply current speed when high vibration is detected (default: 0.8 = 80%)
- `speed_increase_factor`: Factor to multiply current speed when conditions are stable (default: 1.1 = 110%)
- `sample_time`: Interval between vibration measurements in seconds (default: 1.0)
- `stable_threshold`: Vibration magnitude threshold for "stable" conditions in mm/s² (default: 1000.0)

### Enhanced Processing Parameters

- `simulation_mode`: Enable simulation mode to capture decisions without applying them (default: False)
- `sample_frequency`: Expected accelerometer sample frequency in Hz for signal processing (default: 100.0)
- `enable_frequency_analysis`: Enable FFT-based frequency analysis for enhanced decisions (default: True)
- `resonance_frequency_threshold`: Frequency threshold in Hz for resonance detection (default: 50.0)
- `filter_cutoff_frequency`: Low-pass filter cutoff frequency in Hz, 0 to disable (default: 20.0)

## G-code Commands

### ACCELEROMETER_QUALITY_ENABLE
Enables real-time quality control monitoring. The accelerometer must be available and functional.

### ACCELEROMETER_QUALITY_DISABLE  
Disables real-time quality control and restores the original speed factor.

### ACCELEROMETER_QUALITY_STATUS
Reports the current status of the quality control system, including:
- Enable/monitoring state
- Simulation mode status
- Current and baseline speed factors
- Configuration thresholds
- Frequency analysis settings
- Decision count

### ACCELEROMETER_QUALITY_DECISIONS
Shows recent decision history with optional count parameter:
```gcode
ACCELEROMETER_QUALITY_DECISIONS COUNT=20
```
Displays detailed information about recent decisions including:
- Timestamp and reason for each decision
- Vibration levels and speed changes
- Whether decisions were applied or simulated
- Frequency analysis results

### ACCELEROMETER_QUALITY_SIMULATION
Enable or disable simulation mode during operation:
```gcode
ACCELEROMETER_QUALITY_SIMULATION ENABLE=1  ; Enable simulation mode
ACCELEROMETER_QUALITY_SIMULATION ENABLE=0  ; Disable simulation mode
ACCELEROMETER_QUALITY_SIMULATION           ; Show current mode
```

## Usage

1. Ensure your ADXL345 accelerometer is properly configured and attached to the toolhead
2. Start your print normally
3. During printing, use `ACCELEROMETER_QUALITY_ENABLE` to activate real-time control
4. Monitor the system status with `ACCELEROMETER_QUALITY_STATUS`
5. Review decision history with `ACCELEROMETER_QUALITY_DECISIONS`
6. Use `ACCELEROMETER_QUALITY_DISABLE` to deactivate when desired

### Simulation Mode Usage
For testing the system without affecting prints:
1. Set `simulation_mode: True` in configuration, or
2. Enable during operation: `ACCELEROMETER_QUALITY_SIMULATION ENABLE=1`
3. Enable monitoring: `ACCELEROMETER_QUALITY_ENABLE`
4. Observe decisions with `ACCELEROMETER_QUALITY_DECISIONS`
5. All speed adjustments will be logged but not applied

## Enhanced Processing Pipeline

The system implements a sophisticated processing pipeline:

### 1. Data Acquisition
- High-frequency sampling from ADXL345 accelerometer
- Configurable sample frequency (10-1000 Hz)
- Time-windowed data collection

### 2. Signal Processing
- Low-pass filtering to remove high-frequency noise
- Configurable filter cutoff frequency
- DC component removal (gravity compensation)

### 3. Feature Extraction
- FFT-based frequency analysis
- Dominant frequency detection
- Spectral centroid calculation
- Frequency distribution analysis (low/mid/high frequency ratios)
- Total energy calculation

### 4. Decision Engine
- Multi-criteria decision making based on:
  - RMS vibration magnitude
  - Dominant frequency content
  - Frequency distribution
  - Resonance detection
- Enhanced logic for different vibration patterns:
  - Basic high vibration → speed reduction
  - Resonance detected → aggressive speed reduction
  - Stable with high frequency content → conservative speed increase
  - Low vibration → normal speed increase

### 5. Printer Control
- M220 speed factor commands
- Simulation mode support
- Decision logging and analysis

## How It Works

The system works by:

1. **Continuous Monitoring**: A background thread samples accelerometer data at regular intervals
2. **Signal Processing**: Applies low-pass filters and removes DC components
3. **Frequency Analysis**: Uses FFT to analyze frequency content and detect patterns
4. **Enhanced Decision Making**: Considers multiple factors including vibration magnitude and frequency patterns
5. **Speed Adjustment**: Uses M220 commands to modify the speed factor or logs decisions in simulation mode

### Vibration Detection

The system measures vibration by:
- Collecting accelerometer samples over the configured time window
- Applying configurable low-pass filtering
- Calculating the mean acceleration (DC component)
- Computing RMS of variations from the mean (vibrations)
- Performing FFT analysis to extract frequency features
- Comparing to configured thresholds with enhanced logic

### Enhanced Speed Adjustment Logic

- **High Vibration**: When vibration exceeds `vibration_threshold`, speed is reduced by `speed_reduction_factor`
- **Resonance Detection**: When dominant frequency exceeds `resonance_frequency_threshold`, more aggressive speed reduction
- **Stable Conditions**: When vibration is below `stable_threshold`, speed can be increased by `speed_increase_factor`
- **High Frequency Content**: Conservative speed increases when high frequency vibrations are present
- **Safety Limits**: Speed will not be reduced below 30% of the baseline
- **Simulation Mode**: All decisions logged but not applied when enabled

## Safety Considerations

- The system will not reduce print speed below 30% of the original speed
- Speed changes are gradual and controlled to avoid print defects
- The feature can be disabled at any time to restore normal operation
- Memory usage is managed by periodic cleanup of accelerometer data
- Simulation mode allows safe testing without affecting prints

## Tuning Guidelines

### Basic Parameters

#### vibration_threshold
- Start with the default value (5000.0)
- Increase if the system is too sensitive (frequent speed reductions)
- Decrease if vibrations are not being detected appropriately

#### stable_threshold  
- Should be significantly lower than vibration_threshold
- Controls when speed can be increased
- Lower values = more conservative speed increases

#### sample_time
- Longer times = more stable measurements but slower response
- Shorter times = faster response but potentially more noise
- Recommended range: 0.5 to 2.0 seconds

#### speed_reduction_factor
- Conservative starting point: 0.8 (20% reduction)
- For severe vibration issues: 0.5-0.7
- For minor adjustments: 0.85-0.95

### Enhanced Parameters

#### sample_frequency
- Should match your accelerometer's actual sample rate
- Higher values enable better frequency analysis
- Typical range: 50-200 Hz for most applications

#### resonance_frequency_threshold
- Set based on your printer's known resonance frequencies
- Lower values = more sensitive to resonance detection
- Typical range: 30-80 Hz for most printers

#### filter_cutoff_frequency
- Set based on the frequency content you want to preserve
- Lower values = more aggressive filtering
- Set to 0 to disable filtering
- Typical range: 10-30 Hz

## Troubleshooting

**"Accelerometer chip not found"**: Ensure the ADXL345 is properly configured and the chip name matches

**"Failed to start accelerometer client"**: Check ADXL345 wiring and configuration

**No speed changes observed**: 
- Check that vibration thresholds are appropriate for your printer and print conditions
- Enable simulation mode to see what decisions would be made
- Review decision history with `ACCELEROMETER_QUALITY_DECISIONS`

**Excessive speed reductions**: 
- Increase vibration_threshold or resonance_frequency_threshold
- Check for mechanical issues causing vibrations
- Review frequency analysis output for patterns

**Poor frequency analysis results**:
- Verify sample_frequency matches actual accelerometer rate
- Adjust filter_cutoff_frequency
- Ensure sufficient sample_time for meaningful FFT analysis

**System too sensitive/not sensitive enough**:
- Use simulation mode to analyze decision patterns
- Adjust thresholds based on decision history
- Consider mechanical improvements to reduce baseline vibrations