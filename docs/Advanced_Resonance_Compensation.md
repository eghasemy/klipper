# Advanced Resonance Compensation

This document describes the enhanced input shaping capabilities in Klipper that provide comprehensive resonance testing and compensation beyond the basic features described in [Resonance_Compensation.md](Resonance_Compensation.md).

## Overview

The advanced resonance compensation system includes:

- **New sophisticated input shaper algorithms** optimized for different scenarios
- **Comprehensive resonance analysis** with frequency, phase, and harmonic analysis  
- **Multi-point calibration** across the print bed to detect spatial variations
- **Intelligent recommendation system** that adapts to your printer's characteristics
- **Cross-axis coupling detection** and compensation
- **Enhanced diagnostic tools** for mechanical system health monitoring

## Advanced Input Shapers

### New Shaper Types

In addition to the standard shapers (ZV, MZV, ZVD, EI, 2hump_EI, 3hump_EI), the following advanced shapers are available:

#### `smooth`
Optimized for minimal smoothing while maintaining good vibration reduction. Best for printers where print speed is prioritized over maximum vibration suppression.

#### `adaptive_ei` 
An Enhanced Input shaper that automatically adjusts its parameters based on the measured damping ratio of your printer. Provides optimal performance across different mechanical configurations.

#### `multi_freq`
Designed for printers with complex resonance patterns or multiple dominant frequencies. Uses a multi-frequency approach to compensate for harmonic resonances.

#### `ulv` (Ultra Low Vibration)
The most aggressive vibration reduction shaper using a 6-impulse design. Provides maximum vibration suppression at the cost of increased smoothing. Ideal for high-quality prints where surface finish is critical.

### Shaper Selection Guidelines

| Printer Characteristic | Recommended Shapers |
|------------------------|-------------------|
| Simple, well-built frame | `zv`, `mzv`, `smooth` |
| Multiple resonance peaks | `multi_freq`, `ulv`, `3hump_ei` |
| Strong X-Y coupling | `ei`, `2hump_ei`, `adaptive_ei` |
| Variable mechanical properties | `adaptive_ei`, `smooth` |
| Maximum quality needed | `ulv`, `3hump_ei` |

## Enhanced Commands

### COMPREHENSIVE_RESONANCE_TEST

Performs a complete analysis of your printer's resonance characteristics:

```
COMPREHENSIVE_RESONANCE_TEST [AXIS=<axis>] [CHIPS=<chip_names>] [NAME=<name>]
```

This command:
1. Tests resonances at multiple points across the bed
2. Analyzes frequency content, harmonics, and cross-axis coupling  
3. Provides intelligent shaper recommendations
4. Generates detailed analysis reports
5. Compares performance of all available shapers

**Parameters:**
- `AXIS`: Test specific axis ('x', 'y') or both (default: both)
- `CHIPS`: Accelerometer chips to use (default: configured chips)
- `NAME`: Suffix for output files (default: timestamp)

### Enhanced SHAPER_CALIBRATE

The standard `SHAPER_CALIBRATE` command now supports additional options:

```
SHAPER_CALIBRATE [AXIS=<axis>] [COMPREHENSIVE=<0|1>] [MULTI_POINT=<0|1>] [MAX_SMOOTHING=<value>]
```

**New Parameters:**
- `COMPREHENSIVE=1`: Use intelligent recommendations with all advanced shapers
- `MULTI_POINT=1`: Test at multiple bed positions for spatial analysis  
- `MAX_SMOOTHING`: Maximum acceptable smoothing factor

## Advanced Analysis Features

### Peak Detection and Harmonic Analysis

The system automatically detects:
- Primary resonance frequencies
- Harmonic frequencies (2x, 3x, 4x, 5x of fundamentals)
- Secondary resonance peaks
- Frequency spread and stability

### Cross-Axis Coupling Analysis

Measures how X and Y axis motions affect each other:
- Coupling strength (0.0 = independent, 1.0 = fully coupled)
- Frequencies where coupling is strongest
- Recommendations for coupled vs. independent tuning

### Spatial Variation Mapping

When using `MULTI_POINT=1`, the system analyzes:
- How resonance frequencies change across the bed
- Amplitude variations by position
- Mechanical integrity assessment
- Warnings for excessive spatial variation

### Quality Metrics

The system provides comprehensive quality metrics:
- **Noise floor**: Background vibration level
- **Dynamic range**: Signal-to-noise ratio
- **Frequency resolution**: Measurement precision
- **Frequency centroid**: Weighted average of all resonances

## Interpreting Results

### Analysis Output Files

The comprehensive test generates several output files:

1. **`comprehensive_data_*.csv`**: Frequency response data with all shaper responses
2. **`comprehensive_analysis_*.json`**: Detailed analysis in machine-readable format  
3. **`comprehensive_analysis_*.txt`**: Human-readable analysis summary

### Understanding Recommendations

The intelligent recommendation system considers:

- **Simple patterns**: Few, well-separated peaks → Efficient shapers (`zv`, `mzv`, `smooth`)
- **Complex patterns**: Multiple peaks or harmonics → Advanced shapers (`multi_freq`, `ulv`)  
- **High coupling**: Strong X-Y interaction → Robust shapers (`ei`, `adaptive_ei`)
- **Variable characteristics**: Inconsistent measurements → Adaptive shapers (`adaptive_ei`)

### Performance Comparison

Results include a ranking of all tested shapers by:
- **Score**: Overall performance metric balancing vibration reduction and smoothing
- **Vibration reduction**: Percentage of vibrations eliminated
- **Smoothing**: Motion smoothing factor (lower is better for speed)
- **Max acceleration**: Recommended acceleration limit for the shaper

## Configuration Examples

### Basic Enhanced Configuration

```ini
[input_shaper]
shaper_type_x: adaptive_ei
shaper_freq_x: 42.3
shaper_type_y: smooth  
shaper_freq_y: 38.7
```

### High-Quality Configuration

```ini
[input_shaper]
shaper_type_x: ulv
shaper_freq_x: 45.2
shaper_type_y: ulv
shaper_freq_y: 41.8
```

### Multi-Frequency Configuration

```ini
[input_shaper]
shaper_type_x: multi_freq
shaper_freq_x: 39.1
shaper_type_y: multi_freq
shaper_freq_y: 43.6
```

## Troubleshooting

### High Spatial Variation Warning

If the multi-point test shows >10% frequency variation:
1. Check belt tension across the entire range of motion
2. Verify frame rigidity and joint tightness
3. Look for worn linear bearings or loose pulleys
4. Consider bed-specific tuning for large printers

### Complex Resonance Patterns

For multiple strong peaks or harmonics:
1. Use `multi_freq` or `ulv` shapers
2. Consider mechanical modifications to eliminate sources
3. Check for loose components or sympathetic vibrations
4. Review acceleration and jerk settings

### Poor Cross-Axis Coupling

High coupling (>0.6) may indicate:
1. Frame flexing under acceleration
2. Shared drive mechanisms (H-bot, CoreXY) needing tuning
3. Bed or gantry resonances affecting both axes
4. Consider using the same shaper for both axes

## Best Practices

1. **Start with comprehensive analysis**: Use `COMPREHENSIVE_RESONANCE_TEST` to understand your printer
2. **Validate across the bed**: Use multi-point testing for large printers or concerning spatial variation
3. **Regular monitoring**: Re-test after mechanical changes or maintenance
4. **Balance performance**: Consider your print quality vs. speed requirements when choosing shapers
5. **Document changes**: Keep records of what mechanical changes affect resonance patterns

## Compatibility

All enhanced features are backward compatible with existing configurations. Standard shaper commands continue to work exactly as before, with the new features available as opt-in enhancements.

The system gracefully handles missing optional dependencies (scipy) by falling back to simpler analysis methods while still providing enhanced functionality.