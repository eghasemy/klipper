# HTML Report Generation for Input Shaper Calibration

This document describes the new HTML report generation feature that provides comprehensive visual analysis and recommendations for Klipper input shaper calibration.

## Overview

The HTML report generation feature transforms raw calibration data into an interactive, educational web-based report that helps users:

- **Understand** their printer's resonance patterns visually
- **Learn** what each recommendation means and why it matters
- **Improve** their printer's mechanical setup with specific guidance
- **Troubleshoot** common vibration issues
- **Compare** different input shaper configurations

## Features

### 📊 Interactive Visualizations
- **Frequency Response Charts**: Interactive plots showing X, Y, Z, and combined frequency responses
- **Shaper Comparison Charts**: Visual comparison of different input shapers' performance
- **Advanced Metrics Visualization**: Quality indicators and calibration metrics

### 🎯 Intelligent Analysis
- **Peak Frequency Detection**: Identifies and categorizes resonance sources
- **Cross-Axis Coupling Analysis**: Detects frame flexibility and belt tension issues
- **Harmonic Analysis**: Identifies nonlinear mechanical behavior
- **Quality Metrics**: Evaluates calibration data quality and reliability

### 💡 Educational Content
- **Recommendations Explained**: Clear explanations of why specific settings are recommended
- **Mechanical Improvements**: Specific suggestions for hardware upgrades
- **Troubleshooting Guide**: Common issues and their solutions
- **Configuration Examples**: Ready-to-use printer.cfg snippets

### 🔧 User-Friendly Design
- **Professional Styling**: Clean, modern interface with good typography
- **Responsive Layout**: Works on desktop and mobile devices
- **Progressive Enhancement**: Works without JavaScript for basic content
- **Accessibility**: Screen reader friendly with proper semantic markup

## Usage

### Method 1: G-code Command (Recommended)

1. **Run calibration** as usual:
   ```gcode
   SHAPER_CALIBRATE
   ```

2. **Generate HTML report** from the calibration data:
   ```gcode
   GENERATE_SHAPER_REPORT INPUT=/tmp/calibration_data_x_20240722_235800.csv OUTPUT=/tmp/resonance_report.html
   ```

3. **Open the HTML file** in any web browser to view the comprehensive analysis.

### Method 2: Command Line Script

```bash
# Generate HTML report alongside traditional graph
python3 scripts/calibrate_shaper.py \
  --html /tmp/resonance_report.html \
  --output /tmp/frequency_response.png \
  /tmp/calibration_data.csv
```

### Method 3: Python API

```python
# For advanced users integrating into custom workflows
from extras import resonance_report

generator = resonance_report.ResonanceReportGenerator(
    calibration_data, shaper_results, recommended_shaper)
generator.generate_html_report("report.html")
```

## Report Sections

### 1. Executive Summary
- Overall printer health assessment
- Key metrics at a glance
- Quick recommendations

### 2. Frequency Response Analysis
- Interactive frequency response charts
- Peak frequency identification and categorization
- Cross-axis coupling analysis
- Harmonic content analysis

### 3. Input Shaper Comparison
- Performance comparison table
- Interactive charts showing vibration reduction vs. smoothing
- Detailed explanation of metrics

### 4. Specific Recommendations
- **Configuration**: Ready-to-use printer.cfg snippets
- **Mechanical**: Hardware improvement suggestions
- **Tuning**: Print settings recommendations

### 5. Troubleshooting Guide
- Common issues and solutions
- Improvement suggestions based on analysis
- When to re-calibrate

### 6. Advanced Analysis
- Quality metrics visualization
- Frequency domain insights
- Technical details for advanced users

## Benefits

### For Beginners
- **Visual Learning**: See what resonance patterns look like
- **Step-by-Step Guidance**: Clear instructions for improvements
- **Educational Content**: Learn what each setting does
- **Confidence Building**: Understand why recommendations work

### For Advanced Users
- **Detailed Analysis**: Comprehensive frequency domain analysis
- **Quality Assessment**: Evaluate calibration data reliability
- **Comparative Analysis**: Compare multiple calibration runs
- **Integration Ready**: API for custom workflows

### For Service Providers
- **Professional Reports**: Generate client-ready documentation
- **Systematic Analysis**: Consistent evaluation methodology
- **Documentation**: Historical record of calibrations
- **Training Material**: Educational tool for staff

## Technical Implementation

### Architecture
- **Modular Design**: Separate report generator class
- **Template-Based**: Easy to customize styling and content
- **Progressive Enhancement**: Works with and without JavaScript
- **Standards Compliant**: Valid HTML5 with semantic markup

### Dependencies
- **Plotly.js**: Interactive charts (loaded from CDN)
- **Modern CSS**: Grid, Flexbox for responsive layout
- **Web Fonts**: Google Fonts for professional typography

### Performance
- **Optimized Loading**: Efficient data serialization
- **Minimal Dependencies**: Only essential external resources
- **Responsive Images**: Scalable vector graphics
- **Fast Rendering**: Client-side chart generation

## File Structure

```
klippy/extras/
├── resonance_report.py          # Main report generator
├── shaper_calibrate.py          # Enhanced with HTML generation
└── resonance_tester.py          # New G-code command

scripts/
├── calibrate_shaper.py          # Enhanced with --html option
└── ...

examples/
├── demo_html_reports.py         # Demonstration script
└── ...
```

## Configuration

### Default Settings
The HTML report generator uses sensible defaults:
- **Output Format**: Responsive HTML5
- **Chart Library**: Plotly.js (loaded from CDN)
- **Styling**: Professional blue gradient theme
- **Font**: Inter font family for readability

### Customization
The report can be customized by modifying:
- **CSS Styles**: In `_get_css_styles()` method
- **Chart Configuration**: In `_get_javascript()` method
- **Content Sections**: Individual section building methods
- **Analysis Algorithms**: In the CalibrationData class

## Best Practices

### For Optimal Reports
1. **Quality Data**: Ensure good accelerometer mounting
2. **Complete Calibration**: Use sufficient test points and frequency range
3. **Stable Environment**: Minimize external vibrations during testing
4. **Regular Updates**: Re-calibrate after mechanical changes

### For Web Viewing
1. **Modern Browser**: Use Chrome, Firefox, Safari, or Edge
2. **Local Files**: Some browsers may restrict local file access
3. **Network Access**: Required for loading external fonts and chart library
4. **Print-Friendly**: Use browser print function for hard copies

## Examples

### Sample Command Sequence
```gcode
# Complete calibration workflow with HTML report
HOME_ALL
SHAPER_CALIBRATE AXIS=X
SHAPER_CALIBRATE AXIS=Y
GENERATE_SHAPER_REPORT INPUT=/tmp/calibration_data_x_*.csv OUTPUT=/tmp/x_axis_report.html
GENERATE_SHAPER_REPORT INPUT=/tmp/calibration_data_y_*.csv OUTPUT=/tmp/y_axis_report.html
```

### Integration with Macros
```gcode
[gcode_macro CALIBRATE_WITH_REPORT]
gcode:
    SHAPER_CALIBRATE
    GENERATE_SHAPER_REPORT INPUT=/tmp/calibration_data_*.csv OUTPUT=/tmp/latest_report.html
    {action_respond_info("HTML report generated: /tmp/latest_report.html")}
```

## Troubleshooting

### Common Issues

**Report not generating:**
- Check file permissions for output directory
- Ensure numpy is installed: `pip install numpy`
- Verify input file exists and is readable

**Charts not displaying:**
- Check internet connection (for CDN resources)
- Try a different web browser
- Ensure JavaScript is enabled

**Missing data in report:**
- Verify calibration data format
- Check for complete frequency range in input data
- Ensure calibration completed successfully

### Getting Help

If you encounter issues:
1. Check the Klipper logs for error messages
2. Verify your calibration data with existing tools
3. Test with the demonstration script first
4. Report bugs with sample data and error logs

## Future Enhancements

Planned improvements include:
- **Multi-Session Comparison**: Compare calibrations over time
- **Automated Recommendations**: ML-based optimal settings
- **Advanced Visualizations**: 3D surface plots, animations
- **Export Options**: PDF generation, data export
- **Template System**: Customizable report templates
- **Integration**: Direct sharing with support forums

---

This HTML report generation feature significantly enhances the user experience of Klipper's input shaper calibration by making the process more visual, educational, and actionable.