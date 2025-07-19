# Example G-code for testing Modbus VFD spindle

# Test basic spindle commands
M3 S1000    ; Start spindle clockwise at 1000 RPM
G4 P2       ; Wait 2 seconds
M3 S5000    ; Change speed to 5000 RPM  
G4 P2       ; Wait 2 seconds
M5          ; Stop spindle
G4 P1       ; Wait 1 second

# Test direction changes
M3 S2000    ; Start clockwise at 2000 RPM
G4 P2       ; Wait 2 seconds
M5          ; Stop
G4 P0.5     ; Brief pause
M4 S2000    ; Start counter-clockwise at 2000 RPM
G4 P2       ; Wait 2 seconds  
M5          ; Stop spindle

# Test with CNC workflow
G54                    ; Select work coordinate system
M3 S12000             ; Start spindle at 12000 RPM
M8                    ; Turn on flood coolant (if configured)
G0 X10 Y10            ; Rapid move to position
G1 Z-1 F200           ; Feed into material
G1 X20 F300           ; Cut move
G0 Z5                 ; Retract
M5                    ; Stop spindle
M9                    ; Turn off coolant