# Dynamic Motion Compensation System
#
# Real-time adaptive input shaping based on movement characteristics
#
# Copyright (C) 2024  Dmitry Butyugin <dmbutyugin@google.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.

import logging, math, time, collections
import numpy as np
from . import shaper_defs

######################################################################
# Real-time Motion Analysis and Compensation
######################################################################

class MotionAnalyzer:
    """Analyzes motion characteristics and recommends compensation adjustments"""
    
    def __init__(self, config):
        self.analysis_window = config.getfloat('analysis_window', 2.0, minval=0.5, maxval=10.0)
        self.update_interval = config.getfloat('update_interval', 0.1, minval=0.05, maxval=1.0)
        
        # Motion history for analysis
        self.motion_history = collections.deque(maxlen=1000)
        self.last_update_time = 0
        
        # Movement pattern detection
        self.current_pattern = None
        self.pattern_confidence = 0.0
        
    def analyze_motion(self, position, velocity, acceleration, time_stamp):
        """Analyze current motion and return recommended compensation parameters"""
        
        # Add to motion history
        self.motion_history.append({
            'time': time_stamp,
            'position': position[:2],  # X, Y only
            'velocity': velocity[:2],
            'acceleration': acceleration[:2]
        })
        
        # Check if enough time has passed for update
        if time_stamp - self.last_update_time < self.update_interval:
            return None
        
        self.last_update_time = time_stamp
        
        # Analyze recent motion
        if len(self.motion_history) < 10:
            return None
        
        # Detect movement pattern
        pattern = self._detect_movement_pattern()
        
        # Calculate motion characteristics
        motion_stats = self._calculate_motion_statistics()
        
        # Generate compensation recommendations
        recommendations = self._generate_compensation_recommendations(pattern, motion_stats)
        
        return recommendations
    
    def _detect_movement_pattern(self):
        """Detect the current movement pattern type"""
        if len(self.motion_history) < 5:
            return 'unknown'
        
        recent_moves = list(self.motion_history)[-20:]  # Last 20 moves
        
        # Calculate direction changes
        direction_changes = 0
        speed_variations = []
        
        for i in range(1, len(recent_moves)):
            prev_vel = recent_moves[i-1]['velocity']
            curr_vel = recent_moves[i]['velocity']
            
            # Check for direction change
            if np.dot(prev_vel, curr_vel) < 0.5 * np.linalg.norm(prev_vel) * np.linalg.norm(curr_vel):
                direction_changes += 1
            
            # Track speed variation
            prev_speed = np.linalg.norm(prev_vel)
            curr_speed = np.linalg.norm(curr_vel)
            if prev_speed > 0:
                speed_variations.append(abs(curr_speed - prev_speed) / prev_speed)
        
        # Classify pattern
        avg_speed_variation = np.mean(speed_variations) if speed_variations else 0
        direction_change_rate = direction_changes / len(recent_moves)
        
        if direction_change_rate > 0.3:
            return 'corner_heavy'  # Lots of direction changes
        elif avg_speed_variation > 0.2:
            return 'variable_speed'  # Variable speed movement
        elif direction_change_rate < 0.1 and avg_speed_variation < 0.1:
            return 'linear'  # Straight line movement
        else:
            return 'mixed'  # Mixed movement pattern
    
    def _calculate_motion_statistics(self):
        """Calculate statistical characteristics of recent motion"""
        if len(self.motion_history) < 5:
            return {}
        
        recent_moves = list(self.motion_history)[-10:]
        
        velocities = [move['velocity'] for move in recent_moves]
        accelerations = [move['acceleration'] for move in recent_moves]
        
        # Calculate statistics
        speeds = [np.linalg.norm(vel) for vel in velocities]
        accel_magnitudes = [np.linalg.norm(acc) for acc in accelerations]
        
        return {
            'avg_speed': np.mean(speeds),
            'max_speed': np.max(speeds),
            'speed_std': np.std(speeds),
            'avg_accel': np.mean(accel_magnitudes),
            'max_accel': np.max(accel_magnitudes),
            'accel_std': np.std(accel_magnitudes)
        }
    
    def _generate_compensation_recommendations(self, pattern, stats):
        """Generate compensation parameter recommendations"""
        recommendations = {
            'x': {'freq_adjustment': 1.0, 'damping_adjustment': 1.0, 'shaper_hint': None},
            'y': {'freq_adjustment': 1.0, 'damping_adjustment': 1.0, 'shaper_hint': None}
        }
        
        if not stats:
            return recommendations
        
        # Pattern-based adjustments
        if pattern == 'corner_heavy':
            # Corners benefit from higher damping and potentially different shaper
            for axis in ['x', 'y']:
                recommendations[axis]['damping_adjustment'] = 1.2
                recommendations[axis]['shaper_hint'] = 'ei'  # Better for corners
                
        elif pattern == 'linear':
            # Linear movements can use faster shapers
            for axis in ['x', 'y']:
                recommendations[axis]['shaper_hint'] = 'smooth'  # Speed-optimized
                
        elif pattern == 'variable_speed':
            # Variable speed benefits from adaptive shapers
            for axis in ['x', 'y']:
                recommendations[axis]['shaper_hint'] = 'adaptive_ei'
        
        # Speed-based adjustments
        if stats['avg_speed'] > 150:
            # High speed - increase frequency slightly
            for axis in ['x', 'y']:
                recommendations[axis]['freq_adjustment'] *= 1.1
                
        elif stats['avg_speed'] < 50:
            # Low speed - can use higher quality shapers
            for axis in ['x', 'y']:
                recommendations[axis]['shaper_hint'] = 'ulv'
        
        # Acceleration-based adjustments
        if stats['avg_accel'] > 5000:
            # High acceleration - need better damping
            for axis in ['x', 'y']:
                recommendations[axis]['damping_adjustment'] *= 1.15
                recommendations[axis]['freq_adjustment'] *= 1.05
        
        return recommendations

class DynamicInputShaper:
    """Dynamic input shaper that adapts in real-time"""
    
    def __init__(self, config):
        self.printer = config.get_printer()
        self.name = config.get_name().split()[-1]
        
        # Configuration
        self.enabled = config.getboolean('enabled', False)
        self.adaptation_rate = config.getfloat('adaptation_rate', 0.1, minval=0.01, maxval=1.0)
        self.min_update_interval = config.getfloat('min_update_interval', 0.5, minval=0.1, maxval=5.0)
        
        # Motion analyzer
        self.motion_analyzer = MotionAnalyzer(config)
        
        # Current adaptive parameters
        self.base_parameters = {'x': None, 'y': None}  # Base parameters from calibration
        self.current_adjustments = {'x': {'freq': 1.0, 'damping': 1.0}, 
                                   'y': {'freq': 1.0, 'damping': 1.0}}
        self.last_adjustment_time = 0
        
        # Parameter transition smoothing
        self.transition_queue = collections.deque()
        self.target_parameters = {'x': None, 'y': None}
        
        # Hook into motion system
        self.printer.register_event_handler("klippy:connect", self._connect)
        
        # Register gcode commands
        gcode = self.printer.lookup_object('gcode')
        gcode.register_command("ENABLE_DYNAMIC_SHAPING", 
                             self.cmd_ENABLE_DYNAMIC_SHAPING,
                             desc=self.cmd_ENABLE_DYNAMIC_SHAPING_help)
        gcode.register_command("SET_BASE_SHAPER_PARAMS",
                             self.cmd_SET_BASE_SHAPER_PARAMS,
                             desc=self.cmd_SET_BASE_SHAPER_PARAMS_help)
        gcode.register_command("GET_DYNAMIC_SHAPER_STATUS",
                             self.cmd_GET_DYNAMIC_SHAPER_STATUS,
                             desc=self.cmd_GET_DYNAMIC_SHAPER_STATUS_help)

    def _connect(self):
        """Connect to motion system for real-time updates"""
        if self.enabled:
            # Hook into toolhead for motion updates
            toolhead = self.printer.lookup_object('toolhead')
            # Note: In real implementation, this would hook into the motion planner
            # For demonstration, we'll use a timer-based approach
            reactor = self.printer.get_reactor()
            reactor.register_timer(self._motion_update_callback, reactor.NOW)

    def _motion_update_callback(self, eventtime):
        """Periodic callback to update dynamic shaping parameters"""
        if not self.enabled:
            return eventtime + 1.0  # Check again in 1 second
        
        try:
            # Get current motion state
            toolhead = self.printer.lookup_object('toolhead')
            position = toolhead.get_position()
            
            # For demonstration, simulate velocity and acceleration
            # In real implementation, these would come from the motion planner
            velocity = [0, 0, 0, 0]  # Placeholder
            acceleration = [0, 0, 0, 0]  # Placeholder
            
            # Analyze motion and get recommendations
            recommendations = self.motion_analyzer.analyze_motion(
                position, velocity, acceleration, eventtime)
            
            if recommendations:
                self._apply_dynamic_adjustments(recommendations, eventtime)
                
        except Exception as e:
            logging.warning(f"Dynamic shaper update failed: {e}")
        
        return eventtime + self.motion_analyzer.update_interval

    def _apply_dynamic_adjustments(self, recommendations, eventtime):
        """Apply dynamic parameter adjustments with smooth transitions"""
        if eventtime - self.last_adjustment_time < self.min_update_interval:
            return
        
        self.last_adjustment_time = eventtime
        
        # Calculate new target parameters
        for axis in ['x', 'y']:
            if self.base_parameters[axis] is None:
                continue
            
            base_shaper, base_freq, base_damping = self.base_parameters[axis]
            rec = recommendations[axis]
            
            # Apply adjustments
            new_freq = base_freq * rec['freq_adjustment']
            new_damping = base_damping * rec['damping_adjustment']
            new_shaper = rec['shaper_hint'] or base_shaper
            
            # Smooth transition to new parameters
            self._queue_parameter_transition(axis, new_shaper, new_freq, new_damping)

    def _queue_parameter_transition(self, axis, shaper_type, freq, damping):
        """Queue a smooth transition to new parameters"""
        current_time = time.time()
        
        # Calculate transition steps
        transition_time = 0.5  # 500ms transition
        num_steps = 10
        step_time = transition_time / num_steps
        
        if self.target_parameters[axis]:
            current_freq = self.target_parameters[axis][1]
            current_damping = self.target_parameters[axis][2]
        else:
            current_freq = freq
            current_damping = damping
        
        # Create transition steps
        for i in range(num_steps + 1):
            t = i / num_steps
            # Smooth transition using cosine interpolation
            smooth_t = (1 - math.cos(t * math.pi)) / 2
            
            step_freq = current_freq + (freq - current_freq) * smooth_t
            step_damping = current_damping + (damping - current_damping) * smooth_t
            
            self.transition_queue.append({
                'time': current_time + i * step_time,
                'axis': axis,
                'shaper_type': shaper_type,
                'freq': step_freq,
                'damping': step_damping
            })
        
        self.target_parameters[axis] = (shaper_type, freq, damping)

    cmd_ENABLE_DYNAMIC_SHAPING_help = "Enable or disable dynamic input shaping"
    def cmd_ENABLE_DYNAMIC_SHAPING(self, gcmd):
        """Enable or disable dynamic input shaping"""
        self.enabled = gcmd.get_int('ENABLE', 1, minval=0, maxval=1)
        
        if self.enabled:
            gcmd.respond_info("Dynamic input shaping enabled")
        else:
            gcmd.respond_info("Dynamic input shaping disabled")

    cmd_SET_BASE_SHAPER_PARAMS_help = "Set base shaper parameters for dynamic adaptation"
    def cmd_SET_BASE_SHAPER_PARAMS(self, gcmd):
        """Set base shaper parameters"""
        for axis in ['x', 'y']:
            axis_upper = axis.upper()
            shaper_type = gcmd.get(f'SHAPER_TYPE_{axis_upper}', None)
            freq = gcmd.get_float(f'SHAPER_FREQ_{axis_upper}', None)
            damping = gcmd.get_float(f'DAMPING_RATIO_{axis_upper}', 0.1)
            
            if shaper_type and freq:
                self.base_parameters[axis] = (shaper_type.lower(), freq, damping)
                gcmd.respond_info(f"Set base {axis}-axis: {shaper_type} @ {freq}Hz, damping {damping}")

    cmd_GET_DYNAMIC_SHAPER_STATUS_help = "Get current dynamic shaper status"
    def cmd_GET_DYNAMIC_SHAPER_STATUS(self, gcmd):
        """Get current dynamic shaper status"""
        gcmd.respond_info(f"Dynamic shaping enabled: {self.enabled}")
        
        for axis in ['x', 'y']:
            if self.base_parameters[axis]:
                base = self.base_parameters[axis]
                current = self.current_adjustments[axis]
                gcmd.respond_info(f"{axis.upper()}-axis base: {base[0]} @ {base[1]:.1f}Hz")
                gcmd.respond_info(f"{axis.upper()}-axis adjustments: freq×{current['freq']:.2f}, damping×{current['damping']:.2f}")

def load_config(config):
    return DynamicInputShaper(config)