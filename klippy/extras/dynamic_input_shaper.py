# Dynamic Motion Compensation System
#
# Real-time adaptive input shaping based on movement characteristics
#
# Copyright (C) 2024  Dmitry Butyugin <dmbutyugin@google.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.

import logging, math, time, collections, re, pickle, os
import numpy as np
from . import shaper_defs

# Machine learning imports (with fallbacks)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    ML_AVAILABLE = True
except ImportError:
    logging.warning("scikit-learn not available. ML-based pattern recognition will be disabled.")
    ML_AVAILABLE = False

######################################################################
# Machine Learning-based Pattern Recognition
######################################################################

class MLPatternRecognizer:
    """Machine learning-based pattern recognition for motion analysis"""
    
    def __init__(self, config, printer):
        self.printer = printer
        self.enabled = ML_AVAILABLE and config.getboolean('enable_ml_recognition', True)
        
        # Model configuration
        self.model_path = config.get('ml_model_path', '/tmp/klipper_pattern_model.pkl')
        self.training_data_path = config.get('training_data_path', '/tmp/klipper_training_data.pkl')
        
        # Training parameters
        self.min_training_samples = config.getint('min_training_samples', 1000, minval=100)
        self.auto_retrain_interval = config.getfloat('auto_retrain_hours', 24.0, minval=1.0) * 3600
        self.feature_collection_enabled = config.getboolean('collect_training_data', True)
        
        # Models
        self.pattern_classifier = None
        self.motion_clusterer = None
        self.feature_scaler = StandardScaler() if ML_AVAILABLE else None
        
        # Training data collection
        self.training_data = {
            'features': [],
            'labels': [],
            'timestamps': [],
            'effectiveness_scores': []
        }
        
        # Pattern types discovered by ML
        self.discovered_patterns = set()
        self.pattern_effectiveness = {}  # Track effectiveness of compensation for each pattern
        
        # Load existing models and data
        self._load_models()
        self._load_training_data()
        
        # Last training time
        self.last_training_time = time.time()
        
        if not self.enabled:
            logging.info("ML pattern recognition disabled (sklearn not available or disabled in config)")
    
    def _extract_motion_features(self, motion_history):
        """Extract feature vector from motion history for ML analysis"""
        if len(motion_history) < 10:
            return None
        
        recent_moves = list(motion_history)[-20:]  # Last 20 moves
        
        # Feature extraction
        features = []
        
        # Velocity statistics
        velocities = [np.array(move['velocity']) for move in recent_moves]
        speeds = [np.linalg.norm(vel) for vel in velocities]
        
        features.extend([
            np.mean(speeds),           # Average speed
            np.std(speeds),            # Speed variation
            np.max(speeds),            # Peak speed
            np.min(speeds),            # Minimum speed
            np.percentile(speeds, 90), # 90th percentile speed
        ])
        
        # Acceleration statistics
        accelerations = [np.array(move['acceleration']) for move in recent_moves]
        accel_magnitudes = [np.linalg.norm(acc) for acc in accelerations]
        
        features.extend([
            np.mean(accel_magnitudes),           # Average acceleration
            np.std(accel_magnitudes),            # Acceleration variation
            np.max(accel_magnitudes),            # Peak acceleration
            np.percentile(accel_magnitudes, 90), # 90th percentile acceleration
        ])
        
        # Direction change analysis
        direction_changes = 0
        angle_changes = []
        
        for i in range(1, len(velocities)):
            prev_vel = velocities[i-1]
            curr_vel = velocities[i]
            
            prev_speed = np.linalg.norm(prev_vel)
            curr_speed = np.linalg.norm(curr_vel)
            
            if prev_speed > 1.0 and curr_speed > 1.0:  # Ignore very slow moves
                # Calculate angle between velocity vectors
                cos_angle = np.dot(prev_vel, curr_vel) / (prev_speed * curr_speed)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
                angle = np.arccos(cos_angle)
                angle_changes.append(angle)
                
                if angle > math.pi / 4:  # 45 degree threshold
                    direction_changes += 1
        
        features.extend([
            direction_changes / len(recent_moves),      # Direction change rate
            np.mean(angle_changes) if angle_changes else 0,  # Average angle change
            np.std(angle_changes) if angle_changes else 0,   # Angle change variation
        ])
        
        # Movement patterns
        # Detect periodic motion (like infill)
        x_positions = [move['position'][0] for move in recent_moves]
        y_positions = [move['position'][1] for move in recent_moves]
        
        # Calculate movement ranges
        x_range = np.max(x_positions) - np.min(x_positions)
        y_range = np.max(y_positions) - np.min(y_positions)
        
        features.extend([
            x_range,                    # X movement range
            y_range,                    # Y movement range
            x_range / (y_range + 0.1),  # Aspect ratio of movement
        ])
        
        # Speed pattern analysis
        speed_changes = []
        for i in range(1, len(speeds)):
            if speeds[i-1] > 0:
                speed_changes.append((speeds[i] - speeds[i-1]) / speeds[i-1])
        
        features.extend([
            np.mean(speed_changes) if speed_changes else 0,  # Average speed change rate
            np.std(speed_changes) if speed_changes else 0,   # Speed change variation
        ])
        
        # Frequency domain features (detect repetitive patterns)
        if len(speeds) > 5:
            # Simple FFT-based features
            fft = np.fft.fft(speeds)
            fft_mag = np.abs(fft)
            
            features.extend([
                np.max(fft_mag[1:]),            # Dominant frequency magnitude
                np.argmax(fft_mag[1:]) + 1,     # Dominant frequency index
                np.sum(fft_mag[1:5]) / np.sum(fft_mag),  # Low frequency energy ratio
            ])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)
    
    def predict_pattern(self, motion_history, gcode_context=None):
        """Predict motion pattern using ML model"""
        if not self.enabled or self.pattern_classifier is None:
            return self._fallback_pattern_detection(motion_history)
        
        features = self._extract_motion_features(motion_history)
        if features is None:
            return 'unknown'
        
        try:
            # Scale features
            features_scaled = self.feature_scaler.transform([features])
            
            # Predict pattern
            pattern_probs = self.pattern_classifier.predict_proba(features_scaled)[0]
            pattern_classes = self.pattern_classifier.classes_
            
            # Get most confident prediction
            best_idx = np.argmax(pattern_probs)
            confidence = pattern_probs[best_idx]
            predicted_pattern = pattern_classes[best_idx]
            
            # Also try clustering for novel pattern detection
            if self.motion_clusterer is not None:
                cluster = self.motion_clusterer.predict(features_scaled)[0]
                
                # Check if this is a novel cluster
                if f'cluster_{cluster}' not in self.discovered_patterns:
                    self.discovered_patterns.add(f'cluster_{cluster}')
                    logging.info(f"Discovered new motion pattern: cluster_{cluster}")
            
            # Collect training data if enabled
            if self.feature_collection_enabled and gcode_context:
                self._collect_training_sample(features, gcode_context, predicted_pattern)
            
            return {
                'pattern': predicted_pattern,
                'confidence': confidence,
                'alternatives': [(pattern_classes[i], pattern_probs[i]) 
                               for i in np.argsort(pattern_probs)[-3:][::-1]]
            }
            
        except Exception as e:
            logging.warning(f"ML pattern prediction failed: {e}")
            return self._fallback_pattern_detection(motion_history)
    
    def _fallback_pattern_detection(self, motion_history):
        """Fallback rule-based pattern detection when ML is unavailable"""
        if len(motion_history) < 5:
            return 'unknown'
        
        recent_moves = list(motion_history)[-20:]
        
        # Simple rule-based classification
        direction_changes = 0
        speed_variations = []
        
        for i in range(1, len(recent_moves)):
            prev_vel = recent_moves[i-1]['velocity']
            curr_vel = recent_moves[i]['velocity']
            
            if np.dot(prev_vel, curr_vel) < 0.5 * np.linalg.norm(prev_vel) * np.linalg.norm(curr_vel):
                direction_changes += 1
            
            prev_speed = np.linalg.norm(prev_vel)
            curr_speed = np.linalg.norm(curr_vel)
            if prev_speed > 0:
                speed_variations.append(abs(curr_speed - prev_speed) / prev_speed)
        
        avg_speed_variation = np.mean(speed_variations) if speed_variations else 0
        direction_change_rate = direction_changes / len(recent_moves)
        
        if direction_change_rate > 0.3:
            return 'corner_heavy'
        elif avg_speed_variation > 0.2:
            return 'variable_speed'
        elif direction_change_rate < 0.1 and avg_speed_variation < 0.1:
            return 'linear'
        else:
            return 'mixed'
    
    def _collect_training_sample(self, features, gcode_context, predicted_pattern):
        """Collect training sample for model improvement"""
        current_time = time.time()
        
        # Determine ground truth label from G-code context
        true_label = self._determine_ground_truth(gcode_context, predicted_pattern)
        
        self.training_data['features'].append(features)
        self.training_data['labels'].append(true_label)
        self.training_data['timestamps'].append(current_time)
        self.training_data['effectiveness_scores'].append(1.0)  # Will be updated later
        
        # Periodic training check
        if (current_time - self.last_training_time > self.auto_retrain_interval and 
            len(self.training_data['features']) >= self.min_training_samples):
            self._retrain_model()
    
    def _determine_ground_truth(self, gcode_context, predicted_pattern):
        """Determine ground truth label from G-code context"""
        if not gcode_context:
            return predicted_pattern
        
        # Extract feature type from G-code comment
        if 'TYPE:' in gcode_context:
            feature_type = gcode_context.split('TYPE:')[1].split()[0].upper()
            
            # Map feature types to motion patterns
            feature_pattern_map = {
                'WALL-OUTER': 'perimeter',
                'WALL-INNER': 'perimeter', 
                'PERIMETER': 'perimeter',
                'INFILL': 'infill',
                'SPARSE-INFILL': 'infill',
                'SOLID-INFILL': 'solid_infill',
                'SUPPORT': 'support',
                'BRIDGE': 'bridge',
                'TOP-SURFACE': 'top_surface',
                'SKIRT': 'skirt',
                'BRIM': 'brim'
            }
            
            return feature_pattern_map.get(feature_type, predicted_pattern)
        
        return predicted_pattern
    
    def train_model(self, force_retrain=False):
        """Train or retrain the ML models"""
        if not self.enabled:
            logging.warning("Cannot train ML model: sklearn not available")
            return False
        
        if len(self.training_data['features']) < self.min_training_samples and not force_retrain:
            logging.info(f"Not enough training samples ({len(self.training_data['features'])}/{self.min_training_samples})")
            return False
        
        try:
            features = np.array(self.training_data['features'])
            labels = np.array(self.training_data['labels'])
            
            # Split data for training and validation
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Scale features
            self.feature_scaler.fit(X_train)
            X_train_scaled = self.feature_scaler.transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train pattern classifier
            self.pattern_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            self.pattern_classifier.fit(X_train_scaled, y_train)
            
            # Evaluate classifier
            y_pred = self.pattern_classifier.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logging.info(f"Pattern classifier trained with {accuracy:.3f} accuracy")
            logging.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
            
            # Train motion clusterer for novel pattern discovery
            self.motion_clusterer = KMeans(n_clusters=8, random_state=42)
            self.motion_clusterer.fit(X_train_scaled)
            
            # Save models
            self._save_models()
            
            self.last_training_time = time.time()
            return True
            
        except Exception as e:
            logging.error(f"Model training failed: {e}")
            return False
    
    def _retrain_model(self):
        """Automatically retrain model with new data"""
        logging.info("Starting automatic model retraining...")
        success = self.train_model()
        if success:
            logging.info("Model retrained successfully")
        else:
            logging.warning("Model retraining failed")
    
    def update_effectiveness_score(self, pattern, effectiveness_score):
        """Update effectiveness score for a pattern to improve future training"""
        if pattern in self.pattern_effectiveness:
            # Exponential moving average
            self.pattern_effectiveness[pattern] = (
                0.8 * self.pattern_effectiveness[pattern] + 
                0.2 * effectiveness_score
            )
        else:
            self.pattern_effectiveness[pattern] = effectiveness_score
        
        # Update recent training samples
        for i in range(len(self.training_data['labels'])):
            if self.training_data['labels'][i] == pattern:
                # Update effectiveness score for recent samples
                age = time.time() - self.training_data['timestamps'][i]
                if age < 3600:  # Within last hour
                    self.training_data['effectiveness_scores'][i] = effectiveness_score
    
    def _save_models(self):
        """Save trained models to disk"""
        if not self.enabled:
            return
        
        try:
            model_data = {
                'pattern_classifier': self.pattern_classifier,
                'motion_clusterer': self.motion_clusterer,
                'feature_scaler': self.feature_scaler,
                'discovered_patterns': self.discovered_patterns,
                'pattern_effectiveness': self.pattern_effectiveness,
                'last_training_time': self.last_training_time
            }
            
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            logging.info(f"ML models saved to {self.model_path}")
            
        except Exception as e:
            logging.warning(f"Failed to save ML models: {e}")
    
    def _load_models(self):
        """Load trained models from disk"""
        if not self.enabled or not os.path.exists(self.model_path):
            return
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.pattern_classifier = model_data.get('pattern_classifier')
            self.motion_clusterer = model_data.get('motion_clusterer')
            self.feature_scaler = model_data.get('feature_scaler', StandardScaler())
            self.discovered_patterns = model_data.get('discovered_patterns', set())
            self.pattern_effectiveness = model_data.get('pattern_effectiveness', {})
            self.last_training_time = model_data.get('last_training_time', time.time())
            
            logging.info(f"ML models loaded from {self.model_path}")
            
        except Exception as e:
            logging.warning(f"Failed to load ML models: {e}")
    
    def _save_training_data(self):
        """Save training data to disk"""
        if not self.feature_collection_enabled:
            return
        
        try:
            os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
            with open(self.training_data_path, 'wb') as f:
                pickle.dump(self.training_data, f)
                
        except Exception as e:
            logging.warning(f"Failed to save training data: {e}")
    
    def _load_training_data(self):
        """Load training data from disk"""
        if not self.feature_collection_enabled or not os.path.exists(self.training_data_path):
            return
        
        try:
            with open(self.training_data_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            # Merge with existing data
            for key in self.training_data:
                if key in saved_data:
                    self.training_data[key].extend(saved_data[key])
            
            logging.info(f"Loaded {len(self.training_data['features'])} training samples")
            
        except Exception as e:
            logging.warning(f"Failed to load training data: {e}")

######################################################################
# Predictive Compensation System
######################################################################

class PredictiveCompensator:
    """Predictive compensation based on upcoming G-code commands"""
    
    def __init__(self, config, printer):
        self.printer = printer
        self.enabled = config.getboolean('enable_predictive_compensation', True)
        
        # Lookahead configuration
        self.lookahead_time = config.getfloat('lookahead_time', 2.0, minval=0.5, maxval=10.0)
        self.lookahead_commands = config.getint('lookahead_commands', 50, minval=10, maxval=200)
        self.prediction_interval = config.getfloat('prediction_interval', 0.2, minval=0.1, maxval=1.0)
        
        # G-code buffer and analysis
        self.gcode_buffer = collections.deque(maxlen=self.lookahead_commands)
        self.upcoming_moves = collections.deque()
        self.move_predictions = {}
        
        # Motion prediction models
        self.move_pattern_history = collections.deque(maxlen=1000)
        self.transition_models = {}  # Models for predicting motion at pattern transitions
        
        # Compensation pre-calculation
        self.precomputed_compensation = {}
        self.compensation_schedule = collections.deque()
        
        # Hook into G-code processing for lookahead
        if self.enabled:
            gcode = self.printer.lookup_object('gcode')
            self.original_run_script = gcode.run_script
            gcode.run_script = self._enhanced_run_script
            
            # Start prediction timer
            reactor = self.printer.get_reactor()
            reactor.register_timer(self._prediction_callback, reactor.NOW)
        
        if not self.enabled:
            logging.info("Predictive compensation disabled")
    
    def _enhanced_run_script(self, script):
        """Enhanced G-code script runner that captures commands for lookahead"""
        # Parse and buffer upcoming commands
        lines = script.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(';'):
                self._buffer_gcode_command(line)
        
        # Continue with normal processing
        return self.original_run_script(script)
    
    def _buffer_gcode_command(self, command):
        """Buffer G-code command for lookahead analysis"""
        current_time = time.time()
        
        # Parse command
        parsed_cmd = self._parse_gcode_command(command)
        if parsed_cmd:
            self.gcode_buffer.append({
                'time': current_time,
                'command': command,
                'parsed': parsed_cmd,
                'predicted_motion': None,
                'compensation_needed': None
            })
    
    def _parse_gcode_command(self, command):
        """Parse G-code command to extract motion information"""
        cmd_parts = command.upper().split()
        if not cmd_parts:
            return None
        
        cmd_type = cmd_parts[0]
        
        # Only interested in motion commands
        if cmd_type not in ['G0', 'G1', 'G2', 'G3']:
            return None
        
        parsed = {
            'type': cmd_type,
            'coordinates': {},
            'feedrate': None,
            'is_extrusion': False
        }
        
        # Extract coordinates and parameters
        for part in cmd_parts[1:]:
            if part.startswith('X'):
                parsed['coordinates']['X'] = float(part[1:])
            elif part.startswith('Y'):
                parsed['coordinates']['Y'] = float(part[1:])
            elif part.startswith('Z'):
                parsed['coordinates']['Z'] = float(part[1:])
            elif part.startswith('E'):
                parsed['coordinates']['E'] = float(part[1:])
                parsed['is_extrusion'] = True
            elif part.startswith('F'):
                parsed['feedrate'] = float(part[1:])
        
        return parsed
    
    def _prediction_callback(self, eventtime):
        """Periodic callback to analyze upcoming commands and predict compensation needs"""
        if not self.enabled:
            return eventtime + 1.0
        
        try:
            # Analyze buffered commands
            self._analyze_upcoming_commands()
            
            # Predict motion characteristics
            self._predict_motion_patterns()
            
            # Pre-calculate compensation parameters
            self._precompute_compensation()
            
            # Apply scheduled compensation updates
            self._apply_scheduled_compensation(eventtime)
            
        except Exception as e:
            logging.warning(f"Predictive compensation update failed: {e}")
        
        return eventtime + self.prediction_interval
    
    def _analyze_upcoming_commands(self):
        """Analyze upcoming G-code commands to predict motion patterns"""
        if len(self.gcode_buffer) < 5:
            return
        
        # Look at recent commands to detect patterns
        recent_commands = list(self.gcode_buffer)[-20:]
        
        # Detect motion sequences
        motion_sequences = self._identify_motion_sequences(recent_commands)
        
        # Predict motion characteristics for each sequence
        for sequence in motion_sequences:
            motion_prediction = self._predict_sequence_motion(sequence)
            
            # Store predictions
            for cmd in sequence:
                cmd['predicted_motion'] = motion_prediction
                cmd['compensation_needed'] = self._determine_compensation_needs(motion_prediction)
    
    def _identify_motion_sequences(self, commands):
        """Identify coherent motion sequences in G-code commands"""
        sequences = []
        current_sequence = []
        
        for cmd in commands:
            if not cmd['parsed']:
                continue
            
            # Check if this command continues the current sequence
            if self._is_sequence_continuation(current_sequence, cmd):
                current_sequence.append(cmd)
            else:
                # End current sequence and start new one
                if len(current_sequence) >= 2:
                    sequences.append(current_sequence.copy())
                current_sequence = [cmd]
        
        # Add final sequence
        if len(current_sequence) >= 2:
            sequences.append(current_sequence)
        
        return sequences
    
    def _is_sequence_continuation(self, current_sequence, new_cmd):
        """Check if a command continues the current motion sequence"""
        if not current_sequence:
            return True
        
        last_cmd = current_sequence[-1]
        
        # Check for pattern consistency
        last_parsed = last_cmd['parsed']
        new_parsed = new_cmd['parsed']
        
        # Must be same command type
        if last_parsed['type'] != new_parsed['type']:
            return False
        
        # Check for consistent motion pattern
        # (This is simplified - real implementation would be more sophisticated)
        
        # Check coordinate consistency
        last_coords = set(last_parsed['coordinates'].keys())
        new_coords = set(new_parsed['coordinates'].keys())
        
        # Similar coordinates involved
        if len(last_coords.intersection(new_coords)) < 2:
            return False
        
        # Check feedrate consistency
        if (last_parsed['feedrate'] and new_parsed['feedrate'] and 
            abs(last_parsed['feedrate'] - new_parsed['feedrate']) > 0.1 * last_parsed['feedrate']):
            return False
        
        return True
    
    def _predict_sequence_motion(self, sequence):
        """Predict motion characteristics for a command sequence"""
        if len(sequence) < 2:
            return None
        
        # Analyze sequence to predict motion
        coordinates_involved = set()
        feedrates = []
        distances = []
        direction_changes = 0
        
        prev_direction = None
        
        for i, cmd in enumerate(sequence):
            parsed = cmd['parsed']
            coordinates_involved.update(parsed['coordinates'].keys())
            
            if parsed['feedrate']:
                feedrates.append(parsed['feedrate'])
            
            # Calculate direction and distance (simplified)
            if i > 0:
                prev_cmd = sequence[i-1]['parsed']
                
                # Calculate distance
                distance = 0
                for coord in ['X', 'Y', 'Z']:
                    if coord in parsed['coordinates'] and coord in prev_cmd['coordinates']:
                        distance += (parsed['coordinates'][coord] - prev_cmd['coordinates'][coord]) ** 2
                distance = math.sqrt(distance)
                distances.append(distance)
                
                # Calculate direction change
                if prev_direction is not None:
                    # Simplified direction calculation
                    curr_direction = self._calculate_direction(prev_cmd, parsed)
                    if curr_direction and abs(curr_direction - prev_direction) > math.pi / 6:  # 30 degrees
                        direction_changes += 1
                    prev_direction = curr_direction
        
        # Predict motion characteristics
        prediction = {
            'sequence_length': len(sequence),
            'coordinates_involved': list(coordinates_involved),
            'avg_feedrate': np.mean(feedrates) if feedrates else 100,
            'feedrate_variation': np.std(feedrates) if len(feedrates) > 1 else 0,
            'avg_distance': np.mean(distances) if distances else 1,
            'total_distance': sum(distances),
            'direction_changes': direction_changes,
            'direction_change_rate': direction_changes / len(sequence) if sequence else 0,
            'predicted_pattern': self._classify_sequence_pattern(sequence),
            'resonance_risk': self._estimate_resonance_risk(sequence),
            'recommended_compensation': self._recommend_sequence_compensation(sequence)
        }
        
        return prediction
    
    def _calculate_direction(self, prev_cmd, curr_cmd):
        """Calculate movement direction (simplified 2D calculation)"""
        try:
            dx = curr_cmd['coordinates'].get('X', 0) - prev_cmd['coordinates'].get('X', 0)
            dy = curr_cmd['coordinates'].get('Y', 0) - prev_cmd['coordinates'].get('Y', 0)
            
            if abs(dx) < 0.01 and abs(dy) < 0.01:
                return None
            
            return math.atan2(dy, dx)
        except:
            return None
    
    def _classify_sequence_pattern(self, sequence):
        """Classify the motion pattern of a sequence"""
        if len(sequence) < 3:
            return 'simple'
        
        # Analyze sequence characteristics
        distances = []
        direction_changes = 0
        feedrate_changes = 0
        
        for i in range(1, len(sequence)):
            prev_parsed = sequence[i-1]['parsed']
            curr_parsed = sequence[i]['parsed']
            
            # Calculate distance
            distance = 0
            for coord in ['X', 'Y']:
                if coord in curr_parsed['coordinates'] and coord in prev_parsed['coordinates']:
                    distance += (curr_parsed['coordinates'][coord] - prev_parsed['coordinates'][coord]) ** 2
            distances.append(math.sqrt(distance))
            
            # Check for direction change
            if i > 1:
                # Simplified direction change detection
                prev_prev_parsed = sequence[i-2]['parsed']
                
                # Vector from i-2 to i-1
                v1 = self._get_vector(prev_prev_parsed, prev_parsed)
                # Vector from i-1 to i  
                v2 = self._get_vector(prev_parsed, curr_parsed)
                
                if v1 and v2:
                    dot_product = np.dot(v1, v2)
                    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
                    if norms > 0:
                        cos_angle = dot_product / norms
                        if cos_angle < 0.8:  # Significant direction change
                            direction_changes += 1
            
            # Check for feedrate change
            if (prev_parsed['feedrate'] and curr_parsed['feedrate'] and
                abs(prev_parsed['feedrate'] - curr_parsed['feedrate']) > 10):
                feedrate_changes += 1
        
        # Classify based on characteristics
        avg_distance = np.mean(distances) if distances else 0
        direction_change_rate = direction_changes / len(sequence)
        feedrate_change_rate = feedrate_changes / len(sequence)
        
        if direction_change_rate > 0.3:
            return 'corner_heavy'
        elif avg_distance < 1.0 and direction_change_rate > 0.1:
            return 'detailed'
        elif feedrate_change_rate > 0.2:
            return 'variable_speed'
        elif avg_distance > 10.0 and direction_change_rate < 0.1:
            return 'linear'
        else:
            return 'mixed'
    
    def _get_vector(self, from_cmd, to_cmd):
        """Get movement vector between two commands"""
        try:
            dx = to_cmd['coordinates'].get('X', 0) - from_cmd['coordinates'].get('X', 0)
            dy = to_cmd['coordinates'].get('Y', 0) - from_cmd['coordinates'].get('Y', 0)
            dz = to_cmd['coordinates'].get('Z', 0) - from_cmd['coordinates'].get('Z', 0)
            return np.array([dx, dy, dz])
        except:
            return None
    
    def _estimate_resonance_risk(self, sequence):
        """Estimate the resonance excitation risk for a sequence"""
        if len(sequence) < 2:
            return 0.0
        
        risk_factors = []
        
        # High acceleration risk
        for cmd in sequence:
            if cmd['parsed']['feedrate']:
                feedrate = cmd['parsed']['feedrate']
                if feedrate > 3000:  # High speed
                    risk_factors.append(0.3)
                elif feedrate > 1500:  # Medium speed
                    risk_factors.append(0.1)
        
        # Direction change risk
        direction_changes = 0
        for i in range(2, len(sequence)):
            v1 = self._get_vector(sequence[i-2]['parsed'], sequence[i-1]['parsed'])
            v2 = self._get_vector(sequence[i-1]['parsed'], sequence[i]['parsed'])
            
            if v1 is not None and v2 is not None:
                dot_product = np.dot(v1, v2)
                norms = np.linalg.norm(v1) * np.linalg.norm(v2)
                if norms > 0:
                    cos_angle = dot_product / norms
                    if cos_angle < 0.5:  # Sharp direction change
                        risk_factors.append(0.4)
                        direction_changes += 1
        
        # Repetitive motion risk (resonance excitation)
        if direction_changes > len(sequence) * 0.3:
            risk_factors.append(0.5)
        
        return min(1.0, sum(risk_factors))
    
    def _recommend_sequence_compensation(self, sequence):
        """Recommend compensation parameters for a sequence"""
        pattern = self._classify_sequence_pattern(sequence)
        resonance_risk = self._estimate_resonance_risk(sequence)
        
        # Base recommendations
        compensation = {
            'freq_adjustment': 1.0,
            'damping_adjustment': 1.0,
            'shaper_hint': None,
            'priority': 'normal'
        }
        
        # Pattern-specific adjustments
        if pattern == 'corner_heavy':
            compensation['damping_adjustment'] = 1.2
            compensation['shaper_hint'] = 'ei'
            compensation['priority'] = 'high'
        elif pattern == 'detailed':
            compensation['shaper_hint'] = 'ulv'
            compensation['freq_adjustment'] = 0.95
            compensation['priority'] = 'high'
        elif pattern == 'linear':
            compensation['shaper_hint'] = 'smooth'
            compensation['freq_adjustment'] = 1.1
        elif pattern == 'variable_speed':
            compensation['shaper_hint'] = 'adaptive_ei'
            compensation['damping_adjustment'] = 1.1
        
        # Resonance risk adjustments
        if resonance_risk > 0.7:
            compensation['damping_adjustment'] *= 1.3
            compensation['freq_adjustment'] *= 1.05
            compensation['priority'] = 'critical'
        elif resonance_risk > 0.4:
            compensation['damping_adjustment'] *= 1.15
            compensation['priority'] = 'high'
        
        return compensation
    
    def _determine_compensation_needs(self, motion_prediction):
        """Determine if compensation changes are needed"""
        if not motion_prediction:
            return None
        
        compensation = motion_prediction.get('recommended_compensation', {})
        
        # Check if compensation differs significantly from current
        needs_update = False
        
        if compensation.get('priority') in ['high', 'critical']:
            needs_update = True
        
        if abs(compensation.get('freq_adjustment', 1.0) - 1.0) > 0.05:
            needs_update = True
        
        if abs(compensation.get('damping_adjustment', 1.0) - 1.0) > 0.05:
            needs_update = True
        
        if compensation.get('shaper_hint'):
            needs_update = True
        
        return compensation if needs_update else None
    
    def _precompute_compensation(self):
        """Pre-calculate compensation parameters for upcoming moves"""
        current_time = time.time()
        
        # Look at upcoming commands that need compensation
        for cmd in self.gcode_buffer:
            if cmd['compensation_needed'] and cmd['time'] > current_time:
                # Calculate when to apply this compensation
                apply_time = cmd['time'] - 0.5  # Apply 500ms before needed
                
                if apply_time > current_time:
                    # Schedule compensation change
                    self.compensation_schedule.append({
                        'time': apply_time,
                        'compensation': cmd['compensation_needed'],
                        'reason': f"Predicted for {cmd['predicted_motion'].get('predicted_pattern', 'unknown')} pattern"
                    })
    
    def _apply_scheduled_compensation(self, current_time):
        """Apply compensation changes that are scheduled for now"""
        while (self.compensation_schedule and 
               self.compensation_schedule[0]['time'] <= current_time):
            
            scheduled_change = self.compensation_schedule.popleft()
            
            try:
                # Apply the compensation change
                self._apply_compensation_change(scheduled_change['compensation'])
                logging.info(f"Applied predictive compensation: {scheduled_change['reason']}")
                
            except Exception as e:
                logging.warning(f"Failed to apply scheduled compensation: {e}")
    
    def _apply_compensation_change(self, compensation):
        """Apply a compensation parameter change"""
        try:
            # Get dynamic input shaper instance
            dynamic_shaper = self.printer.lookup_object('dynamic_input_shaper')
            
            # Apply the compensation adjustments
            for axis in ['x', 'y', 'z']:
                if axis in dynamic_shaper.current_adjustments:
                    dynamic_shaper.current_adjustments[axis]['freq'] *= compensation.get('freq_adjustment', 1.0)
                    dynamic_shaper.current_adjustments[axis]['damping'] *= compensation.get('damping_adjustment', 1.0)
                    
                    if compensation.get('shaper_hint'):
                        dynamic_shaper.current_adjustments[axis]['shaper_hint'] = compensation['shaper_hint']
                        
        except Exception as e:
            logging.warning(f"Failed to apply compensation change: {e}")
    
    def get_prediction_status(self):
        """Get current prediction system status"""
        return {
            'enabled': self.enabled,
            'buffered_commands': len(self.gcode_buffer),
            'scheduled_compensations': len(self.compensation_schedule),
            'upcoming_patterns': [cmd['predicted_motion'].get('predicted_pattern', 'unknown') 
                                for cmd in self.gcode_buffer 
                                if cmd.get('predicted_motion')]
        }

######################################################################
######################################################################

class FeatureTypeManager:
    """Manages G-code feature type detection and specific compensation configurations"""
    
    def __init__(self, config):
        self.printer = config.get_printer()
        
        # Feature type detection
        self.current_feature = None
        self.feature_regex = re.compile(r';\s*TYPE:\s*([\w-]+)', re.IGNORECASE)
        
        # Feature-specific configurations
        self.feature_configs = {}
        self._load_feature_configs(config)
        
        # Quality presets for each feature type
        self.quality_presets = {
            'speed': {
                'shaper_preference': ['smooth', 'zv', 'mzv'],
                'freq_adjustment': 1.1,
                'damping_adjustment': 0.9
            },
            'balance': {
                'shaper_preference': ['ei', 'adaptive_ei', 'mzv'],
                'freq_adjustment': 1.0,
                'damping_adjustment': 1.0
            },
            'quality': {
                'shaper_preference': ['ulv', 'multi_freq', 'ei'],
                'freq_adjustment': 0.95,
                'damping_adjustment': 1.2
            }
        }
        
        # Hook into G-code processing
        gcode = self.printer.lookup_object('gcode')
        self.original_process_commands = gcode._process_commands
        gcode._process_commands = self._enhanced_process_commands
    
    def _load_feature_configs(self, config):
        """Load feature-specific configuration from config file"""
        
        # Default feature types with their default preferences
        default_features = {
            'WALL-OUTER': 'quality',
            'WALL-INNER': 'balance', 
            'INFILL': 'speed',
            'SUPPORT': 'speed',
            'BRIDGE': 'quality',
            'TOP-SURFACE': 'quality',
            'BOTTOM-SURFACE': 'balance',
            'PERIMETER': 'quality',
            'SOLID-INFILL': 'balance',
            'SPARSE-INFILL': 'speed',
            'SKIRT': 'speed',
            'BRIM': 'balance'
        }
        
        # Load configurations from config file
        for feature, default_pref in default_features.items():
            pref = config.get(f'feature_{feature.lower().replace("-", "_")}_preference', default_pref)
            if pref not in self.quality_presets:
                logging.warning(f"Unknown preference '{pref}' for feature {feature}, using 'balance'")
                pref = 'balance'
            
            self.feature_configs[feature] = {
                'preference': pref,
                'custom_shaper': config.get(f'feature_{feature.lower().replace("-", "_")}_shaper', None),
                'custom_freq_x': config.getfloat(f'feature_{feature.lower().replace("-", "_")}_freq_x', None),
                'custom_freq_y': config.getfloat(f'feature_{feature.lower().replace("-", "_")}_freq_y', None),
                'custom_freq_z': config.getfloat(f'feature_{feature.lower().replace("-", "_")}_freq_z', None),
            }
    
    def _enhanced_process_commands(self, commands, need_ack=True):
        """Enhanced G-code command processor that detects feature types"""
        for line in commands:
            # Check for feature type comments before processing
            comment_match = self.feature_regex.search(line)
            if comment_match:
                new_feature = comment_match.group(1).upper()
                if new_feature != self.current_feature:
                    self.current_feature = new_feature
                    self._notify_feature_change(new_feature)
        
        # Process commands normally
        return self.original_process_commands(commands, need_ack)
    
    def _notify_feature_change(self, feature_type):
        """Notify dynamic input shaper of feature type change"""
        try:
            dynamic_shaper = self.printer.lookup_object('dynamic_input_shaper')
            dynamic_shaper.set_current_feature(feature_type)
        except:
            pass  # Dynamic shaper may not be configured
    
    def get_feature_compensation_params(self, feature_type):
        """Get compensation parameters for a specific feature type"""
        if feature_type not in self.feature_configs:
            feature_type = 'INFILL'  # Default fallback
        
        config = self.feature_configs[feature_type]
        preset = self.quality_presets[config['preference']]
        
        return {
            'shaper_preference': preset['shaper_preference'],
            'freq_adjustment': preset['freq_adjustment'],
            'damping_adjustment': preset['damping_adjustment'],
            'custom_shaper': config['custom_shaper'],
            'custom_freqs': {
                'x': config['custom_freq_x'],
                'y': config['custom_freq_y'], 
                'z': config['custom_freq_z']
            }
        }

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
        
        # ML-based pattern recognizer
        self.ml_recognizer = None  # Will be set by DynamicInputShaper
        
    def set_ml_recognizer(self, ml_recognizer):
        """Set the ML pattern recognizer"""
        self.ml_recognizer = ml_recognizer
        
    def analyze_motion(self, position, velocity, acceleration, time_stamp, gcode_context=None):
        """Analyze current motion and return recommended compensation parameters"""
        
        # Support multi-axis analysis (X, Y, Z, A, B, C)
        num_axes = min(len(position), len(velocity), len(acceleration), 6)
        
        # Add to motion history
        self.motion_history.append({
            'time': time_stamp,
            'position': position[:num_axes],
            'velocity': velocity[:num_axes],
            'acceleration': acceleration[:num_axes],
            'gcode_context': gcode_context
        })
        
        # Check if enough time has passed for update
        if time_stamp - self.last_update_time < self.update_interval:
            return None
        
        self.last_update_time = time_stamp
        
        # Analyze recent motion
        if len(self.motion_history) < 10:
            return None
        
        # Detect movement pattern using ML if available
        if self.ml_recognizer and self.ml_recognizer.enabled:
            pattern_result = self.ml_recognizer.predict_pattern(self.motion_history, gcode_context)
            if isinstance(pattern_result, dict):
                pattern = pattern_result['pattern']
                self.pattern_confidence = pattern_result['confidence']
            else:
                pattern = pattern_result
                self.pattern_confidence = 0.5
        else:
            pattern = self._detect_movement_pattern()
            self.pattern_confidence = 0.5
        
        self.current_pattern = pattern
        
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
        # Support multi-axis recommendations
        axis_names = ['x', 'y', 'z', 'a', 'b', 'c']
        recommendations = {}
        
        for axis in axis_names:
            recommendations[axis] = {
                'freq_adjustment': 1.0, 
                'damping_adjustment': 1.0, 
                'shaper_hint': None
            }
        
        # Pattern-based adjustments
        if pattern == 'corner_heavy':
            # Corners benefit from higher damping and potentially different shaper
            for axis in axis_names:
                recommendations[axis]['damping_adjustment'] = 1.2
                recommendations[axis]['shaper_hint'] = 'ei'  # Better for corners
                
        elif pattern == 'linear':
            # Linear movements can use faster shapers
            for axis in axis_names:
                recommendations[axis]['shaper_hint'] = 'smooth'  # Speed-optimized
                
        elif pattern == 'variable_speed':
            # Variable speed benefits from adaptive shapers
            for axis in axis_names:
                recommendations[axis]['shaper_hint'] = 'adaptive_ei'
        
        # Speed-based adjustments
        if stats['avg_speed'] > 150:
            # High speed - increase frequency slightly
            for axis in axis_names:
                recommendations[axis]['freq_adjustment'] *= 1.1
                
        elif stats['avg_speed'] < 50:
            # Low speed - can use higher quality shapers
            for axis in axis_names:
                recommendations[axis]['shaper_hint'] = 'ulv'
        
        # Acceleration-based adjustments
        if stats['avg_accel'] > 5000:
            # High acceleration - need better damping
            for axis in axis_names:
                recommendations[axis]['damping_adjustment'] *= 1.15
                recommendations[axis]['freq_adjustment'] *= 1.05
        
        return recommendations

class DynamicInputShaper:
    """Dynamic input shaper that adapts in real-time with ML and predictive capabilities"""
    
    def __init__(self, config):
        self.printer = config.get_printer()
        self.name = config.get_name().split()[-1]
        
        # Configuration
        self.enabled = config.getboolean('enabled', False)
        self.adaptation_rate = config.getfloat('adaptation_rate', 0.1, minval=0.01, maxval=1.0)
        self.min_update_interval = config.getfloat('min_update_interval', 0.5, minval=0.1, maxval=5.0)
        
        # Feature type management
        self.feature_manager = FeatureTypeManager(config)
        self.current_feature = None
        
        # ML-based pattern recognition
        self.ml_recognizer = MLPatternRecognizer(config, self.printer)
        
        # Motion analyzer with ML integration
        self.motion_analyzer = MotionAnalyzer(config)
        self.motion_analyzer.set_ml_recognizer(self.ml_recognizer)
        
        # Predictive compensation system
        self.predictive_compensator = PredictiveCompensator(config, self.printer)
        
        # Current adaptive parameters
        self.base_parameters = {}  # Multi-axis base parameters from calibration
        self.current_adjustments = {}  # Multi-axis current adjustments
        
        # Initialize for all potential axes
        axis_names = ['x', 'y', 'z', 'a', 'b', 'c']
        for axis in axis_names:
            self.base_parameters[axis] = None
            self.current_adjustments[axis] = {'freq': 1.0, 'damping': 1.0}
            
        self.last_adjustment_time = 0
        
        # Parameter transition smoothing
        self.transition_queue = collections.deque()
        self.target_parameters = {axis: None for axis in axis_names}
        
        # Performance tracking for ML feedback
        self.compensation_effectiveness = {}
        self.pattern_performance_history = collections.deque(maxlen=1000)
        
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
        gcode.register_command("SET_FEATURE_COMPENSATION",
                             self.cmd_SET_FEATURE_COMPENSATION,
                             desc=self.cmd_SET_FEATURE_COMPENSATION_help)
        gcode.register_command("TRAIN_ML_MODEL",
                             self.cmd_TRAIN_ML_MODEL,
                             desc=self.cmd_TRAIN_ML_MODEL_help)
        gcode.register_command("GET_PREDICTION_STATUS",
                             self.cmd_GET_PREDICTION_STATUS,
                             desc=self.cmd_GET_PREDICTION_STATUS_help)
        gcode.register_command("UPDATE_PATTERN_EFFECTIVENESS",
                             self.cmd_UPDATE_PATTERN_EFFECTIVENESS,
                             desc=self.cmd_UPDATE_PATTERN_EFFECTIVENESS_help)

    def set_current_feature(self, feature_type):
        """Set the current feature type for adaptive compensation"""
        if feature_type != self.current_feature:
            self.current_feature = feature_type
            self._update_feature_compensation()
    
    def _update_feature_compensation(self):
        """Update compensation parameters based on current feature type"""
        if not self.current_feature:
            return
        
        feature_params = self.feature_manager.get_feature_compensation_params(self.current_feature)
        
        # Apply feature-specific adjustments to current parameters
        axis_names = ['x', 'y', 'z', 'a', 'b', 'c']
        for axis in axis_names:
            if self.base_parameters[axis] is not None:
                # Apply feature-specific frequency adjustment
                freq_adj = feature_params['freq_adjustment']
                damping_adj = feature_params['damping_adjustment']
                
                # Check for custom frequency override
                custom_freq = feature_params['custom_freqs'].get(axis)
                if custom_freq is not None:
                    freq_adj = custom_freq / self.base_parameters[axis].get('freq', 1.0)
                
                self.current_adjustments[axis]['freq'] = freq_adj
                self.current_adjustments[axis]['damping'] = damping_adj
                
                # Apply shaper preference if no custom shaper specified
                if not feature_params['custom_shaper']:
                    preferred_shapers = feature_params['shaper_preference']
                    self.current_adjustments[axis]['shaper_hint'] = preferred_shapers[0]

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
            
            # Get current G-code context if available
            gcode_context = getattr(toolhead, 'current_gcode_line', None)
            
            # Analyze motion and get recommendations
            recommendations = self.motion_analyzer.analyze_motion(
                position, velocity, acceleration, eventtime, gcode_context)
            
            if recommendations:
                self._apply_dynamic_adjustments(recommendations, eventtime)
                
                # Track effectiveness for ML feedback
                self._track_compensation_effectiveness(recommendations, eventtime)
                
        except Exception as e:
            logging.warning(f"Dynamic shaper update failed: {e}")
        
        return eventtime + self.motion_analyzer.update_interval
    
    def _track_compensation_effectiveness(self, recommendations, eventtime):
        """Track the effectiveness of applied compensation for ML feedback"""
        current_pattern = self.motion_analyzer.current_pattern
        if current_pattern:
            # Store performance data (simplified - real implementation would measure actual vibration reduction)
            performance_data = {
                'time': eventtime,
                'pattern': current_pattern,
                'confidence': self.motion_analyzer.pattern_confidence,
                'recommendations': recommendations,
                'effectiveness_score': 0.8  # Placeholder - would be measured from actual printer response
            }
            
            self.pattern_performance_history.append(performance_data)
            
            # Update ML model with effectiveness feedback
            if len(self.pattern_performance_history) > 10:
                recent_performance = list(self.pattern_performance_history)[-10:]
                avg_effectiveness = np.mean([p['effectiveness_score'] for p in recent_performance])
                
                if current_pattern in self.compensation_effectiveness:
                    # Exponential moving average
                    self.compensation_effectiveness[current_pattern] = (
                        0.8 * self.compensation_effectiveness[current_pattern] + 
                        0.2 * avg_effectiveness
                    )
                else:
                    self.compensation_effectiveness[current_pattern] = avg_effectiveness
                
                # Provide feedback to ML model
                self.ml_recognizer.update_effectiveness_score(current_pattern, avg_effectiveness)

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
        gcmd.respond_info(f"Current feature type: {self.current_feature or 'None'}")
        
        # Current pattern information
        if self.motion_analyzer.current_pattern:
            gcmd.respond_info(f"Current motion pattern: {self.motion_analyzer.current_pattern}")
            gcmd.respond_info(f"Pattern confidence: {self.motion_analyzer.pattern_confidence:.2f}")
        
        axis_names = ['x', 'y', 'z', 'a', 'b', 'c']
        for axis in axis_names:
            if self.base_parameters[axis]:
                base = self.base_parameters[axis]
                current = self.current_adjustments[axis]
                gcmd.respond_info(f"{axis.upper()}-axis base: {base[0]} @ {base[1]:.1f}Hz")
                gcmd.respond_info(f"{axis.upper()}-axis adjustments: freq×{current['freq']:.2f}, damping×{current['damping']:.2f}")
        
        # ML status
        if self.ml_recognizer.enabled:
            gcmd.respond_info("--- ML Pattern Recognition ---")
            gcmd.respond_info(f"Training samples: {len(self.ml_recognizer.training_data['features'])}")
            gcmd.respond_info(f"Model trained: {'Yes' if self.ml_recognizer.pattern_classifier else 'No'}")
            gcmd.respond_info(f"Discovered patterns: {len(self.ml_recognizer.discovered_patterns)}")
            
            # Recent effectiveness scores
            if self.compensation_effectiveness:
                gcmd.respond_info("Pattern effectiveness scores:")
                for pattern, score in self.compensation_effectiveness.items():
                    gcmd.respond_info(f"  {pattern}: {score:.3f}")
        
        # Predictive status
        if self.predictive_compensator.enabled:
            status = self.predictive_compensator.get_prediction_status()
            gcmd.respond_info("--- Predictive Compensation ---")
            gcmd.respond_info(f"Buffered commands: {status['buffered_commands']}")
            gcmd.respond_info(f"Scheduled compensations: {status['scheduled_compensations']}")
            
            if status['upcoming_patterns']:
                patterns = ', '.join(set(status['upcoming_patterns'][:10]))
                gcmd.respond_info(f"Predicted patterns: {patterns}")

    cmd_SET_FEATURE_COMPENSATION_help = "Configure feature-specific compensation parameters"
    def cmd_SET_FEATURE_COMPENSATION(self, gcmd):
        """Configure feature-specific compensation parameters"""
        feature = gcmd.get('FEATURE')
        preference = gcmd.get('PREFERENCE', 'balance')
        
        if preference not in ['speed', 'balance', 'quality']:
            raise gcmd.error("PREFERENCE must be one of: speed, balance, quality")
        
        # Update feature configuration
        if feature not in self.feature_manager.feature_configs:
            self.feature_manager.feature_configs[feature] = {}
        
        self.feature_manager.feature_configs[feature]['preference'] = preference
        
        # Optional custom parameters
        custom_shaper = gcmd.get('CUSTOM_SHAPER', None)
        if custom_shaper:
            self.feature_manager.feature_configs[feature]['custom_shaper'] = custom_shaper
        
        for axis in ['x', 'y', 'z']:
            freq_param = f'CUSTOM_FREQ_{axis.upper()}'
            custom_freq = gcmd.get_float(freq_param, None)
            if custom_freq:
                self.feature_manager.feature_configs[feature][f'custom_freq_{axis}'] = custom_freq
        
        gcmd.respond_info(f"Set {feature} compensation to {preference} preference")
        
        # Apply immediately if this is the current feature
        if feature == self.current_feature:
            self._update_feature_compensation()

    cmd_TRAIN_ML_MODEL_help = "Train or retrain the ML pattern recognition model"
    def cmd_TRAIN_ML_MODEL(self, gcmd):
        """Train or retrain the ML model"""
        force = gcmd.get_int('FORCE', 0, minval=0, maxval=1)
        
        if not self.ml_recognizer.enabled:
            gcmd.respond_info("ML pattern recognition is disabled")
            return
        
        gcmd.respond_info("Starting ML model training...")
        success = self.ml_recognizer.train_model(force_retrain=bool(force))
        
        if success:
            gcmd.respond_info("ML model trained successfully")
        else:
            gcmd.respond_info("ML model training failed - check logs for details")
    
    cmd_GET_PREDICTION_STATUS_help = "Get predictive compensation system status"
    def cmd_GET_PREDICTION_STATUS(self, gcmd):
        """Get predictive system status"""
        if not self.predictive_compensator.enabled:
            gcmd.respond_info("Predictive compensation is disabled")
            return
        
        status = self.predictive_compensator.get_prediction_status()
        
        gcmd.respond_info(f"Predictive compensation enabled: {status['enabled']}")
        gcmd.respond_info(f"Buffered commands: {status['buffered_commands']}")
        gcmd.respond_info(f"Scheduled compensations: {status['scheduled_compensations']}")
        
        if status['upcoming_patterns']:
            patterns = ', '.join(status['upcoming_patterns'][:5])  # Show first 5
            gcmd.respond_info(f"Upcoming patterns: {patterns}")
        
        # ML status
        if self.ml_recognizer.enabled:
            gcmd.respond_info(f"ML training samples: {len(self.ml_recognizer.training_data['features'])}")
            gcmd.respond_info(f"Discovered patterns: {len(self.ml_recognizer.discovered_patterns)}")
            if self.ml_recognizer.pattern_classifier:
                gcmd.respond_info("ML model: Trained and active")
            else:
                gcmd.respond_info("ML model: Not trained")
        else:
            gcmd.respond_info("ML recognition: Disabled")
    
    cmd_UPDATE_PATTERN_EFFECTIVENESS_help = "Update effectiveness score for a motion pattern"
    def cmd_UPDATE_PATTERN_EFFECTIVENESS(self, gcmd):
        """Update pattern effectiveness score for ML learning"""
        pattern = gcmd.get('PATTERN')
        score = gcmd.get_float('SCORE', minval=0.0, maxval=1.0)
        
        if not self.ml_recognizer.enabled:
            gcmd.respond_info("ML pattern recognition is disabled")
            return
        
        self.ml_recognizer.update_effectiveness_score(pattern, score)
        gcmd.respond_info(f"Updated effectiveness score for pattern '{pattern}': {score:.2f}")
        
        # Save training data
        self.ml_recognizer._save_training_data()

def load_config(config):
    return DynamicInputShaper(config)