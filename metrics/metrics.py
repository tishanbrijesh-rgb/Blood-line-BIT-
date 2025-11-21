import numpy as np
import cv2
from collections import deque
from landmark.landmarks import LandmarkExtractor

class MetricsCalculator:
    """
    Module C: Calculates frame-level metrics (EAR, MAR, Pose) and aggregates
    them over time to derive temporal features (rate, variance) for the 
    Decision Engine.
    """
    
    # Constants for temporal history (adjust for desired smoothing/lag)
    # 60 frames = approx 2 seconds at 30 FPS
    HISTORY_SIZE = 60 

    def __init__(self):
        """Initializes history buffers for all metrics and helpers."""
        self.extractor = LandmarkExtractor() # Reuse extractor for helper formulas
        
        # Temporal buffers for frame-level values
        self.left_ear_history = deque(maxlen=self.HISTORY_SIZE)
        self.right_ear_history = deque(maxlen=self.HISTORY_SIZE)
        self.mar_history = deque(maxlen=self.HISTORY_SIZE)
        self.yaw_history = deque(maxlen=self.HISTORY_SIZE)
        self.pitch_history = deque(maxlen=self.HISTORY_SIZE)
        self.luminance_history = deque(maxlen=self.HISTORY_SIZE)
        
        # Blink detection state
        self.ear_threshold = 0.25 # Typical threshold for a blink
        self.is_blinking = False
        self.blink_count = 0
        self.frame_count = 0

    def _calculate_frame_metrics(self, landmarks_dict, frame):
        """Calculates instantaneous frame-level metrics."""
        
        # 1. EAR (Eye Aspect Ratio)
        left_ear = self.extractor.get_eye_aspect_ratio(landmarks_dict['left_eye'])
        right_ear = self.extractor.get_eye_aspect_ratio(landmarks_dict['right_eye'])
        avg_ear = (left_ear + right_ear) / 2.0
        
        # 2. MAR (Mouth Aspect Ratio)
        mar = self.extractor.get_mouth_aspect_ratio(landmarks_dict['mouth_outer'])
        
        # 3. Head Pose (Yaw, Pitch, Roll)
        head_pose = self.extractor.get_head_pose(landmarks_dict)
        yaw = head_pose['yaw']
        pitch = head_pose['pitch']
        
        # 4. Lighting (Luminance)
        # Convert frame to grayscale and calculate average luminance (quick proxy)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luminance = np.mean(gray)
        
        return avg_ear, mar, yaw, pitch, luminance

    def _update_temporal_history(self, avg_ear, mar, yaw, pitch, luminance):
        """Updates all history buffers."""
        self.left_ear_history.append(avg_ear)
        self.right_ear_history.append(avg_ear) # Use average for consistency
        self.mar_history.append(mar)
        self.yaw_history.append(yaw)
        self.pitch_history.append(pitch)
        self.luminance_history.append(luminance)
        self.frame_count += 1
        
        # Blink Counting Logic
        ear_avg = np.mean(self.left_ear_history) # Use average EAR over history
        if ear_avg < self.ear_threshold and not self.is_blinking:
            self.is_blinking = True
        elif ear_avg >= self.ear_threshold and self.is_blinking:
            self.blink_count += 1
            self.is_blinking = False

    def calculate_metrics(self, landmarks_dict, frame):
        """
        Main function to calculate all features and temporal metrics.

        Returns:
            dict: Processed metrics ready for the DecisionEngine.
        """
        if landmarks_dict is None:
            return None

        avg_ear, mar, yaw, pitch, luminance = self._calculate_frame_metrics(landmarks_dict, frame)
        self._update_temporal_history(avg_ear, mar, yaw, pitch, luminance)

        # 1. Blink Rate (Blips per second, calculated over the history window)
        # Avoid division by zero
        time_elapsed = self.frame_count / 30.0 if self.frame_count > 0 else 1 
        avg_blink_rate = self.blink_count / time_elapsed 
        
        # 2. MAR Variance (Mouth movement randomness)
        mar_variance = np.var(self.mar_history) if len(self.mar_history) > 1 else 0
        
        # 3. Head Tilt Variance (Micro-movement consistency)
        # Use variance of yaw and pitch combined for overall stability
        yaw_var = np.var(self.yaw_history) if len(self.yaw_history) > 1 else 0
        pitch_var = np.var(self.pitch_history) if len(self.pitch_history) > 1 else 0
        head_tilt_variance = (yaw_var + pitch_var) / 2.0
        
        # 4. Lighting Fluctuation
        light_variance = np.var(self.luminance_history) if len(self.luminance_history) > 1 else 0

        # Return the final metrics dictionary for the Decision Engine
        return {
            'avg_ear': avg_ear,
            'mar': mar,
            'avg_blink_rate': avg_blink_rate,
            'mar_variance': mar_variance,
            'head_tilt_variance': head_tilt_variance,
            'light_variance': light_variance,
            'current_yaw': yaw,
            'current_pitch': pitch,
        }