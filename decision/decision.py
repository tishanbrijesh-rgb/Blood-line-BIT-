import numpy as np
from collections import deque

class DecisionEngine:
    """
    Implements the rule-based anomaly detection engine for FaceTrap.
    
    It analyzes the calculated motion metrics (from metrics.py) against
    predefined thresholds to classify the video feed in real-time.
    
    Output classification: REAL, SUSPICIOUS, POSSIBLE DEEPFAKE.
    """

    def __init__(self, history_size=60):
        """
        Initializes the engine with default thresholds and a buffer for recent decisions.
        
        history_size: Number of frames to keep in history for temporal consistency (e.g., 60 frames = 2 seconds at 30fps).
        """
        # --- Configurable Anomaly Thresholds (Adjust based on testing) ---
        
        # Blink Rate (Blip per second) - Normal: 0.1 to 0.6 blinks/sec
        self.LOW_BLINK_RATE_THRESHOLD = 0.08  # Too few blinks (suspicious)
        self.NO_BLINK_RATE_THRESHOLD = 0.03   # Virtually no blinking (highly suspicious/fake)
        
        # MAR Variance (Mouth Aspect Ratio Standard Deviation) - During speech
        self.LOW_MAR_VAR_THRESHOLD = 0.0005   # Too smooth mouth movement
        self.HIGH_MAR_VAR_THRESHOLD = 0.005   # Too erratic mouth movement
        
        # Head Tilt Variance (Standard Deviation of head movement in degrees)
        self.LOW_HEAD_TILT_VAR_THRESHOLD = 0.5 # Unnaturally still head
        self.HIGH_HEAD_TILT_VAR_THRESHOLD = 5.0 # Unnaturally shaky head
        
        # Lighting Fluctuation (Standard Deviation of average frame luminance)
        self.LOW_LIGHT_VAR_THRESHOLD = 1.0    # Too static/flat lighting
        
        # --- Decision Thresholds (Anomaly Score) ---
        # Anomaly Score is the sum of weighted rule violations.
        self.SUSPICIOUS_SCORE_THRESHOLD = 3   # Score >= 3 triggers SUSPICIOUS
        self.DEEPFAKE_SCORE_THRESHOLD = 6     # Score >= 6 triggers POSSIBLE DEEPFAKE
        
        # --- History Buffer for Temporal Consistency ---
        # Keep track of recent anomaly flags to avoid flickers
        self.history = deque([0] * history_size, maxlen=history_size)
        
        # Weights for each rule when scoring an anomaly
        self.weights = {
            'blink_low': 2,
            'blink_none': 4,
            'mar_low': 2,
            'head_tilt_low': 3,
            'light_low': 1,
            'mar_high': 1,
            'head_tilt_high': 1,
        }


    def analyze_metrics(self, metrics: dict) -> dict:
        """
        Analyzes the motion metrics and returns a detection result.

        Args:
            metrics: A dictionary containing the latest calculated metrics.
                     Example: {
                         'avg_blink_rate': 0.15, 
                         'mar_variance': 0.0012, 
                         'head_tilt_variance': 1.5,
                         'light_variance': 3.4
                     }

        Returns:
            A dictionary with the final classification and anomaly score.
        """
        anomaly_score = 0
        current_frame_is_fake = False
        
        # 1. Blink Rate Analysis
        blink_rate = metrics.get('avg_blink_rate', 0.0)
        if blink_rate < self.NO_BLINK_RATE_THRESHOLD:
            anomaly_score += self.weights['blink_none']
            current_frame_is_fake = True
        elif blink_rate < self.LOW_BLINK_RATE_THRESHOLD:
            anomaly_score += self.weights['blink_low']

        # 2. Mouth Aspect Ratio (MAR) Variance Analysis
        mar_variance = metrics.get('mar_variance', 0.0)
        if mar_variance < self.LOW_MAR_VAR_THRESHOLD:
            anomaly_score += self.weights['mar_low']
        elif mar_variance > self.HIGH_MAR_VAR_THRESHOLD:
             anomaly_score += self.weights['mar_high']

        # 3. Head Tilt Variance Analysis
        head_tilt_variance = metrics.get('head_tilt_variance', 0.0)
        if head_tilt_variance < self.LOW_HEAD_TILT_VAR_THRESHOLD:
            # High weight because this is a common, strong deepfake artifact
            anomaly_score += self.weights['head_tilt_low']
            current_frame_is_fake = True
        elif head_tilt_variance > self.HIGH_HEAD_TILT_VAR_THRESHOLD:
            anomaly_score += self.weights['head_tilt_high']

        # 4. Lighting Fluctuation Analysis
        light_variance = metrics.get('light_variance', 0.0)
        if light_variance < self.LOW_LIGHT_VAR_THRESHOLD:
            anomaly_score += self.weights['light_low']

        # --- Temporal Smoothing and Final Classification ---
        
        # Add a flag to the history buffer (1 if score > SUSPICIOUS_THRESHOLD, 0 otherwise)
        self.history.append(1 if anomaly_score >= self.SUSPICIOUS_SCORE_THRESHOLD else 0)
        
        # Calculate the proportion of 'suspicious' frames in the recent history
        recent_anomaly_count = sum(self.history)
        
        # Default status
        detection_label = "REAL"
        
        if recent_anomaly_count > self.DEEPFAKE_SCORE_THRESHOLD:
            # If the cumulative score over the time window exceeds the deepfake threshold (e.g., 6 seconds of flags)
            detection_label = "POSSIBLE DEEPFAKE"
        elif recent_anomaly_count > (self.history.maxlen / 5): # e.g., 1/5 of the history is suspicious
            detection_label = "SUSPICIOUS"
        
        # Final decision payload
        return {
            'label': detection_label,
            'raw_score': anomaly_score,
            'recent_flags': recent_anomaly_count,
            'metrics_used': metrics # Echoing the metrics for visualization/debugging
        }

# --- Example Usage (for testing by Member D) ---
if __name__ == "__main__":
    engine = DecisionEngine(history_size=30) # 1 second history at 30fps

    # Simulating 3 scenarios: Real, Suspicious, Deepfake

    # Scenario 1: REAL USER (Natural movement, high variability)
    real_metrics = {
        'avg_blink_rate': 0.35,          # Normal
        'mar_variance': 0.0018,          # Normal speech variance
        'head_tilt_variance': 1.8,       # Natural micro-movement
        'light_variance': 5.0            # Normal environmental fluctuation
    }
    
    # Scenario 2: SUSPICIOUS (Slightly reduced movement/smoothness)
    suspicious_metrics = {
        'avg_blink_rate': 0.10,          # On the low end
        'mar_variance': 0.0006,          # Close to too smooth
        'head_tilt_variance': 0.8,       # Slightly stiff
        'light_variance': 2.0
    }

    # Scenario 3: POSSIBLE DEEPFAKE (Highly unnatural pattern: no blink, stiff head)
    fake_metrics = {
        'avg_blink_rate': 0.02,          # Virtually no blinking (High Anomaly)
        'mar_variance': 0.0001,          # Extremely smooth/static mouth (High Anomaly)
        'head_tilt_variance': 0.1,       # Almost perfectly still head (Highest Anomaly)
        'light_variance': 0.5            # Flat lighting (Low Anomaly)
    }

    print("--- Testing Real Scenario (Expected: REAL) ---")
    for i in range(5):
        result = engine.analyze_metrics(real_metrics)
        print(f"Frame {i}: Score={result['raw_score']}, Recent Flags={result['recent_flags']}, Label: {result['label']}")

    print("\n--- Testing Deepfake Scenario (Expected: POSSIBLE DEEPFAKE after history fills) ---")
    for i in range(35):
        # Simulate a deepfake stream over 35 frames (over 1 second history)
        result = engine.analyze_metrics(fake_metrics)
        print(f"Frame {i+5}: Score={result['raw_score']}, Recent Flags={result['recent_flags']}, Label: {result['label']}")

    print("\n--- Testing Suspicious Scenario (Expected: SUSPICIOUS) ---")
    for i in range(5):
        result = engine.analyze_metrics(suspicious_metrics)
        print(f"Frame {i+40}: Score={result['raw_score']}, Recent Flags={result['recent_flags']}, Label: {result['label']}")