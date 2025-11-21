import cv2
import time
import numpy as np

# --- IMPORTS UPDATED for the new parallel folder structure ---
# Import structure: from folder_name.file_name import ClassName
from caputre.capture import VideoCapture
from landmark.landmarks import LandmarkExtractor
from metrics.metrics import MetricsCalculator
from decision.decision import DecisionEngine

def draw_info_box(frame, detection_result):
    """Draws the final detection result and score on the frame."""
    
    label = detection_result.get('label', 'NO FACE')
    score = detection_result.get('raw_score', 0)
    
    # Define colors based on detection status
    if label == "POSSIBLE DEEPFAKE":
        color = (0, 0, 255) # Red
    elif label == "SUSPICIOUS":
        color = (0, 165, 255) # Orange
    elif label == "REAL":
        color = (0, 255, 0) # Green
    else:
        color = (255, 255, 255) # White

    # Define the text box area (bottom right)
    h, w, _ = frame.shape
    box_w, box_h = 350, 150
    start_x, start_y = w - box_w - 10, h - box_h - 10

    # Draw transparent background box
    overlay = frame.copy()
    cv2.rectangle(overlay, (start_x, start_y), (w - 10, h - 10), (30, 30, 30), -1)
    alpha = 0.5
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Corrected vertical placement using start_y
    text_y = start_y + 30
    
    # 1. Status Label (Largest Text)
    cv2.putText(frame, "STATUS:", (start_x + 10, start_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, label, (start_x + 150, start_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    text_y = start_y + 70
    
    # 2. Anomaly Score
    cv2.putText(frame, f"Anomaly Score: {score:.1f}", (start_x + 10, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    text_y += 25

    # 3. Key Metrics (for Explainability)
    metrics = detection_result.get('metrics_used', {})
    
    cv2.putText(frame, f"Blink Rate: {metrics.get('avg_blink_rate', 0.0):.2f} blinks/sec", 
                (start_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    text_y += 20
    
    cv2.putText(frame, f"Head Stability: {metrics.get('head_tilt_variance', 0.0):.2f} var", 
                (start_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    text_y += 20
    
    cv2.putText(frame, f"Light Var: {metrics.get('light_variance', 0.0):.2f}", 
                (start_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def main():
    """Main function to run the FaceTrap pipeline."""
    
    # --- 1. Initialization (Integrator/You) ---
    print("Initializing FaceTrap modules...")
    try:
        # Module A: Video Capture
        capture = VideoCapture(source=0) 
        # Module B: Landmark Extraction
        extractor = LandmarkExtractor()
        # Module C: Metrics Calculation
        metrics_calc = MetricsCalculator()
        # Module D: Decision Logic
        decision_engine = DecisionEngine(history_size=90) # 3 seconds of history at 30 FPS
        
        print("Initialization complete. Starting video loop. Press 'q' to quit.")
    except Exception as e:
        print(f"[FATAL ERROR] Failed to initialize FaceTrap: {e}")
        return

    # --- 2. Main Processing Loop ---
    for frame in capture.frame_generator():
        
        # --- Stage 1: Landmark Extraction (Member B's work) ---
        landmarks_dict = extractor.extract_landmarks(frame)

        if landmarks_dict is None:
            # No face detected: reset metrics to prevent false flags
            metrics_calc.frame_count = 0
            cv2.putText(frame, "Awaiting Face Detection...", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Use the NO FACE state for the UI box
            detection_result = {'label': 'NO FACE', 'raw_score': 0, 'metrics_used': {}}
        else:
            # Annotate frame with landmarks (optional visualization)
            annotated_frame = extractor.draw_landmarks(frame, landmarks_dict)
            frame = annotated_frame

            # --- Stage 2: Metrics Calculation (Member C's work) ---
            metrics = metrics_calc.calculate_metrics(landmarks_dict, frame)
            
            # --- Stage 3: Decision Logic (Member D's work) ---
            detection_result = decision_engine.analyze_metrics(metrics)

        # --- Stage 4: UI/Visualization (Member E's work - simplified here) ---
        draw_info_box(frame, detection_result)
        
        # Display the final processed frame
        cv2.imshow('FaceTrap - Real-Time Deepfake Detection', frame)
        
        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 3. Cleanup ---
    capture.release()
    cv2.destroyAllWindows()
    print("FaceTrap shut down successfully.")

if __name__ == "__main__":
    main()