# landmarks.py - Module B: Facial Landmark Extraction
"""
Extracts 468 facial landmarks using MediaPipe Face Mesh.
Returns key points needed for micro-expression analysis.
"""

import cv2
import mediapipe as mp
import numpy as np

class LandmarkExtractor:
    def __init__(self):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Define key landmark indices for micro-expressions
        self.KEY_LANDMARKS = {
            # Eyes (for blink detection)
            'left_eye': [33, 160, 158, 133, 153, 144],  # Upper and lower eyelid
            'right_eye': [362, 385, 387, 263, 373, 380],
            
            # Mouth (for speech sync analysis)
            'mouth_outer': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308],
            'mouth_inner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324],
            
            # Eyebrows (for expression analysis)
            'left_eyebrow': [70, 63, 105, 66, 107],
            'right_eyebrow': [336, 296, 334, 293, 300],
            
            # Nose (for head pose)
            'nose': [1, 2, 98, 327],
            
            # Face contour (for head movement)
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        }
    
    def extract_landmarks(self, frame):
        """
        Extract facial landmarks from a frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            dict with landmarks or None if no face detected
        """
        if frame is None:
            return None
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face (we only track one face)
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract coordinates
        h, w, _ = frame.shape
        landmarks_dict = {}
        
        # Store all 468 landmarks as numpy array
        all_landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            z = landmark.z  # Depth information
            all_landmarks.append([x, y, z])
        
        landmarks_dict['all_points'] = np.array(all_landmarks)
        
        # Extract key regions
        for region_name, indices in self.KEY_LANDMARKS.items():
            region_points = []
            for idx in indices:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z
                region_points.append([x, y, z])
            landmarks_dict[region_name] = np.array(region_points)
        
        # Add frame dimensions for reference
        landmarks_dict['frame_width'] = w
        landmarks_dict['frame_height'] = h
        
        return landmarks_dict
    
    def draw_landmarks(self, frame, landmarks_dict):
        """
        Draw landmarks on frame for visualization.
        
        Args:
            frame: Original frame
            landmarks_dict: Dictionary from extract_landmarks()
            
        Returns:
            Frame with landmarks drawn
        """
        if landmarks_dict is None:
            return frame
        
        annotated_frame = frame.copy()
        
        # Draw key regions with different colors
        colors = {
            'left_eye': (0, 255, 0),      # Green
            'right_eye': (0, 255, 0),     # Green
            'mouth_outer': (255, 0, 0),   # Blue
            'mouth_inner': (0, 0, 255),   # Red
            'left_eyebrow': (255, 255, 0), # Cyan
            'right_eyebrow': (255, 255, 0), # Cyan
            'nose': (255, 0, 255),        # Magenta
        }
        
        for region_name, color in colors.items():
            if region_name in landmarks_dict:
                points = landmarks_dict[region_name][:, :2].astype(np.int32)
                
                # Draw points
                for point in points:
                    cv2.circle(annotated_frame, tuple(point), 2, color, -1)
                
                # Draw connecting lines
                for i in range(len(points) - 1):
                    cv2.line(annotated_frame, tuple(points[i]), 
                            tuple(points[i + 1]), color, 1)
                
                # Close the loop for eyes and mouth
                if 'eye' in region_name or 'mouth' in region_name:
                    cv2.line(annotated_frame, tuple(points[-1]), 
                            tuple(points[0]), color, 1)
        
        return annotated_frame
    
    def get_eye_aspect_ratio(self, eye_points):
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        EAR = (vertical distance) / (horizontal distance)
        Low EAR indicates closed eye (blink).
        
        Args:
            eye_points: numpy array of eye landmark coordinates
            
        Returns:
            float: EAR value
        """
        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # EAR formula
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def get_mouth_aspect_ratio(self, mouth_points):
        """
        Calculate Mouth Aspect Ratio (MAR) for speech detection.
        High MAR indicates open mouth.
        
        Args:
            mouth_points: numpy array of mouth landmark coordinates
            
        Returns:
            float: MAR value
        """
        # Vertical distance (top to bottom)
        vertical = np.linalg.norm(mouth_points[3] - mouth_points[9])
        
        # Horizontal distance (left to right)
        horizontal = np.linalg.norm(mouth_points[0] - mouth_points[6])
        
        # MAR formula
        mar = vertical / horizontal
        return mar
    
    def get_head_pose(self, landmarks_dict):
        """
        Estimate head pose (pitch, yaw, roll) using nose and face oval.
        
        Args:
            landmarks_dict: Dictionary from extract_landmarks()
            
        Returns:
            dict: {'pitch': float, 'yaw': float, 'roll': float} in degrees
        """
        if 'nose' not in landmarks_dict or 'face_oval' not in landmarks_dict:
            return None
        
        nose_points = landmarks_dict['nose']
        face_points = landmarks_dict['face_oval']
        
        # Calculate face center
        face_center = np.mean(face_points[:, :2], axis=0)
        
        # Calculate nose tip relative to center
        nose_tip = nose_points[0][:2]
        
        # Simple yaw estimation (left-right head turn)
        yaw = (nose_tip[0] - face_center[0]) / landmarks_dict['frame_width'] * 100
        
        # Simple pitch estimation (up-down head tilt)
        pitch = (nose_tip[1] - face_center[1]) / landmarks_dict['frame_height'] * 100
        
        # Roll estimation (head tilt to side) using eye line
        if 'left_eye' in landmarks_dict and 'right_eye' in landmarks_dict:
            left_eye_center = np.mean(landmarks_dict['left_eye'][:, :2], axis=0)
            right_eye_center = np.mean(landmarks_dict['right_eye'][:, :2], axis=0)
            
            # Calculate angle of eye line
            eye_vector = right_eye_center - left_eye_center
            roll = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
        else:
            roll = 0
        
        return {
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll
        }


def extract_landmarks(frame):
    """
    Convenience function for pipeline integration.
    Creates a persistent extractor instance if needed.
    """
    if not hasattr(extract_landmarks, 'extractor'):
        extract_landmarks.extractor = LandmarkExtractor()
    
    return extract_landmarks.extractor.extract_landmarks(frame)


# Test code
if __name__ == '__main__':
    # Test with webcam
    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit, 's' to save screenshot")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract landmarks
        landmarks = extractor.extract_landmarks(frame)
        
        if landmarks is not None:
            # Draw landmarks
            annotated_frame = extractor.draw_landmarks(frame, landmarks)
            
            # Calculate and display metrics
            left_ear = extractor.get_eye_aspect_ratio(landmarks['left_eye'])
            right_ear = extractor.get_eye_aspect_ratio(landmarks['right_eye'])
            mar = extractor.get_mouth_aspect_ratio(landmarks['mouth_outer'])
            head_pose = extractor.get_head_pose(landmarks)
            
            # Display metrics on frame
            cv2.putText(annotated_frame, f"Left EAR: {left_ear:.3f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Right EAR: {right_ear:.3f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"MAR: {mar:.3f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            if head_pose:
                cv2.putText(annotated_frame, 
                           f"Yaw: {head_pose['yaw']:.1f} Pitch: {head_pose['pitch']:.1f}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('FaceTrap - Landmark Detection', annotated_frame)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('FaceTrap - Landmark Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('landmark_test.jpg', annotated_frame if landmarks else frame)
            print("Screenshot saved!")
    
    cap.release()
    cv2.destroyAllWindows()
