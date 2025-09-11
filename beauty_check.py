import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()import cv2
import mediapipe as mp
import numpy as np
import math
from deepface import DeepFace

# --- Optimization & Configuration ---
ANALYSIS_INTERVAL = 15 
frame_counter = 0
# This dictionary will hold the last calculated results to display on every frame
last_analysis_results = {}

# --- Initialize MediaPipe ---
mp_drawing = mp.solutions.drawing_utils #type: ignore
mp_face_mesh = mp.solutions.face_mesh #type: ignore

# --- Define Custom Drawing Styles for a Cleaner Look ---
TESSELATION_STYLE = mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=0)
CONTOUR_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
IRIS_STYLE = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)

# --- Helper & Calculation Functions ---
def get_distance(p1, p2, image):
    """Calculates the Euclidean distance between two MediaPipe landmarks."""
    h, w, _ = image.shape
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_beauty_metrics(landmarks, image):
    """Calculates symmetry and proportion scores based on facial landmarks."""
    h, w, _ = image.shape
    center_line_x = landmarks[1].x * w
    symmetry_pairs = [(359, 130), (291, 61), (287, 57), (454, 234)]
    symmetry_scores = []
    for p_left_idx, p_right_idx in symmetry_pairs:
        dist_left = abs(center_line_x - (landmarks[p_left_idx].x * w))
        dist_right = abs((landmarks[p_right_idx].x * w) - center_line_x)
        total_dist = dist_left + dist_right
        if total_dist > 0:
            score = 1.0 - (abs(dist_left - dist_right) / total_dist)
            symmetry_scores.append(score)
    avg_symmetry_score = (sum(symmetry_scores) / len(symmetry_scores)) * 100 if symmetry_scores else 0
    
    face_height = get_distance(landmarks[10], landmarks[152], image)
    face_width = get_distance(landmarks[234], landmarks[454], image)
    height_width_ratio = face_height / face_width if face_width > 0 else 0

    eye_width = get_distance(landmarks[133], landmarks[130], image)
    eye_separation = get_distance(landmarks[133], landmarks[362], image)
    eye_ratio = eye_separation / eye_width if eye_width > 0 else 0
    
    return {
        "symmetry_score": avg_symmetry_score, "height_width_ratio": height_width_ratio, "eye_ratio": eye_ratio
    }

def get_face_shape(landmarks):
    """Approximates face shape based on landmark ratios."""
    face_top = landmarks[10]; chin = landmarks[152]; jaw_left = landmarks[172]
    jaw_right = landmarks[397]; forehead_left = landmarks[70]; forehead_right = landmarks[300]
    face_length = np.linalg.norm(np.array([face_top.x, face_top.y]) - np.array([chin.x, chin.y]))
    jaw_width = np.linalg.norm(np.array([jaw_left.x, jaw_left.y]) - np.array([jaw_right.x, jaw_right.y]))
    forehead_width = np.linalg.norm(np.array([forehead_left.x, forehead_left.y]) - np.array([forehead_right.x, forehead_right.y]))
    if forehead_width == 0 or jaw_width == 0: return "Unknown"
    jaw_to_forehead = jaw_width / forehead_width; height_to_width = face_length / jaw_width
    if height_to_width > 1.6: return "Oval"
    elif jaw_to_forehead > 0.95: return "Square"
    elif jaw_to_forehead < 0.85: return "Heart"
    else: return "Round"

# --- Main Application Logic ---
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Starting webcam feed. Press 'q' in the video window to exit.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    frame_counter += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_mesh = face_mesh.process(image_rgb)
    
    if results_mesh.multi_face_landmarks:
        face_landmarks_obj = results_mesh.multi_face_landmarks[0]
        all_landmarks = face_landmarks_obj.landmark

        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=TESSELATION_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=CONTOUR_STYLE)
        mp_drawing.draw_landmarks(image=image, landmark_list=face_landmarks_obj, connections=mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None, connection_drawing_spec=IRIS_STYLE)

        # --- Perform Heavy Analysis Periodically ---
        if frame_counter % ANALYSIS_INTERVAL == 0:
            # Create a NEW dictionary for this frame's analysis
            current_results = {}
            
            # 1. Calculate base metrics and add them to the new dictionary
            beauty_metrics = calculate_beauty_metrics(all_landmarks, image)
            current_results.update(beauty_metrics)
            current_results['shape'] = get_face_shape(all_landmarks)
            
            # 2. Perform DeepFace analysis and add results
            try:
                analysis = DeepFace.analyze(image_rgb, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
                
                # Check that DeepFace returned a valid list with a dictionary inside
                if isinstance(analysis, list) and len(analysis) > 0 and isinstance(analysis[0], dict):
                    face_data_dict = analysis[0]
                    current_results['emotion'] = face_data_dict.get('dominant_emotion', 'N/A')
                    current_results['gender'] = face_data_dict.get('dominant_gender', 'N/A')
                else:
                    # Handle cases where DeepFace finds no faces
                    current_results['emotion'] = "Not Found"
                    current_results['gender'] = "Not Found"

            except Exception as e:
                # This happens if DeepFace throws an error during analysis
                print(f"DeepFace analysis error: {e}")
                current_results['emotion'] = "Error"
                current_results['gender'] = "Error"

            # 3. CRUCIAL STEP: Overwrite the global variable with the newly created dictionary
            last_analysis_results = current_results

    # --- Display Logic ---
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 240), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    y_pos = 30
    cv2.putText(image, "Facial Analysis", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # This check now works reliably because last_analysis_results is always a dict
    if last_analysis_results:
        gender = last_analysis_results.get('gender', 'Analyzing...')
        emotion = last_analysis_results.get('emotion', 'Analyzing...')
        shape = last_analysis_results.get('shape', 'Analyzing...')
        sym_score = last_analysis_results.get("symmetry_score", 0)
        hw_ratio = last_analysis_results.get("height_width_ratio", 0)
        eye_ratio = last_analysis_results.get("eye_ratio", 0)
        
        cv2.putText(image, f"Gender: {gender}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Emotion: {emotion}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        cv2.putText(image, f"Face Shape: {shape}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 40
        cv2.putText(image, f"Symmetry: {sym_score:.2f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"H/W Ratio: {hw_ratio:.2f} (Ideal ~1.62)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(image, f"Eye Ratio: {eye_ratio:.2f} (Ideal ~1.0)", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "No Face Detected or Analyzing...", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Advanced Face Analyzer - Press Q to Quit', image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

print("Closing application.")
cap.release()
cv2.destroyAllWindows()