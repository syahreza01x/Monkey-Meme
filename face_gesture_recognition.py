import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Face Detection and Hands
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set window size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load images
def load_image(path, width=300, height=300):
    """Load and resize image"""
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (width, height))
            return img
    # Create placeholder if image doesn't exist
    placeholder = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Image Not Found", (20, height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return placeholder

# Load all images
img_standby = load_image('img/standby.jpg')
img_confuse = load_image('img/confused.jpg')
img_know = load_image('img/know.jpg')
img_shock = load_image('img/shock.jpg')

def get_gesture_state(face_landmarks, hand_landmarks, frame_width, frame_height, face_mesh_landmarks=None):
    """
    Determine gesture state based on finger position relative to face
    Returns: 'standby', 'confuse', 'know', or 'shock'
    """
    if face_landmarks is None:
        return 'standby', None
    
    # Get face bounding box
    face_bbox = face_landmarks.location_data.relative_bounding_box
    face_x = face_bbox.xmin * frame_width
    face_y = face_bbox.ymin * frame_height
    face_w = face_bbox.width * frame_width
    face_h = face_bbox.height * frame_height
    
    # Check if mouth is open using face mesh landmarks
    is_mouth_open = False
    if face_mesh_landmarks is not None:
        # Upper lip: landmark 13, Lower lip: landmark 14
        # More accurate: Upper lip center: 13, Lower lip center: 14
        upper_lip = face_mesh_landmarks.landmark[13]
        lower_lip = face_mesh_landmarks.landmark[14]
        
        # Calculate vertical distance between upper and lower lip
        mouth_distance = abs(lower_lip.y - upper_lip.y)
        
        # If mouth distance is greater than threshold, mouth is open
        if mouth_distance > 0.02:  # Threshold for open mouth
            is_mouth_open = True
    
    # Get mouth position (approximately lower third of face)
    mouth_y_min = face_y + face_h * 0.55
    mouth_y_max = face_y + face_h * 0.85
    mouth_x_min = face_x + face_w * 0.25
    mouth_x_max = face_x + face_w * 0.75
    
    # Get right side of head position (for "I know" gesture)
    right_side_x_min = face_x + face_w * 1.1
    right_side_x_max = face_x + face_w * 1.7
    right_side_y_min = face_y - face_h * 0.1
    right_side_y_max = face_y + face_h * 0.5
    
    # Get bottom of head position (for "shock" gesture - below chin)
    bottom_x_min = face_x - face_w * 0.2
    bottom_x_max = face_x + face_w * 1.2
    bottom_y_min = face_y + face_h * 1.05
    bottom_y_max = face_y + face_h * 1.8
    
    # Detection areas for visualization
    detection_areas = {
        'know_area': (int(right_side_x_min), int(right_side_y_min), 
                     int(right_side_x_max), int(right_side_y_max)),
        'shock_area': (int(bottom_x_min), int(bottom_y_min), 
                      int(bottom_x_max), int(bottom_y_max))
    }
    
    if hand_landmarks is None:
        return 'standby', detection_areas
    
    # Count hands and fingers in shock area (below head)
    hands_in_shock_area = 0
    total_fingers_in_shock = 0
    hand_centers_in_shock = []
    
    # Check index finger position (landmark 8) and hand positions
    for hand in hand_landmarks:
        index_finger_tip = hand.landmark[8]
        index_finger_pip = hand.landmark[6]  # Middle joint of index finger
        wrist = hand.landmark[0]
        middle_finger_tip = hand.landmark[12]
        palm_center = hand.landmark[9]  # Center of palm
        
        finger_x = index_finger_tip.x * frame_width
        finger_y = index_finger_tip.y * frame_height
        finger_pip_y = index_finger_pip.y * frame_height
        wrist_y = wrist.y * frame_height
        wrist_x = wrist.x * frame_width
        palm_x = palm_center.x * frame_width
        palm_y = palm_center.y * frame_height
        
        # Check if hand (palm center) is in shock area (below head)
        if (bottom_x_min <= palm_x <= bottom_x_max and 
            bottom_y_min <= palm_y <= bottom_y_max):
            hands_in_shock_area += 1
            hand_centers_in_shock.append((palm_x, palm_y))
            
            # Count visible finger tips in shock area
            finger_tips = [8, 12, 16, 20, 4]  # Index, middle, ring, pinky, thumb
            for tip_id in finger_tips:
                tip = hand.landmark[tip_id]
                tip_x = tip.x * frame_width
                tip_y = tip.y * frame_height
                if (bottom_x_min <= tip_x <= bottom_x_max and 
                    bottom_y_min <= tip_y <= bottom_y_max):
                    total_fingers_in_shock += 1
        
        # Check if finger is at mouth area (only check if not in shock area)
        if (mouth_x_min <= finger_x <= mouth_x_max and 
            mouth_y_min <= finger_y <= mouth_y_max):
            return 'confuse', detection_areas
        
        # Check if finger is at right side of head AND finger is pointing up (straight)
        # Finger is straight if tip is higher than pip joint and pip is higher than wrist
        is_finger_straight = (finger_y < finger_pip_y and finger_pip_y < wrist_y)
        
        if (is_finger_straight and
            right_side_x_min <= finger_x <= right_side_x_max and 
            right_side_y_min <= finger_y <= right_side_y_max):
            return 'know', detection_areas
    
    # Check for shock gesture:
    # Option 1: Both hands detected in shock area
    # Option 2: One hand in shock area with many fingers (5+) - indicating overlapping hands
    # Option 3: Multiple hand centers detected (even if MediaPipe merges them)
    # AND mouth must be open
    if hands_in_shock_area >= 1 and total_fingers_in_shock >= 5 and is_mouth_open:
        return 'shock', detection_areas
    
    return 'standby', detection_areas

# Main loop
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    print("Program started. Press 'q' to quit.")
    print("Gestures:")
    print("- Normal face: Standby image")
    print("- Finger at mouth: Confuse image")
    print("- Straight finger at right side of head: Know image (like 'I know' gesture)")
    print("- Both hands below head + mouth open: Shock image")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face
        face_results = face_detection.process(rgb_frame)
        
        # Detect face mesh for mouth detection
        face_mesh_results = face_mesh.process(rgb_frame)
        
        # Detect hands
        hand_results = hands.process(rgb_frame)
        
        # Draw face detection
        face_landmarks = None
        if face_results.detections:
            for detection in face_results.detections:
                mp_drawing.draw_detection(frame, detection)
                face_landmarks = detection
                break  # Use first detected face
        
        # Draw hand landmarks
        hand_landmarks_list = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_landmarks_list.append(hand_landmarks)
        
        # Get face mesh landmarks for mouth detection
        face_mesh_landmark = None
        if face_mesh_results.multi_face_landmarks:
            face_mesh_landmark = face_mesh_results.multi_face_landmarks[0]
        
        # Determine current state
        current_state, detection_areas = get_gesture_state(
            face_landmarks, 
            hand_landmarks_list if hand_landmarks_list else None,
            frame_width, 
            frame_height,
            face_mesh_landmark
        )
        
        # Draw detection area boxes if face is detected
        if detection_areas:
            # Draw "know" area box (right side of head) - Green
            know_area = detection_areas['know_area']
            cv2.rectangle(frame, 
                         (know_area[0], know_area[1]), 
                         (know_area[2], know_area[3]), 
                         (0, 255, 0), 2)  # Green box
            
            # Draw "shock" area box (below head) - Blue
            shock_area = detection_areas['shock_area']
            cv2.rectangle(frame, 
                         (shock_area[0], shock_area[1]), 
                         (shock_area[2], shock_area[3]), 
                         (255, 0, 0), 2)  # Blue box
        
        # Select appropriate image based on state
        if current_state == 'confuse':
            display_img = img_confuse
            state_text = "State: CONFUSE"
            state_color = (0, 165, 255)  # Orange
        elif current_state == 'know':
            display_img = img_know
            state_text = "State: KNOW"
            state_color = (0, 255, 0)  # Green
        elif current_state == 'shock':
            display_img = img_shock
            state_text = "State: SHOCK"
            state_color = (255, 0, 255)  # Magenta
        else:
            display_img = img_standby
            state_text = "State: STANDBY"
            state_color = (255, 255, 255)  # White
        
        # Create combined display
        img_height, img_width = display_img.shape[:2]
        combined_width = frame_width + img_width + 20
        combined_height = max(frame_height, img_height)
        
        # Create black canvas
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Place camera frame on the left
        combined_frame[0:frame_height, 0:frame_width] = frame
        
        # Place status image on the right
        y_offset = (combined_height - img_height) // 2
        combined_frame[y_offset:y_offset+img_height, 
                      frame_width+20:frame_width+20+img_width] = display_img
        
        # Add state text
        cv2.putText(combined_frame, state_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)
        
        # Display the combined frame
        cv2.imshow('Face Gesture Recognition', combined_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Program ended.")
