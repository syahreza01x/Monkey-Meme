import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Face Detection and Hands
mp_face_detection = mp.solutions.face_detection
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

def get_gesture_state(face_landmarks, hand_landmarks, frame_width, frame_height):
    """
    Determine gesture state based on finger position relative to face
    Returns: 'standby', 'confuse', or 'know'
    """
    if face_landmarks is None or hand_landmarks is None:
        return 'standby'
    
    # Get face bounding box
    face_bbox = face_landmarks.location_data.relative_bounding_box
    face_x = face_bbox.xmin * frame_width
    face_y = face_bbox.ymin * frame_height
    face_w = face_bbox.width * frame_width
    face_h = face_bbox.height * frame_height
    
    # Get mouth position (approximately lower third of face)
    mouth_y_min = face_y + face_h * 0.55
    mouth_y_max = face_y + face_h * 0.85
    mouth_x_min = face_x + face_w * 0.25
    mouth_x_max = face_x + face_w * 0.75
    
    # Get right side of head position (for "I know" gesture)
    right_side_x_min = face_x + face_w * 0.85
    right_side_x_max = face_x + face_w * 1.4
    right_side_y_min = face_y - face_h * 0.1
    right_side_y_max = face_y + face_h * 0.5
    
    # Check index finger position (landmark 8)
    for hand in hand_landmarks:
        index_finger_tip = hand.landmark[8]
        index_finger_pip = hand.landmark[6]  # Middle joint of index finger
        wrist = hand.landmark[0]
        
        finger_x = index_finger_tip.x * frame_width
        finger_y = index_finger_tip.y * frame_height
        finger_pip_y = index_finger_pip.y * frame_height
        wrist_y = wrist.y * frame_height
        
        # Check if finger is at mouth area
        if (mouth_x_min <= finger_x <= mouth_x_max and 
            mouth_y_min <= finger_y <= mouth_y_max):
            return 'confuse'
        
        # Check if finger is at right side of head AND finger is pointing up (straight)
        # Finger is straight if tip is higher than pip joint and pip is higher than wrist
        is_finger_straight = (finger_y < finger_pip_y and finger_pip_y < wrist_y)
        
        if (is_finger_straight and
            right_side_x_min <= finger_x <= right_side_x_max and 
            right_side_y_min <= finger_y <= right_side_y_max):
            return 'know'
    
    return 'standby'

# Main loop
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    print("Program started. Press 'q' to quit.")
    print("Gestures:")
    print("- Normal face: Standby image")
    print("- Finger at mouth: Confuse image")
    print("- Straight finger at right side of head: Know image (like 'I know' gesture)")
    
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
        
        # Determine current state
        current_state = get_gesture_state(
            face_landmarks, 
            hand_landmarks_list if hand_landmarks_list else None,
            frame_width, 
            frame_height
        )
        
        # Select appropriate image based on state
        if current_state == 'confuse':
            display_img = img_confuse
            state_text = "State: CONFUSE"
            state_color = (0, 165, 255)  # Orange
        elif current_state == 'know':
            display_img = img_know
            state_text = "State: KNOW"
            state_color = (0, 255, 0)  # Green
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
