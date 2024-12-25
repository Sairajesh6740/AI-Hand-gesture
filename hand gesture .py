import cv2
import mediapipe as mp
import os
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow

# Function to read the image from the uploaded path
def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image could not be loaded from path: {image_path}")
    return image

# Function to initialize the MediaPipe Hand module
def initialize_mediapipe_hands():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9,
        static_image_mode=True
    )
    mp_draw = mp.solutions.drawing_utils
    return hands, mp_draw

# Function to detect the hand gesture based on landmarks
def detect_hand_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

    if thumb_tip.y < index_tip.y:
        return "Thumbs Up"

    palm_base = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
    if all(landmark.y > palm_base.y for landmark in hand_landmarks.landmark[:5]):
        return "Fist"

    return "Peace Gesture"

# Function to draw landmarks on the image
def draw_landmarks_on_image(image, result, mp_draw):
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    return image

# Function to display the image with gesture text
def display_image_with_gesture(image, gesture):
    cv2.putText(image, f"Detected Gesture: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2_imshow(image)

# Main function
def main():
    uploaded = files.upload()
    image_path = next(iter(uploaded))

    try:
        image = read_image(image_path)
    except ValueError as e:
        print(e)
        return

    hands, mp_draw = initialize_mediapipe_hands()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)

    gesture = "Peace Gesture"
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            gesture = detect_hand_gesture(hand_landmarks)

    final_image = draw_landmarks_on_image(image, result, mp_draw)
    display_image_with_gesture(final_image, gesture)

if _name_ == "_main_":
    main()