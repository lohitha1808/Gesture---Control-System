import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from math import hypot
from pynput.keyboard import Controller, Key
import screen_brightness_control as sbc
from pynput.mouse import Button, Controller as MouseController

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.75)
Draw = mp.solutions.drawing_utils

# Initialize controllers
volume_control = Controller()
media_control = Controller()
mouse_control = MouseController()

# Start capturing video
cap = cv2.VideoCapture(0)
prev_index = None

# State to manage the pause action
pause_triggered = False

def count_fingers(landmarks):
    if len(landmarks) < 21: return 0
    fingers = [0] * 5
    fingers[0] = int(landmarks[4][1] < landmarks[3][1])
    for i in range(1, 5):
        fingers[i] = int(landmarks[i * 4 + 4][2] < landmarks[i * 4 + 3][2])
    return sum(fingers)

def calculate_distance(landmarks, a, b):
    if len(landmarks) <= b: return 0
    x1, y1 = landmarks[a][1], landmarks[a][2]
    x2, y2 = landmarks[b][1], landmarks[b][2]
    return hypot(x2 - x1, y2 - y1)

while True:
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    if results.multi_hand_landmarks:
        for handlm in results.multi_hand_landmarks:
            landmarks = [[_id, int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])] 
                         for _id, lm in enumerate(handlm.landmark)]
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)
            num_fingers = count_fingers(landmarks)
            distance_thumb_index = calculate_distance(landmarks, 4, 8)
            distance_index_middle = calculate_distance(landmarks, 8, 12)

            if num_fingers == 5:
                if prev_index:
                    index_x, index_y = landmarks[8][1], landmarks[8][2]
                    movement_distance = hypot(index_x - prev_index[0], index_y - prev_index[1])
                    if movement_distance > 10:
                        screen_width, screen_height = pyautogui.size()
                        mouse_x = int(index_x / frame.shape[1] * screen_width)
                        mouse_y = int(index_y / frame.shape[0] * screen_height)
                        pyautogui.moveTo(mouse_x, mouse_y)
                prev_index = (landmarks[8][1], landmarks[8][2])
                fingers_detected_as_5 = True
            elif (num_fingers == 4) and fingers_detected_as_5:
                mouse_control.click(Button.left)
                fingers_detected_as_5 = False  # Reset to avoid multiple clicks
            elif num_fingers == 2:
                brightness = np.clip(np.interp(distance_thumb_index, [150, 200], [0, 100]), 0, 100)
                sbc.set_brightness(int(brightness))
                volume = np.clip(np.interp(distance_index_middle, [75, 125], [0, 100]), 0, 100)
                if volume > 50:
                    volume_control.press(Key.media_volume_up)
                else:
                    volume_control.press(Key.media_volume_down)
            elif num_fingers == 0:
                pause_triggered = True
            elif num_fingers == 3 and pause_triggered:
                media_control.press(Key.media_play_pause)
                pause_triggered=False

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
