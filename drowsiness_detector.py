import cv2
import mediapipe as mp
from math import dist
import pygame
import pandas as pd
from datetime import datetime
print("MediaPipe version:", mp.__version__)  # Should print 0.10.21
# Initialize sound alert
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alarm.wav")  # Place alarm.wav in the same folder

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR calculator
def calculate_EAR(eye):
    A = dist(eye[1], eye[5])
    B = dist(eye[2], eye[4])
    C = dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Drowsiness parameters
EAR_THRESHOLD = 0.25
SCORE = 0
SCORE_THRESHOLD = 15
LOG_FILE = "drowsiness_log.csv"

# Create CSV log file with headers if not exists
try:
    pd.read_csv(LOG_FILE)
except FileNotFoundError:
    pd.DataFrame(columns=["Time", "EAR", "Score", "Alert"]).to_csv(LOG_FILE, index=False)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        left_eye = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in LEFT_EYE]
        right_eye = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in RIGHT_EYE]

        left_ear = calculate_EAR(left_eye)
        right_ear = calculate_EAR(right_eye)
        ear = (left_ear + right_ear) / 2

        # Update score
        if ear < EAR_THRESHOLD:
            SCORE += 1
        else:
            SCORE -= 1
            if SCORE < 0:
                SCORE = 0

        # Show EAR and Score
        cv2.putText(frame, f'EAR: {ear:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'SCORE: {SCORE}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

        # Drowsiness Alert
        alert_triggered = False
        if SCORE > SCORE_THRESHOLD:
            cv2.putText(frame, 'DROWSINESS ALERT!', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            pygame.mixer.Sound.play(alert_sound)
            alert_triggered = True

        # Log data to CSV
        log_entry = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "EAR": round(ear, 3),
            "Score": SCORE,
            "Alert": "Yes" if alert_triggered else "No"
        }
        pd.DataFrame([log_entry]).to_csv(LOG_FILE, mode='a', index=False, header=False)

        # Draw eyes
        for eye in [left_eye, right_eye]:
            for (x, y) in eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Driver Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()