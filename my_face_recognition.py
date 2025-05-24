import face_recognition
import cv2
import pickle
import numpy as np
import pyttsx3
import time
import os
import requests
from datetime import datetime

# Load known face encodings
with open("face_encodings.pkl", "rb") as f:
    known_faces = pickle.load(f)

known_face_encodings = list(known_faces.values())
known_face_names = list(known_faces.keys())

# Telegram bot config
BOT_TOKEN = "8190992615:AAEDMz9SY9UsAzS8rmmsF0bx_te-3DGtyKg"
CHAT_ID = "711184137"

# Initialize speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Track last spoken time for each person
last_spoken = {}

# Function to send photo via Telegram
def send_telegram_photo(image_path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(image_path, 'rb') as photo:
        files = {'photo': photo}
        data = {'chat_id': CHAT_ID, 'caption': 'Unknown person detected!'}
        response = requests.post(url, files=files, data=data)
        print("Telegram response:", response.json())

# Initialize camera
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(distances)

        # Adjust this threshold for sensitivity (lower = stricter match)
        threshold = 0.45
        name = "Unknown"

        if distances[best_match_index] < threshold:
            name = known_face_names[best_match_index]

        # Scale back face location to original size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle and name
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        # Speak name if not spoken in last 10 seconds
        now = time.time()
        if name not in last_spoken or now - last_spoken[name] > 10:
            engine.say(f"{name} detected")
            engine.runAndWait()
            last_spoken[name] = now

            if name == "Unknown":
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                image_path = f"unknown_{timestamp}.jpg"
                cv2.imwrite(image_path, frame)
                print(f"Unknown face detected. Image saved as {image_path}")

                # Send photo via Telegram
                send_telegram_photo(image_path)

    # Show frame
    cv2.imshow("Enhanced Face Recognition", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
