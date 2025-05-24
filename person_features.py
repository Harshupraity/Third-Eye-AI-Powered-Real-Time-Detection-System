import cv2
import numpy as np
from imutils.video import VideoStream
import imutils
import time

# Load pre-trained gender detection model
gender_net = cv2.dnn.readNetFromCaffe(
    'gender_deploy.prototxt',
    'gender_net.caffemodel')
GENDER_LIST = ["Male", "Female"]

# Load Haar face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam stream
vs = VideoStream(src=0).start()
time.sleep(2.0)

def detect_dominant_color(image):
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    n_colors = 1
    _, labels, centers = cv2.kmeans(
        pixels, n_colors, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS
    )
    dominant_color = centers[0].astype("uint8")
    return dominant_color

def classify_skin_tone(rgb):
    r, g, b = rgb
    if r > 190:
        return "Very Fair"
    elif r > 160:
        return "Fair"
    elif r > 135:
        return "Wheatish"
    elif r > 100:
        return "Brown"
    elif r > 70:
        return "Dark Brown"
    else:
        return "Very Dark"

def classify_hair_color(rgb):
    r, g, b = rgb
    if r > 200 and g > 200 and b > 200:
        return "White"
    elif r > 160 and g > 100 and b > 80:
        return "Light Brown"
    elif r > 100 and g > 60 and b > 40:
        return "Brown"
    elif r < 80 and g < 80 and b < 80:
        return "Black"
    else:
        return "Unknown"

def classify_beard(face_img):
    h, w = face_img.shape[:2]
    beard_region = face_img[int(h*0.6):, int(w*0.25):int(w*0.75)]
    gray_beard = cv2.cvtColor(beard_region, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_beard, cv2.CV_64F)
    variance = laplacian.var()

    if variance < 50:
        return "No Beard"
    elif variance < 150:
        return "Mild Beard"
    else:
        return "Dense Beard"

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.426337, 87.768914, 114.895847), swapRB=False)

        # Gender prediction
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]

        # Beard classification
        beard = classify_beard(face_img)

        # Hair color detection
        hair_region = face_img[:int(h*0.3), :]
        hair_color = classify_hair_color(detect_dominant_color(hair_region))

        # Skin tone detection
        skin_region = face_img[int(h*0.4):int(h*0.6), int(w*0.3):int(w*0.7)]
        skin_tone = classify_skin_tone(detect_dominant_color(skin_region))

        # Display info
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        info = f"{gender}, {beard}, Hair: {hair_color}, Skin: {skin_tone}"
        cv2.putText(frame, info, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)

    cv2.imshow("Person Features Detector", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vs.stop()
cv2.destroyAllWindows()
