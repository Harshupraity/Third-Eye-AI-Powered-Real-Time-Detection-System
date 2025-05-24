import cv2
import numpy as np
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speaking speed

# Load class labels
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor", "phone", "pen", "face", "eye", "mouth", "hand", "finger"
]

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_ssd.caffemodel")

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

detected_objects = set()  # Prevents repeated announcements

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.3:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx] if idx < len(CLASSES) else f"Object {idx}"
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box
            color = (0, 255, 0) if label == "phone" else (255, 0, 0) if label == "person" else (0, 0, 255)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Announce detected object once per session
            if label not in detected_objects:
                engine.say(f"{label} detected")
                engine.runAndWait()
                detected_objects.add(label)

    # Show output
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
