import cv2
import pytesseract
import pyttsx3
import threading
import time
import queue
from pytesseract import Output

# Languages for OCR
languages = 'eng+hin+urd'

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)

# Shared state
last_text = ""
last_confidence = 0
last_ocr_time = 0
ocr_interval = 1.5  # seconds
text_lock = threading.Lock()
tts_queue = queue.Queue()

# Dedicated TTS thread
def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()

# Start TTS thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# OCR processing
def run_ocr(frame):
    global last_text, last_confidence

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Fast blur instead of bilateral
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # OCR
    data = pytesseract.image_to_data(gray, output_type=Output.DICT, lang=languages)

    new_text = ""
    confidence_scores = []

    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if word:
            try:
                conf = int(data['conf'][i])
                if conf > 70:
                    new_text += word + " "
                    confidence_scores.append(conf)
            except ValueError:
                continue

    avg_conf = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

    with text_lock:
        if new_text and new_text != last_text and avg_conf > 70:
            last_text = new_text.strip()
            last_confidence = avg_conf
            if tts_queue.empty():  # prevent speaking queue backup
                tts_queue.put(last_text)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot access the camera.")
    exit()

print("📷 OCR + TTS started — Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not captured.")
        break

    display_frame = cv2.resize(frame, (640, 480))

    # Run OCR every few seconds
    if time.time() - last_ocr_time > ocr_interval:
        threading.Thread(target=run_ocr, args=(frame.copy(),), daemon=True).start()
        last_ocr_time = time.time()

    with text_lock:
        if last_text:
            cv2.putText(display_frame, f"Confidence: {int(last_confidence)}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Text: {last_text[:40]}...", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Real-Time OCR (Press 'q' to Quit)", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
tts_queue.put(None)  # Stop TTS thread
tts_thread.join()
