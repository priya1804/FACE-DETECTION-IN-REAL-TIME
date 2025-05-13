# utils.py
import cv2
import os
import numpy as np
import pickle
import time

# --- Configuration (can be moved to Flask app.config later if needed) ---
DATASET_DIR = 'dataset'
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
MODEL_FILENAME = 'model.pkl'
LABELS_FILENAME = 'labels.pkl'

# Capture settings
NUM_IMAGES_TO_CAPTURE = 30 # Reduced for quicker web demo
IMG_WIDTH = 200
IMG_HEIGHT = 200
CAPTURE_DELAY_SEC = 0.2

# Recognition settings
RECOGNITION_THRESHOLD = 75

face_cascade = None
recognizer = None
id_to_label_map = None

def load_resources():
    """Loads Haar cascade, recognizer, and labels if they exist."""
    global face_cascade, recognizer, id_to_label_map

    if not os.path.exists(HAAR_CASCADE_PATH):
        print(f"Error: Haar Cascade file not found at {HAAR_CASCADE_PATH}")
        return False # Indicate failure
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    if os.path.exists(MODEL_FILENAME) and os.path.exists(LABELS_FILENAME):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        try:
            recognizer.read(MODEL_FILENAME)
            with open(LABELS_FILENAME, 'rb') as f:
                id_to_label_map = pickle.load(f)
            print("Recognizer and labels loaded successfully.")
            return True # Indicate success
        except Exception as e:
            print(f"Error loading model or labels: {e}")
            recognizer = None
            id_to_label_map = None
            return False # Indicate failure
    else:
        print("Model or labels not found. Please train the model first.")
        return False # Indicate that model/labels are not ready

# Call load_resources once when the module is imported to try and load them.
# It will print errors if files are missing, which is fine for now.
load_resources()


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_dir(DATASET_DIR) # Ensure dataset dir exists on startup

def capture_face_images_util(name):
    """Captures and saves face images for a given name. Returns status."""
    if face_cascade is None:
        return "Error: Face detector (Haar Cascade) not loaded."

    person_dir = os.path.join(DATASET_DIR, name)
    ensure_dir(person_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Could not open webcam."

    img_count = 0
    captured_count = 0
    max_attempts = NUM_IMAGES_TO_CAPTURE * 5 # Try a bit more to get enough good shots
    attempt_count = 0

    print(f"Starting capture for {name}...")
    messages = []

    while captured_count < NUM_IMAGES_TO_CAPTURE and attempt_count < max_attempts:
        ret, frame = cap.read()
        if not ret:
            messages.append("Error: Failed to capture frame.")
            break
        attempt_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            face_roi_gray = gray[y:y+h, x:x+w]
            resized_face = cv2.resize(face_roi_gray, (IMG_WIDTH, IMG_HEIGHT))
            
            img_count += 1
            img_path = os.path.join(person_dir, f"user.{name}.{img_count}.jpg")
            cv2.imwrite(img_path, resized_face)
            captured_count += 1
            messages.append(f"Captured image {captured_count}/{NUM_IMAGES_TO_CAPTURE} for {name}.")
            print(f"Captured {img_path}") # For server log
            time.sleep(CAPTURE_DELAY_SEC)
        elif len(faces) > 1:
            messages.append("Multiple faces detected. Please ensure only one face is visible.")
            time.sleep(0.5) # Give user time to react
        else:
            messages.append("No face detected or face too small.")
            time.sleep(0.1)

        # To avoid blocking Flask for too long if webcam isn't working or no face is found
        if attempt_count % 50 == 0:
            print(f"Capture attempt {attempt_count} for {name}...")


    cap.release()
    cv2.destroyAllWindows() # Important for releasing webcam if it was opened by this process

    if captured_count == NUM_IMAGES_TO_CAPTURE:
        return f"Successfully captured {captured_count} images for {name}."
    else:
        return f"Capture incomplete for {name}. Captured {captured_count}/{NUM_IMAGES_TO_CAPTURE} images. Check webcam and ensure single face visibility."


def train_model_util():
    """Trains the LBPH face recognizer. Returns status message."""
    global recognizer, id_to_label_map # Allow updating global instances
    face_samples = []
    ids = []
    label_map = {}
    current_id = 0
    messages = []

    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        return "Error: Dataset directory is empty. Please capture images first."

    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_path):
            continue
        messages.append(f"Processing images for: {person_name}")
        if person_name not in label_map:
            label_map[person_name] = current_id
            person_id = current_id
            current_id += 1

        for image_name in os.listdir(person_path):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            image_path = os.path.join(person_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                messages.append(f"Warning: Could not read {image_path}")
                continue
            face_samples.append(np.array(img, 'uint8'))
            ids.append(person_id)

    if not face_samples or not ids:
        return "Error: No valid face samples found to train the model."

    messages.append(f"Found {len(face_samples)} samples from {len(label_map)} persons.")
    messages.append("Training LBPH recognizer...")

    temp_recognizer = cv2.face.LBPHFaceRecognizer_create()
    temp_recognizer.train(face_samples, np.array(ids))
    temp_recognizer.write(MODEL_FILENAME)

    # Update global recognizer instance
    recognizer = temp_recognizer

    # Save label map (id -> name)
    id_to_label_map_temp = {v: k for k, v in label_map.items()}
    with open(LABELS_FILENAME, 'wb') as f:
        pickle.dump(id_to_label_map_temp, f)
    
    # Update global id_to_label_map
    id_to_label_map = id_to_label_map_temp

    messages.append(f"Training complete! Model saved to {MODEL_FILENAME}, labels to {LABELS_FILENAME}.")
    messages.append(f"Labels trained: {label_map}")
    return "\n".join(messages)


def recognize_faces_on_frame(frame):
    """Recognizes faces in a single frame. Returns frame with detections."""
    if face_cascade is None or recognizer is None or id_to_label_map is None:
        cv2.putText(frame, "Model not loaded", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

    for (x, y, w, h) in faces:
        face_roi_gray = gray[y:y+h, x:x+w]
        try:
            resized_face = cv2.resize(face_roi_gray, (IMG_WIDTH, IMG_HEIGHT))
            label_id, confidence = recognizer.predict(resized_face)

            if confidence < RECOGNITION_THRESHOLD:
                person_name = id_to_label_map.get(label_id, "Unknown")
                display_text = f"{person_name} ({confidence:.2f})"
                color = (0, 255, 0)
            else:
                display_text = f"Unknown ({confidence:.2f})"
                color = (0, 0, 255)
        except cv2.error: # Handle cases where ROI might be invalid momentarily
            display_text = "Processing..."
            color = (255, 0, 0)
            
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def recognize_faces_in_image_util(image_path):
    """
    Detects and recognizes faces in a given image file.
    Returns the path to the processed image and list of detected names.
    """
    if face_cascade is None or recognizer is None or id_to_label_map is None:
        return None, ["Error: Model or resources not loaded. Please train first."]

    frame = cv2.imread(image_path)
    if frame is None:
        return None, [f"Error: Could not read image from {image_path}"]

    processed_frame = recognize_faces_on_frame(frame.copy()) # Pass a copy

    # Save the processed image
    ensure_dir('static/recognized_images')
    filename = "processed_" + os.path.basename(image_path)
    output_image_path = os.path.join('static', 'recognized_images', filename)
    cv2.imwrite(output_image_path, processed_frame)

    # Extract names (simplified for this example)
    detected_persons = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    for (x,y,w,h) in faces:
        face_roi_gray = gray[y:y+h, x:x+w]
        try:
            resized_face = cv2.resize(face_roi_gray, (IMG_WIDTH, IMG_HEIGHT))
            label_id, confidence = recognizer.predict(resized_face)
            if confidence < RECOGNITION_THRESHOLD:
                detected_persons.append(id_to_label_map.get(label_id, "Unknown Person"))
            else:
                detected_persons.append("Unknown Person")
        except cv2.error:
            detected_persons.append("Error processing face")


    if not detected_persons and len(faces) > 0: # Faces detected but not recognized
        detected_persons.append("Unknown (no match or error)")
    elif not detected_persons and len(faces) == 0:
        detected_persons.append("No faces detected in image.")


    return output_image_path, detected_persons