# recognize_faces.py
import cv2
import os
import pickle
import numpy as np

# --- Configuration ---
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
MODEL_FILENAME = 'model.pkl' # Model file from training
LABELS_FILENAME = 'labels.pkl' # Labels file from training
RECOGNITION_THRESHOLD = 75 # LBPH Confidence threshold (lower is better match)
                           # Adjust this based on testing. Values ~ 50-100 often work.
IMG_WIDTH = 200 # Must match the training image size
IMG_HEIGHT = 200 # Must match the training image size
# ---------------------

def recognize_faces_realtime():
    """Performs real-time face detection and recognition using webcam."""

    # --- Load Haar Cascade ---
    if not os.path.exists(HAAR_CASCADE_PATH):
        print(f"Error: Haar Cascade file not found at {HAAR_CASCADE_PATH}")
        return
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    # --- Load the trained model and labels ---
    if not os.path.exists(MODEL_FILENAME) or not os.path.exists(LABELS_FILENAME):
        print(f"Error: Model ('{MODEL_FILENAME}') or Labels ('{LABELS_FILENAME}') file not found.")
        print("Please run train_model.py first.")
        return

    # Load the LBPH recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILENAME) # Use read() as we saved with write()

    # Load the id -> name mapping
    with open(LABELS_FILENAME, 'rb') as f:
        labels = pickle.load(f) # labels is now a dict {id: name}

    # --- Initialize Webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nStarting real-time face recognition...")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # --- Process each detected face ---
        for (x, y, w, h) in faces:
            # Extract the face ROI
            face_roi_gray = gray[y:y+h, x:x+w]

            # Resize ROI to match the training image size
            try:
                resized_face = cv2.resize(face_roi_gray, (IMG_WIDTH, IMG_HEIGHT))
            except cv2.error as e:
                 # Handle cases where ROI might be invalid momentarily
                 # print(f"Resize error: {e}")
                 continue


            # --- Perform Recognition ---
            label_id, confidence = recognizer.predict(resized_face)

            # --- Interpret Results ---
            # LBPH confidence is a distance measure. Lower values = better match.
            if confidence < RECOGNITION_THRESHOLD:
                person_name = labels.get(label_id, "Unknown") # Get name from ID
                display_text = f"{person_name} ({confidence:.2f})"
                color = (0, 255, 0) # Green for recognized
            else:
                # Confidence is too high (poor match)
                person_name = "Unknown"
                display_text = f"{person_name} ({confidence:.2f})"
                color = (0, 0, 255) # Red for unknown

            # --- Draw rectangle and text on the frame ---
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display the resulting frame
        cv2.imshow('Real-time Face Recognition - Press "q" to quit', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Recognition stopped.")

if __name__ == "__main__":
    recognize_faces_realtime()