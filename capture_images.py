# capture_images.py
import cv2
import os
import time

# --- Configuration ---
DATASET_DIR = 'dataset'
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
NUM_IMAGES_TO_CAPTURE = 50 # Number of images to capture per person
IMG_WIDTH = 200 # Resize captured faces to this width
IMG_HEIGHT = 200 # Resize captured faces to this height
CAPTURE_DELAY_SEC = 0.1 # Delay between captures to allow movement
# ---------------------

def ensure_dir(directory):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def capture_face_images(name):
    """Captures and saves face images for a given name."""
    person_dir = os.path.join(DATASET_DIR, name)
    ensure_dir(DATASET_DIR)
    ensure_dir(person_dir)

    # Load Haar Cascade for face detection
    if not os.path.exists(HAAR_CASCADE_PATH):
        print(f"Error: Haar Cascade file not found at {HAAR_CASCADE_PATH}")
        print("Download it from: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml")
        return
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    # Initialize webcam
    cap = cv2.VideoCapture(0) # 0 is usually the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"\nCapturing images for: {name}")
    print("Please look at the camera. Press 'q' to quit early.")

    img_count = 0
    while img_count < NUM_IMAGES_TO_CAPTURE:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # We expect only one face during capture
        if len(faces) == 1:
            (x, y, w, h) = faces[0]

            # Extract the face ROI (Region of Interest)
            face_roi_gray = gray[y:y+h, x:x+w]

            # --- Save the image ---
            img_count += 1
            # Resize for consistency
            resized_face = cv2.resize(face_roi_gray, (IMG_WIDTH, IMG_HEIGHT))
            img_path = os.path.join(person_dir, f"user.{name}.{img_count}.jpg")
            cv2.imwrite(img_path, resized_face)
            print(f"Saved {img_path} ({img_count}/{NUM_IMAGES_TO_CAPTURE})")

            # --- Draw rectangle and text on the main frame for feedback ---
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            status_text = f"Captured: {img_count}/{NUM_IMAGES_TO_CAPTURE}"
            cv2.putText(frame, status_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Small delay to allow slight pose changes
            time.sleep(CAPTURE_DELAY_SEC)

        elif len(faces) > 1:
             cv2.putText(frame, "Multiple faces detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
             cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Capturing Faces - Press "q" to quit', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture interrupted by user.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished capturing images for {name}.")

if __name__ == "__main__":
    person_name = input("Enter the name of the person: ").strip().replace(" ", "_")
    if not person_name:
        print("Error: Name cannot be empty.")
    else:
        capture_face_images(person_name)