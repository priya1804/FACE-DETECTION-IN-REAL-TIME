# recognize_from_image.py
import cv2
import os
import pickle
import numpy as np
import argparse # For command-line arguments

# --- Configuration ---
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
MODEL_FILENAME = 'model.pkl'       # Model file from training
LABELS_FILENAME = 'labels.pkl'     # Labels file from training
RECOGNITION_THRESHOLD = 75         # LBPH Confidence threshold (lower is better)
                                   # Adjust based on testing.
IMG_WIDTH = 200                    # Must match the training image size
IMG_HEIGHT = 200                   # Must match the training image size
OUTPUT_IMAGE_FILENAME = 'recognized_output.jpg' # Name for the saved output image
# ---------------------

def recognize_faces_in_image(image_path, output_path=OUTPUT_IMAGE_FILENAME):
    """
    Detects and recognizes faces in a given image file.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the image with detections.
    """

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
    try:
        recognizer.read(MODEL_FILENAME)
    except cv2.error as e:
        print(f"Error loading model file '{MODEL_FILENAME}': {e}")
        print("Ensure it's a valid OpenCV recognizer model file (usually .yml or .xml, even if named .pkl here).")
        return


    # Load the id -> name mapping
    try:
        with open(LABELS_FILENAME, 'rb') as f:
            labels = pickle.load(f) # labels is now a dict {id: name}
    except FileNotFoundError:
        print(f"Error: Labels file '{LABELS_FILENAME}' not found.")
        return
    except Exception as e:
        print(f"Error loading labels file '{LABELS_FILENAME}': {e}")
        return

    # --- Load and Process the Input Image ---
    if not os.path.exists(image_path):
        print(f"Error: Input image file not found at {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image from {image_path}")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30) # Minimum face size to detect
    )

    if len(faces) == 0:
        print("No faces detected in the image.")
    else:
        print(f"Detected {len(faces)} face(s).")

    # --- Process each detected face ---
    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi_gray = gray[y:y+h, x:x+w]

        # Resize ROI to match the training image size
        try:
            resized_face = cv2.resize(face_roi_gray, (IMG_WIDTH, IMG_HEIGHT))
        except cv2.error as e:
            print(f"Warning: Could not resize detected face region. Skipping. Error: {e}")
            continue # Skip this face if resizing fails

        # --- Perform Recognition ---
        label_id, confidence = recognizer.predict(resized_face)

        # --- Interpret Results ---
        # LBPH confidence is a distance measure. Lower values = better match.
        if confidence < RECOGNITION_THRESHOLD:
            person_name = labels.get(label_id, "Unknown") # Get name from ID
            display_text = f"{person_name} ({confidence:.2f})"
            color = (0, 255, 0) # Green for recognized
        else:
            # Confidence is too high (poor match) or label_id not in labels
            person_name = "Unknown"
            display_text = f"{person_name} ({confidence:.2f})"
            color = (0, 0, 255) # Red for unknown

        # --- Draw rectangle and text on the original color frame ---
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display the resulting image
    cv2.imshow('Face Recognition in Image', frame)
    print(f"\nDisplaying image. Press any key to close and save to '{output_path}'.")
    cv2.waitKey(0) # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()

    # Save the output image
    try:
        cv2.imwrite(output_path, frame)
        print(f"Output image saved to {output_path}")
    except Exception as e:
        print(f"Error saving output image: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and recognize faces in an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument("--output", type=str, default=OUTPUT_IMAGE_FILENAME,
                        help=f"Path to save the output image with detections (default: {OUTPUT_IMAGE_FILENAME}).")

    args = parser.parse_args()
    recognize_faces_in_image(args.image_path, args.output)