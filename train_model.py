# train_model.py
import cv2
import os
import numpy as np
import pickle # Using pickle as requested for saving model state

# --- Configuration ---
DATASET_DIR = 'dataset'
MODEL_FILENAME = 'model.pkl' # Output model file
LABELS_FILENAME = 'labels.pkl' # Output labels file
# ---------------------

def train_face_recognizer():
    """Trains the LBPH face recognizer using images in the dataset directory."""
    print("Preparing data...")
    face_samples = []
    ids = []
    label_map = {} # Maps name -> id
    current_id = 0

    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' is empty or does not exist.")
        print("Please run capture_images.py first to collect face data.")
        return

    # Iterate through each person's directory in the dataset
    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        print(f"Processing images for: {person_name}")

        # Assign a unique numerical ID to this person
        if person_name not in label_map:
            label_map[person_name] = current_id
            person_id = current_id
            current_id += 1
        else:
            person_id = label_map[person_name] # Should ideally not happen if structured well

        # Iterate through images in the person's directory
        for image_name in os.listdir(person_path):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue # Skip non-image files

            image_path = os.path.join(person_path, image_name)
            try:
                # Load image in grayscale
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not read image {image_path}. Skipping.")
                    continue

                img_numpy = np.array(img, 'uint8')
                face_samples.append(img_numpy)
                ids.append(person_id)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

    if not face_samples or not ids:
        print("Error: No valid face samples found to train the model.")
        return

    print(f"\nFound {len(face_samples)} face samples from {len(label_map)} persons.")
    print("Training the LBPH face recognizer...")

    # --- Train the Recognizer ---
    # Note: cv2.face was moved to cv2.face in OpenCV 3+
    # If using older OpenCV, it might be cv2.createLBPHFaceRecognizer()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face_samples, np.array(ids))

    # --- Save the Model and Labels ---
    # OpenCV's recognizer.save uses its own format (often XML/YML).
    # To strictly adhere to '.pkl', we can pickle the recognizer object itself,
    # but `recognizer.write` (or `save`) is the standard way.
    # Let's save using OpenCV's method but name it .pkl as requested.
    # IMPORTANT: recognizer.read() will be needed to load this.
    recognizer.write(MODEL_FILENAME)

    # Save the label map (name -> id) and its inverse (id -> name)
    # We need id -> name for recognition display
    id_to_label_map = {v: k for k, v in label_map.items()}
    with open(LABELS_FILENAME, 'wb') as f:
        pickle.dump(id_to_label_map, f)

    print(f"\nTraining complete!")
    print(f"Model saved to: {MODEL_FILENAME}")
    print(f"Label map saved to: {LABELS_FILENAME}")
    print(f"Labels trained: {label_map}")

if __name__ == "__main__":
    train_face_recognizer()