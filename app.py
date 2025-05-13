# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, Response, session
import os
import cv2
import time # For timestamp to bust cache for processed image
from werkzeug.utils import secure_filename
import utils # Your refactored utility functions

app = Flask(__name__)
app.secret_key = "your_very_secret_key_here" # Important for flash messages and session
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RECOGNIZED_IMAGES_FOLDER'] = 'static/recognized_images/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['NUM_IMAGES_TO_CAPTURE_WEB'] = utils.NUM_IMAGES_TO_CAPTURE # Make it accessible in templates

# Ensure upload and recognized images directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RECOGNIZED_IMAGES_FOLDER'], exist_ok=True)

# --- Helper Function ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Routes ---
@app.route('/')
def index():
    utils.load_resources() # Try to load resources on home page visit
    return render_template('index.html')

@app.route('/capture', methods=['GET'])
def capture_page():
    datasets = [d for d in os.listdir(utils.DATASET_DIR) if os.path.isdir(os.path.join(utils.DATASET_DIR, d))]
    return render_template('capture.html', datasets=datasets)

@app.route('/capture/start', methods=['POST'])
def start_capture():
    name = request.form.get('name')
    if not name:
        flash('Name cannot be empty.', 'danger')
        return redirect(url_for('capture_page'))

    # Sanitize name to be used as a directory name
    safe_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in name)
    if not safe_name:
        flash('Invalid characters in name. Please use alphanumeric, underscore, or hyphen.', 'danger')
        return redirect(url_for('capture_page'))

    flash(f"Attempting to capture images for: {safe_name}. Please look at the webcam.", 'info')
    
    # This will block, consider threading for long operations in real production apps
    # For this project, we'll keep it simple.
    status_message = utils.capture_face_images_util(safe_name)

    if "Successfully captured" in status_message:
        flash(status_message, 'success')
    else:
        flash(status_message, 'warning') # Could be warning or danger based on severity
    return redirect(url_for('capture_page'))


@app.route('/train', methods=['GET'])
def train_page():
    # Check if model and labels exist to give some status
    model_exists = os.path.exists(utils.MODEL_FILENAME)
    labels_exist = os.path.exists(utils.LABELS_FILENAME)
    status = ""
    if model_exists and labels_exist:
        status = "Model and labels found. You can retrain if needed."
    elif not os.listdir(utils.DATASET_DIR):
        status = "Dataset is empty. Please capture images first."
    else:
        status = "Model not trained yet. Click 'Start Training'."
    flash(status, 'info')
    return render_template('train_status.html', status_message=None)


@app.route('/train/start', methods=['POST'])
def start_train():
    flash("Training started... This might take a moment.", 'info')
    # This will block.
    status_message = utils.train_model_util()
    if "Error" in status_message:
        flash(status_message, 'danger')
    else:
        flash("Training process finished.", 'success')
    # Re-render the train_status page with the log
    return render_template('train_status.html', status_message=status_message)


@app.route('/recognize_image', methods=['GET'])
def recognize_image_page():
    if not utils.recognizer or not utils.id_to_label_map:
        flash("Model not trained or loaded. Please train the model first.", "warning")
    return render_template('recognize_image.html')

@app.route('/recognize_image/upload', methods=['POST'])
def upload_and_recognize_image():
    if 'imagefile' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('recognize_image_page'))
    file = request.files['imagefile']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('recognize_image_page'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_image_path)

        processed_image_rel_path, detected_names = utils.recognize_faces_in_image_util(temp_image_path)

        if processed_image_rel_path:
            # Pass a timestamp to prevent browser caching of the image
            processed_image_url = url_for('static', filename=processed_image_rel_path.replace('static/', ''), _external=False) + f"?t={time.time()}"
            flash(f"Recognition complete. Detected: {', '.join(detected_names) if detected_names else 'None'}", 'success')
            return render_template('recognize_image.html', processed_image_url=processed_image_url, detected_names=detected_names, timestamp=time.time())
        else:
            flash(f"Error during recognition: {', '.join(detected_names) if detected_names else 'Unknown error'}", 'danger')
            return redirect(url_for('recognize_image_page'))
    else:
        flash('Invalid file type. Allowed types: png, jpg, jpeg', 'danger')
        return redirect(url_for('recognize_image_page'))


# --- Real-time Recognition Video Feed ---
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam for real-time feed.")
        # Yield a placeholder or error image if you want
        # For now, it will just not stream if webcam fails
        return

    # Ensure resources are loaded for the recognition part
    if utils.face_cascade is None or utils.recognizer is None or utils.id_to_label_map is None:
        print("Error: Model/resources not loaded for real-time feed.")
        # Optionally, try to load them again here
        # utils.load_resources()
        # if utils.face_cascade is None or utils.recognizer is None or utils.id_to_label_map is None:
        #     cap.release()
        #     return # Still not loaded, can't proceed

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from webcam.")
            break
        else:
            # Process frame for recognition
            frame = utils.recognize_faces_on_frame(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame.")
                continue # Skip this frame
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()
    print("Webcam released for real-time feed.")


@app.route('/realtime')
def realtime_page():
    if not utils.recognizer or not utils.id_to_label_map:
        flash("Model not trained or loaded. Please train the model first for recognition.", "warning")
        # You might redirect or just show a message on the realtime_feed page
    return render_template('realtime_feed.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True) # debug=True is for development only