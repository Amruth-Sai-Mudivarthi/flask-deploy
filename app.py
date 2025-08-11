import cv2
from pyzbar.pyzbar import decode
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response, flash
import numpy as np
import base64
import os
from PIL import Image
import time
import threading
import csv
from functools import wraps
from math import radians, cos, sin, asin, sqrt

# --- 1. App Configuration and Setup ---
app = Flask(__name__)
app.secret_key = 'your-super-secret-key-for-flask'

OFFICE_COORDINATES = (12.9124934, 77.5045989)
MAX_DISTANCE_METERS = 1000

# --- Define paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAINER_DIR = os.path.join(BASE_DIR, 'trainer')
USER_DB_FILE = os.path.join(BASE_DIR, 'users.csv')
CASCADE_PATH = os.path.join(DATA_DIR, 'haarcascade_frontalface_default.xml')
EYE_CASCADE_PATH = os.path.join(DATA_DIR, 'haarcascade_eye.xml')
TRAINER_FILE = os.path.join(TRAINER_DIR, 'trainer.yml')
PROTOTXT_PATH = os.path.join(DATA_DIR, 'MobileNetSSD_deploy.prototxt.txt')
MODEL_PATH = os.path.join(DATA_DIR, 'MobileNetSSD_deploy.caffemodel')

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- 2. Load Models and Data at Startup ---
face_detector = cv2.CascadeClassifier(CASCADE_PATH)
eye_detector = cv2.CascadeClassifier(EYE_CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

if face_detector.empty():
    print(f"!!! FATAL ERROR: Could not load face detector from {CASCADE_PATH}")
if eye_detector.empty():
    print(f"!!! FATAL ERROR: Could not load eye detector from {EYE_CASCADE_PATH}")

try:
    person_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    print("Successfully loaded Person Detection model for tailgating.")
except cv2.error as e:
    person_net = None
    print(f"!!! WARNING: Could not load Person Detection model. Tailgating feature will be disabled. Error: {e}")

# In-memory data stores
face_id_to_user_map = {}
authorized_users_for_barcode = {}
liveness_auth_data = { 'status': 'pending', 'user': None, 'message': None, 'lock': threading.Lock() }
tailgating_event_data = { 'detected': False, 'lock': threading.Lock() }

def load_user_data():
    global authorized_users_for_barcode, face_id_to_user_map
    users_for_barcode = {}
    face_id_to_user_map.clear()
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'name', 'face_id'])
        return
    try:
        with open(USER_DB_FILE, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                users_for_barcode[row['id']] = row['name']
                face_id_to_user_map[int(row['face_id'])] = {'user_id': row['id'], 'name': row['name']}
        print(f"Successfully loaded {len(users_for_barcode)} users.")
        authorized_users_for_barcode = users_for_barcode
    except Exception as e:
        print(f"FATAL ERROR: Could not read user database '{USER_DB_FILE}'. Error: {e}")

load_user_data()

def find_barcode_in_image(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        barcodes = decode(gray)
        return barcodes[0].data.decode("utf-8") if barcodes else None
    except Exception: return None

def get_images_and_labels_for_training(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples, ids = [], []
    for image_path in image_paths:
        try:
            pil_image = Image.open(image_path).convert('L')
            img_numpy = np.array(pil_image, 'uint8')
            face_id = int(os.path.split(image_path)[-1].split(".")[1])
            faces = face_detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(face_id)
        except Exception as e:
            print(f"Warning: Skipping image {image_path}: {e}")
    return face_samples, ids


def perform_face_recognition(image_gray, face_roi):
    if not os.path.exists(TRAINER_FILE) or os.path.getsize(TRAINER_FILE) == 0:
        return None, "Face model file not found or is empty. Please train the model."
    try:
        recognizer.read(TRAINER_FILE)
        (x, y, w, h) = face_roi
        predicted_face_id, confidence = recognizer.predict(image_gray[y:y+h, x:x+w])
        threshold = 75.0
        if confidence < threshold:
            user_details = face_id_to_user_map.get(predicted_face_id)
            if user_details:
                return user_details, "Recognition successful."
        return None, "User not recognized."
    except cv2.error as e:
        return None, "Model data is invalid. Please re-train the model."
    except Exception as e:
        return None, "An unexpected error occurred."

def detect_persons_in_frame(frame):
    if person_net is None: return frame, 0
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    person_net.setInput(blob)
    detections = person_net.forward()
    person_count = 0
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person":
                person_count += 1
    return frame, person_count

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


# =====================================================================
# ================== AUTHENTICATION DECORATORS ========================
# =====================================================================

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('is_admin'):
            flash('You must be an admin to access this page.', 'warning')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_name' not in session:
            flash("You must be logged in to view this page.", "warning")
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# =====================================================================
# ======================== ADMIN ROUTES ===============================
# =====================================================================

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin@1234':
            session['is_admin'] = True
            flash('Admin login successful!', 'success')
            return redirect(url_for('add_user'))
        else:
            flash('Invalid credentials. Please try again.', 'error')
            return redirect(url_for('admin_login'))
    return render_template('admin_login.html')

@app.route('/admin_logout')
def admin_logout():
    session.pop('is_admin', None)
    flash('You have been logged out from the admin panel.', 'info')
    return redirect(url_for('admin_login'))


# =====================================================================
# ============= MAIN APPLICATION ROUTES (Public and User) =============
# =====================================================================

@app.route('/')
def index():
    if 'user_id' in session: return redirect(url_for('dashboard')) # MODIFIED
    return render_template('index.html')

@app.route('/face_auth')
def face_auth_page():
    if 'user_id' in session: return redirect(url_for('dashboard')) # MODIFIED
    if face_detector.empty() or eye_detector.empty():
        flash("Liveness detection models not loaded. Please check server logs.", "error")
        return redirect(url_for('index'))
    with liveness_auth_data['lock']:
        liveness_auth_data['status'] = 'pending'
        liveness_auth_data['user'] = None
    return render_template('face_auth.html')

# ================= NEW DASHBOARD ROUTE (REPLACES MAP) ================
@app.route('/dashboard')
@login_required
def dashboard():
    # This is the main page for a logged-in user.
    return render_template('dashboard.html', user_name=session['user_name'])
# =====================================================================

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_name', None)
    flash("You have been successfully logged out.", "success")
    return redirect(url_for('index'))

@app.route('/entry_monitoring')
@login_required
def entry_monitoring():
    with tailgating_event_data['lock']:
        tailgating_event_data['detected'] = False
    return render_template('entry_monitoring.html', user_name=session['user_name'])

@app.route('/tailgating_feed')
def tailgating_feed():
    def generate():
        cam = cv2.VideoCapture(0)
        start_time = time.time()
        monitoring_duration = 8
        while time.time() - start_time < monitoring_duration:
            success, frame = cam.read()
            if not success: break
            _, person_count = detect_persons_in_frame(frame.copy())
            if person_count > 1:
                with tailgating_event_data['lock']:
                    tailgating_event_data['detected'] = True
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cam.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_tailgating_status')
def check_tailgating_status():
    with tailgating_event_data['lock']:
        detected = tailgating_event_data['detected']
    return jsonify({'tailgating_detected': detected})

@app.route('/liveness_feed')
def liveness_feed():
    def generate():
        cam = cv2.VideoCapture(0)
        EYE_CLOSED_FRAMES, REQUIRED_BLINKS, closed_counter, total_blinks = 3, 2, 0, 0
        eyes_were_open = True
        start_time, timeout = time.time(), 20
        while time.time() - start_time < timeout:
            with liveness_auth_data['lock']:
                if liveness_auth_data['status'] != 'pending': break
            success, frame = cam.read()
            if not success: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_detector.detectMultiScale(face_roi_gray)
                if len(eyes) == 0:
                    closed_counter += 1
                    eyes_were_open = False
                else:
                    if not eyes_were_open and closed_counter >= EYE_CLOSED_FRAMES:
                        total_blinks += 1
                    eyes_were_open = True
                    closed_counter = 0
                cv2.putText(frame, f"Blinks: {total_blinks}/{REQUIRED_BLINKS}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if total_blinks >= REQUIRED_BLINKS:
                    user, message = perform_face_recognition(gray, (x, y, w, h))
                    with liveness_auth_data['lock']:
                        if user:
                            liveness_auth_data['status'] = 'success'
                            liveness_auth_data['user'] = user
                        else:
                            liveness_auth_data['status'] = 'fail'
                            liveness_auth_data['message'] = message
                    break
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cam.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_liveness_status')
def check_liveness_status():
    with liveness_auth_data['lock']:
        status = liveness_auth_data['status']
        user = liveness_auth_data.get('user')
        message = liveness_auth_data.get('message', 'An unknown error occurred.')

    if status == 'success' and user:
        session['user_id'] = user['user_id']
        session['user_name'] = user['name']
        return jsonify({'status': 'success', 'redirect_url': url_for('entry_monitoring')})
    elif status == 'fail':
        return jsonify({'status': 'fail', 'message': f'Liveness confirmed, but auth failed. Reason: {message}'})
    else:
        return jsonify({'status': 'pending'})

@app.route('/scan_barcode', methods=['POST'])
def scan_barcode():
    data = request.get_json()
    if not data or 'image' not in data: return jsonify({'status': 'error', 'message': 'No image data.'}), 400
    try:
        _, encoded = data['image'].split(",", 1)
        img = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    except Exception: return jsonify({'status': 'error', 'message': 'Invalid image data.'}), 400
    if img is None: return jsonify({'status': 'error', 'message': 'Could not decode image.'}), 400

    scanned_id = find_barcode_in_image(img)
    if scanned_id and scanned_id in authorized_users_for_barcode:
        session['pending_location_check_id'] = scanned_id
        return jsonify({'status': 'success', 'redirect_url': url_for('location_check')})
    else:
        message = "User ID not recognized." if scanned_id else "No barcode found."
        return jsonify({'status': 'not_found', 'message': message})

@app.route('/location_check')
def location_check():
    if 'pending_location_check_id' not in session:
        flash("Please scan your barcode first.", "warning")
        return redirect(url_for('index'))
    return render_template('location_check.html')

@app.route('/verify_location_and_login', methods=['POST'])
def verify_location_and_login():
    if 'pending_location_check_id' not in session:
        return jsonify({'status': 'error', 'message': 'No pending user session.'}), 403

    user_id = session.pop('pending_location_check_id')
    data = request.get_json()
    lat, lon = data.get('latitude'), data.get('longitude')
    office_lat, office_lon = OFFICE_COORDINATES
    distance_km = haversine(lon, lat, office_lon, office_lat)
    distance_meters = distance_km * 1000

    if distance_meters <= MAX_DISTANCE_METERS:
        user_name = authorized_users_for_barcode[user_id]
        session['user_id'] = user_id
        session['user_name'] = user_name
        return jsonify({
            'status': 'success',
            'message': f"On-site location confirmed ({int(distance_meters)}m). Access granted.",
            'redirect_url': url_for('dashboard') # MODIFIED
        })
    else:
        return jsonify({
            'status': 'fail',
            'message': f"Access Denied: You are {distance_km:.2f}km away from the office."
        })
    
@app.route('/post_auth_location_check')
@login_required
def post_auth_location_check():
    """Renders the page that will perform the final location check."""
    return render_template('post_auth_location_check.html')

# ✅ STEP 2: NEW ROUTE TO VERIFY THE LOCATION AND GRANT/DENY DASHBOARD ACCESS
@app.route('/verify_final_location', methods=['POST'])
@login_required
def verify_final_location():
    """Receives coords, checks distance, and decides final access."""
    data = request.get_json()
    if not data or 'latitude' not in data or 'longitude' not in data:
        return jsonify({'status': 'error', 'message': 'Invalid location data.'}), 400

    user_lat, user_lon = data.get('latitude'), data.get('longitude')
    office_lat, office_lon = OFFICE_COORDINATES

    distance_meters = haversine(user_lon, user_lat, office_lon, office_lat)

    if distance_meters <= MAX_DISTANCE_METERS:
        # User is near, grant access to dashboard
        flash(f"Location confirmed ({int(distance_meters)}m). Welcome!", "success")
        return jsonify({
            'status': 'success',
            'redirect_url': url_for('dashboard')
        })
    else:
        # User is too far, log them out and deny access
        user_name = session.get('user_name', 'User')
        session.clear() # Log the user out completely
        flash(f"Access for {user_name} denied. You are {distance_meters/1000:.2f}km away from the office.", "error")
        return jsonify({
            'status': 'fail',
            'message': 'You are too far from the office to access the dashboard.',
            'redirect_url': url_for('index') # Send them back to the login page
        })

# =====================================================================
# =================== SECURED ADMIN ROUTES ============================
# =====================================================================
@app.route('/add_user', methods=['POST', 'GET'])
@admin_required
def add_user():
    if request.method == 'POST':
        long_user_id = request.form.get('user_id')
        user_name = request.form.get('user_name', '').strip()
        if not long_user_id or not user_name:
            flash("User ID and Name are required.", "error")
            return redirect(url_for('add_user'))
        if '.' in user_name:
            flash("Usernames cannot contain periods.", "error")
            return redirect(url_for('add_user'))
        if long_user_id in authorized_users_for_barcode:
            flash(f"User ID {long_user_id} already exists.", "error")
            return redirect(url_for('add_user'))
        new_face_id = max(face_id_to_user_map.keys()) + 1 if face_id_to_user_map else 1
        with open(USER_DB_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([long_user_id, user_name, new_face_id])
        load_user_data()
        flash(f"User {user_name} added. Now capture face.", "success")
        return redirect(url_for('capture_face', face_id=new_face_id, user_name=user_name))
    return render_template('add_user.html')

@app.route('/capture/<int:face_id>/<string:user_name>')
@admin_required
def capture_face(face_id, user_name):
    return render_template('capture.html', face_id=face_id, user_name=user_name)

@app.route('/capture_feed/<int:face_id>/<string:user_name>')
@admin_required
def capture_feed(face_id, user_name):
    def generate():
        cam = cv2.VideoCapture(0)
        count, max_images = 0, 50
        while count < max_images:
            success, frame = cam.read()
            if not success: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                count += 1
                file_path = os.path.join(DATASET_DIR, f"User.{face_id}.{user_name}.{count}.jpg")
                cv2.imwrite(file_path, gray[y:y+h, x:x+w])
                progress_text = f"Captured: {count}/{max_images}"
                cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                time.sleep(0.1)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cam.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train')
@admin_required
def train_page():
    return render_template('train.html')

@app.route('/train_model')
@admin_required
def train_model():
    flash("Training started. This might take a moment...", "info")
    faces, ids = get_images_and_labels_for_training(DATASET_DIR)
    if not faces:
        flash("No faces found in dataset to train.", "error")
        return redirect(url_for('train_page'))
    recognizer.train(faces, np.array(ids))
    recognizer.write(TRAINER_FILE)
    num_users = len(np.unique(ids))
    flash(f"Model trained successfully on {num_users} user(s).", "success")
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)