from flask import Flask, render_template, Response, jsonify, request
import cv2
import os
from deepface import DeepFace
import sqlite3

app = Flask(__name__)
camera = None

# Load Haar cascades
face_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

# Database setup
DB_FILE = "users.db"
if not os.path.exists(DB_FILE):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def gen_frames():  
    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 7)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    if camera is not None:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return jsonify({'error': 'Camera is off'}), 400

@app.route('/start_camera')
def start_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return jsonify({'status': 'camera started'}), 200

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'status': 'camera stopped'}), 200

@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    global camera
    if camera is not None and camera.isOpened():
        success, frame = camera.read()
        if success:
            # Save the captured frame
            name = request.form.get('name')
            if not name:
                return jsonify({'error': 'Name is required'}), 400

            file_path = f'static/{name}.jpg'
            cv2.imwrite(file_path, frame)

            # Save to database
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, image_path) VALUES (?, ?)", (name, file_path))
            conn.commit()
            conn.close()

            return jsonify({'status': 'photo captured', 'file_path': file_path}), 200
    return jsonify({'error': 'Failed to capture photo'}), 400

@app.route('/analyze_photo', methods=['POST'])
def analyze_photo():
    image_path = request.form.get('image_path')
    if not image_path:
        return jsonify({'error': 'Image path is required'}), 400

    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'emotion'])
        return jsonify({'analysis': analysis}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/match_live_face', methods=['POST'])
def match_live_face():
    global camera
    if camera is not None and camera.isOpened():
        success, frame = camera.read()
        if success:
            # Save the temporary live frame
            live_file_path = 'static/live_temp.jpg'
            cv2.imwrite(live_file_path, frame)

            # Retrieve all user images from the database
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT name, image_path FROM users")
            users = cursor.fetchall()
            conn.close()

            # Compare the live photo with stored user photos
            for user in users:
                name, image_path = user
                try:
                    result = DeepFace.verify(img1_path=live_file_path, img2_path=image_path)
                    if result['verified']:
                        return jsonify({'status': 'match', 'name': name}), 200
                except Exception as e:
                    return jsonify({'error': str(e)}), 500

            return jsonify({'status': 'no match'}), 200

    return jsonify({'error': 'Camera is not active or failed to capture live photo'}), 400


if __name__ == '__main__':
    app.run(debug=True)
