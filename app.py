import os
import glob
import cv2
import face_recognition
import numpy as np
import datetime
import csv
from flask import Flask, render_template, Response, jsonify, send_from_directory, request

app = Flask(__name__)

# -------------------------------
# Facial Recognition Class & Helpers
# -------------------------------
class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25
        self.tolerance = 0.45
        self.margin_threshold = 0.03

    def load_encoding_images(self, images_path):
        images_files = glob.glob(os.path.join(images_path, "*.*"))
        print(f"{len(images_files)} encoding images found.")
        for img_path in images_files:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to read {img_path}")
                continue
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            basename = os.path.basename(img_path)
            (filename, _) = os.path.splitext(basename)
            encodings = face_recognition.face_encodings(rgb_img)
            if encodings:
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(filename.upper())
            else:
                print(f"Warning: No face found in {img_path}")
        print("Encoding images loaded.")

    def detect_known_faces(self, frame):
        # Resize frame for faster processing.
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if face_distances.size > 0 else None
            if best_match_index is not None and face_distances[best_match_index] < self.tolerance:
                sorted_indices = np.argsort(face_distances)
                if len(sorted_indices) > 1:
                    second_best = face_distances[sorted_indices[1]]
                    if (second_best - face_distances[best_match_index]) < self.margin_threshold:
                        name = "Uncertain"
                    else:
                        name = self.known_face_names[best_match_index]
                else:
                    name = self.known_face_names[best_match_index]
            else:
                name = "Unknown"
            face_names.append(name)
        # Scale face locations back to the original frame size.
        face_locations = np.array(face_locations) / self.frame_resizing
        return face_locations.astype(int), face_names

def mark_attendance_event(name, event, attendance_file="attendance.csv"):
    """
    Append an attendance event (IN or OUT) into a CSV file.
    CSV format: Name,Event,Timestamp
    """
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(attendance_file, "a") as f:
        f.write(f"{name},{event},{now}\n")
    print(f"{event} marked for {name} at {now}")

def save_unknown_face(frame, face_location, folder="unknown_faces"):
    """
    Crop the unknown face from the frame and save it to a folder.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    top, right, bottom, left = face_location
    face_img = frame[top:bottom, left:right]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"unknown_{timestamp}.jpg")
    cv2.imwrite(filename, face_img)
    print(f"Saved unknown face to {filename}")

# -------------------------------
# Global Variables and Initialization
# -------------------------------
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")  # Folder containing known face images

present_students = {}  # key: student name, value: last seen datetime
out_threshold = 10  # seconds of absence to mark OUT
last_unknown_save_time = datetime.datetime.min

# Initialize video capture (using default webcam)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# -------------------------------
# Video Streaming Generator
# -------------------------------
def gen_frames():
    global present_students, last_unknown_save_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = datetime.datetime.now()
        face_locations, face_names = sfr.detect_known_faces(frame)
        current_detected_names = set()
        for (face_location, name) in zip(face_locations, face_names):
            if name not in ["Unknown", "Uncertain"]:
                current_detected_names.add(name)
                if name not in present_students:
                    present_students[name] = current_time
                    mark_attendance_event(name, "IN")
                else:
                    present_students[name] = current_time
            color = (0, 0, 255) if name == "Unknown" else ((0, 255, 255) if name == "Uncertain" else (0, 255, 0))
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        for student in list(present_students.keys()):
            if student not in current_detected_names:
                if (current_time - present_students[student]).total_seconds() > out_threshold:
                    mark_attendance_event(student, "OUT")
                    del present_students[student]
        if "Unknown" in face_names:
            if (current_time - last_unknown_save_time).total_seconds() > 10:
                for (face_location, name) in zip(face_locations, face_names):
                    if name == "Unknown":
                        save_unknown_face(frame, face_location)
                last_unknown_save_time = current_time
            cv2.putText(frame, "Alert: Unknown Person Detected!", (50, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def read_attendance_log(attendance_file="attendance.csv"):
    if not os.path.exists(attendance_file):
        return []
    with open(attendance_file, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

# -------------------------------
# API and Utility Endpoints
# -------------------------------
@app.route('/')
def index():
    # Always serve the single-page HTML template.
    attendance = read_attendance_log()
    return render_template('index.html', attendance=attendance)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/attendance')
def api_attendance():
    data = []
    if os.path.exists("attendance.csv"):
        with open("attendance.csv", newline='') as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=['name', 'event', 'time'])
            for row in reader:
                data.append(row)
    return jsonify(data)

@app.route('/download')
def download():
    if os.path.exists("attendance.csv"):
        return send_from_directory(directory=os.getcwd(), path='attendance.csv', as_attachment=True)
    return "Attendance file not found.", 404

@app.route('/clear', methods=['POST'])
def clear_attendance():
    open("attendance.csv", "w").close()
    return jsonify({"status": "Attendance log cleared."})

# New API endpoint for filtered reports
@app.route('/api/reports')
def api_reports():
    data = []
    if os.path.exists("attendance.csv"):
        with open("attendance.csv", newline='') as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=['name', 'event', 'time'])
            for row in reader:
                data.append(row)
    filter_name = request.args.get('name', '').upper()
    filter_date = request.args.get('date', '')
    filtered = []
    from datetime import datetime
    for row in data:
        include = True
        if filter_name and filter_name not in row['name']:
            include = False
        if filter_date:
            try:
                dt = datetime.strptime(row['time'], '%Y-%m-%d %H:%M:%S')
                if dt.strftime('%Y-%m-%d') != filter_date:
                    include = False
            except Exception:
                pass
        if include:
            filtered.append(row)
    return jsonify(filtered)

# New API endpoint for unknown faces
@app.route('/api/unknown_faces')
def api_unknown_faces():
    folder = "unknown_faces"
    images = []
    if os.path.exists(folder):
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return jsonify(images)

# Modified manual entry endpoint to handle AJAX (returns JSON)
@app.route('/manual_entry', methods=['POST'])
def manual_entry():
    name = request.form.get('name', '').strip().upper()
    event = request.form.get('event', '').strip().upper()
    if name and event in ['IN', 'OUT']:
        mark_attendance_event(name, event)
        return jsonify({"message": f"Manual entry recorded for {name} as {event}."})
    else:
        return jsonify({"message": "Please provide a valid name and event (IN/OUT)."}), 400

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5001, debug=True)
    finally:
        cap.release()
