import face_recognition
import cv2
import os
import glob
import numpy as np
import datetime

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
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if face_distances.size > 0 else None
            if best_match_index is not None and face_distances[best_match_index] < self.tolerance:
                # Check if the best match is sufficiently better than the second best
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
    Append an attendance event (IN or OUT) for a student into a CSV file with a timestamp.
    The CSV file format is: Name,Event,Time
    """
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(attendance_file, "a") as f:
        f.write(f"{name},{event},{now}\n")
    print(f"{event} marked for {name} at {now}")

def save_unknown_face(frame, face_location, folder="unknown_faces"):
    """
    Crop the unknown face from the frame and save it to a folder with a timestamped filename.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    top, right, bottom, left = face_location
    face_img = frame[top:bottom, left:right]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"unknown_{timestamp}.jpg")
    cv2.imwrite(filename, face_img)
    print(f"Saved unknown face to {filename}")

def main():
    # Initialize facial recognition and load known face encodings.
    sfr = SimpleFacerec()
    sfr.load_encoding_images("images/")  # Folder containing known face images

    # Dictionary to keep track of currently present students and their last seen time.
    present_students = {}  # key: student name, value: last seen datetime
    out_threshold = 10  # seconds of absence after which a student is considered to have left

    # Variable to manage last time an unknown face was saved (rate limit: 10 seconds).
    last_unknown_save_time = datetime.datetime.min

    # Open the default webcam (index 0). Change the index if necessary.
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Press 'Esc' to exit facial recognition mode.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        current_time = datetime.datetime.now()

        # Detect faces and their names in the current frame.
        face_locations, face_names = sfr.detect_known_faces(frame)

        # Set to keep track of the names detected in the current frame.
        current_detected_names = set()

        # Process each detected face.
        for (face_location, name) in zip(face_locations, face_names):
            # For known students (skip "Unknown" and "Uncertain" for attendance purposes):
            if name not in ["Unknown", "Uncertain"]:
                current_detected_names.add(name)
                # If the student appears for the first time or re-enters after an absence, mark IN.
                if name not in present_students:
                    present_students[name] = current_time
                    mark_attendance_event(name, "IN")
                else:
                    # Update the student's last seen time.
                    present_students[name] = current_time

            # Choose rectangle color based on recognition result:
            # Green: known; Yellow: uncertain; Red: unknown.
            if name == "Unknown":
                color = (0, 0, 255)  # Red
            elif name == "Uncertain":
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green

            # Draw rectangle and label on the frame.
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

        # Check for students who were previously present but are not detected in the current frame.
        # If they haven't been seen for more than the threshold, mark an OUT event.
        for student in list(present_students.keys()):
            if student not in current_detected_names:
                if (current_time - present_students[student]).total_seconds() > out_threshold:
                    mark_attendance_event(student, "OUT")
                    del present_students[student]

        # Check for unknown faces and save snapshots (only once every 10 seconds).
        if "Unknown" in face_names:
            if (current_time - last_unknown_save_time).total_seconds() > 10:
                for (face_location, name) in zip(face_locations, face_names):
                    if name == "Unknown":
                        save_unknown_face(frame, face_location)
                last_unknown_save_time = current_time
            # Display an alert on the frame for unknown persons.
            cv2.putText(frame, "Alert: Unknown Person Detected!", (50, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Facial Recognition - Attendance", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'Esc' key.
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
