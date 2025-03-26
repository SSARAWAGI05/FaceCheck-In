import face_recognition
import cv2
import os
import glob
import numpy as np
import datetime
import shutil  # For copying files during review mode

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
    CSV format: Name,Event,Time
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

def review_unknown_faces(unknown_folder="unknown_faces", known_folder="images"):
    """
    Review unknown faces saved in the unknown_faces folder.
    For each unknown face image, an admin can:
      - Press 'a' to add it as a known face (input the name via console).
      - Press 'd' to delete the image.
      - Press any other key to skip the image.
      - Press 'q' to exit review mode.
    """
    unknown_files = glob.glob(os.path.join(unknown_folder, "*.jpg"))
    if not unknown_files:
        print("No unknown faces to review.")
        return
    print("Entering review mode for unknown faces...")
    for file in unknown_files:
        img = cv2.imread(file)
        if img is None:
            continue
        cv2.imshow("Review Unknown Face", img)
        print(f"Reviewing {file}.")
        print("Press 'a' to add as known, 'd' to delete, any other key to skip, or 'q' to quit review mode.")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('a'):
            name = input("Enter name for this face (will be saved to known faces): ").strip().upper()
            if name:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                new_filename = os.path.join(known_folder, f"{name}_{timestamp}.jpg")
                shutil.copy(file, new_filename)
                print(f"Added {name} as a known face in {new_filename}")
                os.remove(file)
            else:
                print("No name entered, skipping addition.")
        elif key == ord('d'):
            os.remove(file)
            print(f"Deleted {file}")
        elif key == ord('q'):
            print("Exiting review mode.")
            cv2.destroyWindow("Review Unknown Face")
            break
        # For any other key, the image is skipped.
    cv2.destroyWindow("Review Unknown Face")

def main():
    # Initialize facial recognition and load known face encodings.
    sfr = SimpleFacerec()
    sfr.load_encoding_images("images/")  # Folder containing known face images

    # Dictionary to track currently present students and their last seen time.
    present_students = {}  # key: student name, value: last seen datetime
    out_threshold = 10  # seconds after which a student is considered to have left

    # Variable to manage last time an unknown face was saved (rate limit: 10 seconds).
    last_unknown_save_time = datetime.datetime.min

    # Open the webcam (change the index if necessary).
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Press 'Esc' to exit facial recognition mode.")
    print("Press 'r' to enter review mode for unknown faces.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        current_time = datetime.datetime.now()

        # Detect faces and their names.
        face_locations, face_names = sfr.detect_known_faces(frame)
        current_detected_names = set()

        for (face_location, name) in zip(face_locations, face_names):
            # For known students (skip "Unknown" and "Uncertain" for attendance):
            if name not in ["Unknown", "Uncertain"]:
                current_detected_names.add(name)
                if name not in present_students:
                    present_students[name] = current_time
                    mark_attendance_event(name, "IN")
                else:
                    present_students[name] = current_time

            # Choose rectangle color:
            # Green: known; Yellow: uncertain; Red: unknown.
            if name == "Unknown":
                color = (0, 0, 255)  # Red
            elif name == "Uncertain":
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

        # Mark OUT event if a student has not been seen for longer than the threshold.
        for student in list(present_students.keys()):
            if student not in current_detected_names:
                if (current_time - present_students[student]).total_seconds() > out_threshold:
                    mark_attendance_event(student, "OUT")
                    del present_students[student]

        # Process unknown faces: save snapshots (only once every 10 seconds).
        if "Unknown" in face_names:
            if (current_time - last_unknown_save_time).total_seconds() > 10:
                for (face_location, name) in zip(face_locations, face_names):
                    if name == "Unknown":
                        save_unknown_face(frame, face_location)
                last_unknown_save_time = current_time
            # Display an alert on the frame.
            cv2.putText(frame, "Alert: Unknown Person Detected!", (50, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Facial Recognition - Attendance", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Exit on 'Esc' key.
            break
        elif key == ord('r'):
            # Enter review mode for unknown faces.
            review_unknown_faces()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
