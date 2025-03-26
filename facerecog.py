import face_recognition
import cv2
import os
import glob
import numpy as np

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

def main():
    # Initialize facial recognition and load known face encodings
    sfr = SimpleFacerec()
    sfr.load_encoding_images("images/")  # Folder containing known face images

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

        face_locations, face_names = sfr.detect_known_faces(frame)
        # Draw a rectangle and label for each detected face.
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Facial Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'Esc' key.
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
