import cv2
import os
import numpy as np

def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    return faces

def load_known_faces(folder_path):
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                faces = detect_faces(image)
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face_image = image[y:y+h, x:x+w]
                    face_encoding = cv2.resize(face_image, (128, 128)).flatten()
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(os.path.splitext(filename)[0])
    return known_face_encodings, known_face_names

def main():
    folder_path = './'  # Replace with your folder path
    known_face_encodings, known_face_names = load_known_faces(folder_path)

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            frame = cv2.resize(frame, (640, 480))
            faces = detect_faces(frame)

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                face_roi_resized = cv2.resize(face_roi, (128, 128)).flatten()
                
                distances = [np.linalg.norm(face_roi_resized - known_face) for known_face in known_face_encodings]
                if distances:
                    min_distance_index = np.argmin(distances)
                    name = known_face_names[min_distance_index] if distances[min_distance_index] < 100 else "Unknown"
                else:
                    name = "Unknown"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Program terminated.")
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
