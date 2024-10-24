import cv2
import numpy as np
from face_detection import detect_faces
from face_recognition import load_known_faces, open_matched_image

def main():
    folder_path = './Images'  # Replace with your folder path
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

                # Open the matched image if found
                if name != "Unknown":
                    open_matched_image(folder_path, name)

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
