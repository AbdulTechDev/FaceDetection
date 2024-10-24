import os
import cv2
import numpy as np
from face_detection import detect_faces

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

def open_matched_image(folder_path, name):
    image_path = os.path.join(folder_path, f"{name}.jpg")
    if os.path.exists(image_path):
        matched_image = cv2.imread(image_path)
        if matched_image is not None:
            # Open the image in a new window
            cv2.imshow(f'Matched Image: {name}', matched_image)
            cv2.waitKey(0)
            cv2.destroyWindow(f'Matched Image: {name}')
