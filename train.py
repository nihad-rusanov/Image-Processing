import cv2
import os
import numpy as np

# Prepare the data
def prepare_data():
    faces = []
    ids = []
    # Loop through the dataset folder to collect images
    for filename in os.listdir('dataset'):
        if filename.endswith('.jpg'):
            img = cv2.imread(f"dataset/{filename}", cv2.IMREAD_GRAYSCALE)
            id = int(filename.split('.')[1])  # User ID is the second element in the filename
            faces.append(img)
            ids.append(id)
    return faces, ids

# Train the face recognizer
def train_recognizer():
    faces, ids = prepare_data()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.save('trainer.yml')  # Save the trained model

train_recognizer()
