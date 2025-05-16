import cv2
import os

# Initialize the camera
cap = cv2.VideoCapture(0)
# Initialize the face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a directory to store face data
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Function to capture and save faces
def capture_faces(user_id):
    face_id = user_id
    count = 0
    while(True):
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces
        
        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y+h, x:x+w]  # Crop the face from the image
            cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", face)  # Save the face image
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw rectangle around the face
        cv2.imshow('Capture Face', img)
        if count >= 20:  # Take 20 face pictures per person
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected 20 images for User {user_id}")


user_id = input("Enter user ID: ")
capture_faces(user_id)
