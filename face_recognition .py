import cv2

# Load the pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

# Recognize face from the webcam
def recognize_face():
    cap = cv2.VideoCapture(0)  # Initialize webcam
    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]  # Crop the detected face from the image
            id, confidence = recognizer.predict(face)  # Predict the ID of the face
            
            # If the confidence is less than 100, it's a good match
            if confidence < 100:
                cv2.putText(img, f"User {id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw a rectangle around the face
        
        cv2.imshow('Face Recognition', img)  # Show the video feed with face recognition
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
            break
    
    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

recognize_face()
