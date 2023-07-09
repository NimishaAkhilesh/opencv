import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('face_recognition_model.h5')

def recognize_faces(image):
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    recognized_faces = []
    for (x, y, w, h) in faces:
        # Extract the face region from the image
        face = gray[y:y + h, x:x + w]

        # Preprocess the face for the model input
        face = cv2.resize(face, (160, 160))
        face = (face - 127.5) / 127.5
        face = np.expand_dims(face, axis=0)

        # Perform face recognition using the loaded model
        embeddings = model.predict(face)
        predicted_label = np.argmax(embeddings)

        recognized_faces.append((x, y, w, h, predicted_label))

    return recognized_faces

# Load the image
image_path = 'image.jpg'
image = cv2.imread(image_path)

# Perform face recognition
recognized_faces = recognize_faces(image)

# Display the recognized faces
for (x, y, w, h, label) in recognized_faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f'Label: {label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Show the image with recognized faces
cv2.imshow('Face Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Open the video capture
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# Process frames from the video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Perform face recognition
    recognized_faces = recognize_faces(frame)

    # Display the recognized faces
    for (x, y, w, h, label) in recognized_faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Label: {label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame with recognized faces
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()