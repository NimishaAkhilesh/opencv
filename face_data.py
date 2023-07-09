import cv2
import os

# Create a folder to store the captured images
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

# Prompt user for the individual's name
name = input("Enter the name of the individual: ")

# Create a folder with the corresponding name
folder_path = name.replace(" ", "_")
create_folder(folder_path)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Set the initial frame count
count = 0

# Capture images until 200 images are captured
while count < 200:
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face region
        face_crop = frame[y:y+h, x:x+w]

        # Save the cropped face image
        image_name = f"{folder_path}/{count}.jpg"
        cv2.imwrite(image_name, face_crop)

        count += 1

    # Display the frame with detected faces
    cv2.imshow("Face Capture", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break