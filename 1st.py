import cv2
import os

# Create a folder for storing the captured images
def create_folder(images):
    if not os.path.exists(images):
        os.makedirs(images)

# Capture facial images and save them
def capture_images(face1,num_images):
    # Create folder with the given name
    create_folder(face1.jpg)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Load the face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0
    while count < num_images:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the captured frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the face region
            face_crop = frame[y:y + h, x:x + w]

            # Save the cropped face image
            img_name = f'{name}/image_{count}.jpg'
            cv2.imwrite(img_name, face_crop)

            count += 1

        # Display the resulting frame
        cv2.imshow('Capture Faces', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

# Prompt the user to enter the name of the individual
name = input("Enter the name of the individual: ")

# Capture 200 images
capture_images(name, 200)