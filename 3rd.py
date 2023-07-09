import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical



# Define the paths to the train and test folders
train_folder = 'train'
test_folder = 'test'

# Initialize lists to store the images and labels
train_images = []
train_labels = []
test_images = []
test_labels = []

# Load train images and labels
for file_name in os.listdir(train_folder):
    img_path = os.path.join(train_folder, file_name)
    label = int(file_name.split('_')[0])
    image = cv2.imread(img_path)
    train_images.append(image)
    train_labels.append(label)

# Load test images and labels
for file_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, file_name)
    label = int(file_name.split('_')[0])
    image = cv2.imread(img_path)
    test_images.append(image)
    test_labels.append(label)

# Convert the lists to NumPy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Load the FaceNet model
facenet_model = cv2.dnn.readNetFromTensorflow('facenet.pb')

# Preprocess the images for FaceNet input
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (160, 160))
    image = (image - 127.5) / 127.5  # Normalize to the range [-1, 1]
    image = np.transpose(image, (2, 0, 1))  # Reshape to (3, 160, 160)
    return image

# Perform facial feature extraction using FaceNet
def extract_features(images, model):
    preprocessed_images = np.array([preprocess_image(image) for image in images])
    model.setInput(cv2.dnn.blobFromImages(preprocessed_images, 1.0, (160, 160), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    features = model.forward()
    return features

# Extract facial features for train and test images
train_features = extract_features(train_images, facenet_model)
test_features = extract_features(test_images, facenet_model)

# Define the CNN model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(train_features.shape[1],)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(train_labels.max() + 1, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_features, to_categorical(train_labels), epochs=10, batch_size=32, validation_data=(test_features, to_categorical(test_labels)))


# Plot the accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()