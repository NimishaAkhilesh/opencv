import os
import random
import shutil

# Get the current directory
current_directory = os.getcwd()

# Create the parent folder with the same name as the original folder
parent_folder = os.path.basename(current_directory)
os.makedirs(parent_folder, exist_ok=True)

# Create the "train" and "test" folders within the parent folder
train_folder = os.path.join(parent_folder, "train")
test_folder = os.path.join(parent_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get the list of image files in the current directory
image_files = [file for file in os.listdir(current_directory) if file.endswith(".jpg") or file.endswith(".png")]

# Shuffle the image files randomly
random.shuffle(image_files)

# Calculate the number of images for the train and test sets (50% each)
total_images = len(image_files)
train_size = total_images // 2
test_size = total_images - train_size

# Move the images to the train folder
for image_file in image_files[:train_size]:
    source_path = os.path.join(current_directory, image_file)
    target_path = os.path.join(train_folder, image_file)
    shutil.move(source_path, target_path)

# Move the images to the test folder
for image_file in image_files[train_size:]:
    source_path = os.path.join(current_directory, image_file)
    target_path = os.path.join(test_folder, image_file)
    shutil.move(source_path, target_path)

print("Images distributed successfully!")