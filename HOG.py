import cv2
import os
import csv
import numpy as np
from skimage.feature import hog
from tqdm import tqdm

# Path to the DeepFashion dataset
dataset_path = 'Deep_Fashion_Dataset - Feature/img'

# Define HOG parameters
orientations = 9  # Number of orientation bins
pixels_per_cell = (16, 16)  # Size of a cell
cells_per_block = (2, 2)  # Number of cells in each block

# CSV file to store the features
csv_file = 'Features DB/HOG_FEATURES.csv'

# Get the total number of images in the dataset for the progress bar
total_images = sum([len(files) for root, dirs, files in os.walk(dataset_path) if files])

# Open the CSV file in write mode
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    
    # Traverse the dataset folders
    with tqdm(total=total_images, desc="Extracting features") as pbar:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.jpg'):
                    # Load the image
                    image_path = os.path.join(root, file)
                    print(image_path)
                    image = cv2.imread(image_path)

                    # Resize the image
                    resized_image = cv2.resize(image, (204, 204))

                    # Convert the resized image to grayscale
                    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

                    # Compute HOG features
                    features = hog(gray_image, orientations=orientations,
                                   pixels_per_cell=pixels_per_cell,
                                   cells_per_block=cells_per_block)

                    # Replace forward slashes with backward slashes in the image path
                    image_path = image_path.replace('/', '\\')
                    print(image_path)
                    
                    # Write the features and image path to the CSV file
                    writer.writerow(list(features) + [image_path])
                    
                    # Update the progress bar
                    pbar.update(1)

# Print the completion message
print("Feature extraction completed and saved to:", csv_file)
