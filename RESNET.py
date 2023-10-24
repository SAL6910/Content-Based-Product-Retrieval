import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import time
import os
import csv

model = ResNet50(weights='imagenet', include_top=False)

dataset_path = 'Deep_Fashion_Dataset - Feature/img'
csvfile_path = "Features DB/RESNET_FEATURES.csv"

# Open the CSV file to write features
with open(csvfile_path, "w", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Iterate through each category folder (e.g., MEN, WOMEN)
    for category_folder in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category_folder)

        # Iterate through each image file
        for filename in os.listdir(category_path):
            image_path = os.path.join(category_path, filename)
            frame = cv2.imread(image_path)
            newsize = (224, 224)
            img = cv2.resize(frame, newsize, interpolation=cv2.INTER_AREA)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = model.predict(x)

            # Write features of the current image to a new row in the CSV file
            csv_writer.writerow([image_path] + features.flatten().tolist())

            print("Processing image:", image_path)

print("RESNET program finished")