import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import time
import os
import csv

csvfile = open("Features DB/VGGNET_FEATURES.csv", "w")
model = VGG16(weights='imagenet')

Final_VGGFeature = np.zeros((1, 1000), np.uint16)

start_time = time.time()

dataset_path = 'Deep_Fashion_Dataset - Feature/img'

# Function to extract features from images using VGG16
def extract_features(img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    return features

# Iterate through each category folder (e.g., MEN, WOMEN)
for category_folder in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category_folder)
    
    # Iterate through each image file
    for filename in os.listdir(category_path):
        image_path = os.path.join(category_path, filename)
        frame = cv2.imread(image_path)
        
        # Extract features from the image
        features = extract_features(frame)
        Final_VGGFeature = np.add(Final_VGGFeature, features)

        print("Processing image:", image_path)
        print("--- %s seconds ---" % (time.time() - start_time))

        temp = 0
        for featuredata in Final_VGGFeature[0]:
            if temp < 1000:
                csvfile.write(str(featuredata) + ",")
                temp = temp + 1
        
        image_path = image_path.replace('/', '\\')
        csvfile.write(image_path + "\n")

        temp = 0

csvfile.flush()
csvfile.close()

print("VGGNET program finished")