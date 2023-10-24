import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def calculate_ldn(img):
    # Function to calculate LDN feature from image
    height, width = img.shape[:2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ldn = np.zeros((height, width), dtype=np.uint8)

    def calculate_mask(img, mask, x, y):
        # Helper function to calculate the mask value at a specific pixel
        total = 0
        try:
            image = img[x-1:x+2, y-1:y+2]
            total = np.sum(image * mask)
        except:
            pass
        return total

    def ldn_value(img, x, y):
        # Calculate the LDN value at a specific pixel
        masks = [
            [[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]],
            [[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]],
            [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]],
            [[5, 5, -3], [5, 0, -3], [-3, -3, -3]],
            [[5, -3, -3], [5, 0, -3], [5, -3, -3]],
            [[-3, -3, -3], [5, 0, -3], [5, 5, -3]],
            [[-3, -3, -3], [-3, 0, -3], [5, 5, 5]],
            [[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]
        ]
        ldn_values = []
        for mask in masks:
            ldn_values.append(calculate_mask(img, mask, x, y))
        ldn_values = [max(ldn_values), min(ldn_values)]
        ldn = ldn_values[0] * 8 + ldn_values[1]
        return ldn

    for i in range(1, height-1):
        for j in range(1, width-1):
            ldn[i, j] = ldn_value(img_gray, i, j)

    return ldn

def extract_features(image_dir, output_file):
    data = []
    image_paths = []  # List to store image paths
    
    total_files = sum([len(files) for _, _, files in os.walk(image_dir)])

    with tqdm(total=total_files, desc="Extracting Features") as pbar:
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                try:
                    # Extract image path from file information
                    image_path = os.path.join(root, file).replace('/', '\\')
                    image_paths.append(image_path)  # Store image path

                    img = cv2.imread(image_path)
                    ldn = calculate_ldn(img)

                    hist_ldn = cv2.calcHist([ldn], [0], None, [255], [0, 255])
                    hist_ldn = hist_ldn.flatten()

                    row = {i: hist_ldn[i] for i in range(255)}
                    data.append(row)
                except Exception as e:
                    print(f"Error processing image: {image_path}")
                    print(str(e))

                pbar.update(1)

    df = pd.DataFrame(data)
    df["image_path"] = image_paths  # Add image_path column
    df.to_csv(output_file, index=False, header=False)  # Set header=True
    print(f"Features extracted and saved to: {output_file}")

# Specify the directory containing the 'img' folder of DeepFashion dataset
image_directory = "Deep_Fashion_Dataset - Feature/img"  
output_csv = "Features DB/LDN_Features.csv"  # Output file name
extract_features(image_directory, output_csv)
