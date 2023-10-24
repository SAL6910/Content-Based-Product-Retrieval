import cv2
import numpy as np
import os
import csv
from tqdm import tqdm

dataset_path = 'Deep_Fashion_Dataset - Feature'  # Set the path to the Deep Fashion dataset folder

file2 = open("Features DB/LBP_FEATURES.csv", "w", newline='')
csv_writer = csv.writer(file2)

folders = ['MEN', 'WOMEN']
total_images = sum(len(os.listdir(os.path.join(dataset_path, 'img', folder))) for folder in folders)

pbar = tqdm(total=total_images, desc='Processing Images')

# Iterate over the folders (MEN and WOMEN)
for folder in folders:
    folder_path = os.path.join(dataset_path, 'img', folder)
    categories = os.listdir(folder_path)
        
    # Iterate over the images in each item folder
    for image_name in categories:
        image_path = os.path.join(folder_path, image_name)
        img_bgr = cv2.imread(image_path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        height, width = img_gray.shape
        img_lbp = np.zeros((height, width), np.uint8)

        # Pre-processing: Thresholding
        for i in range(1, height-1):
            for j in range(1, width-1):
                if img_gray[i][j] > 130:
                    img_gray[i][j] = 0

        center_value = 0
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        f = 0
        g = 0
        h = 0

        # LBP calculation
        for i in range(1, height-1):
            for j in range(1, width-1):
                if img_gray[i-1][j-1] > img_gray[i][j]:
                    a = 1
                if img_gray[i-1][j] > img_gray[i][j]:
                    b = 128
                if img_gray[i-1][j+1] > img_gray[i][j]:
                    c = 64
                if img_gray[i][j+1] > img_gray[i][j]:
                    d = 32
                if img_gray[i+1][j+1] > img_gray[i][j]:
                    e = 16
                if img_gray[i+1][j] > img_gray[i][j]:
                    f = 8
                if img_gray[i+1][j-1] > img_gray[i][j]:
                    g = 4
                if img_gray[i][j-1] > img_gray[i][j]:
                    h = 2

                img_lbp[i][j] = a + b + c + d + e + f + g + h
        
                a = 0
                b = 0
                c = 0
                d = 0
                e = 0
                f = 0
                g = 0
                h = 0

        # Calculate the LBP histogram
        hist_lbp = cv2.calcHist([img_lbp], [0], None, [255], [0, 255])

        # Extract LBP features and image path
        features = hist_lbp[0:254, 0].flatten()

        # Combine features and image path into a single row
        row = np.concatenate((features, [image_path]))
        csv_writer.writerow(row)

        pbar.update(1)

file2.close()
pbar.close()
print("LBP Program is finished.")
