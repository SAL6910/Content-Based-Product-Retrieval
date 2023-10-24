import cv2
import numpy as np
import os
import csv
from tqdm import tqdm

dataset_path = 'Deep_Fashion_Dataset - Feature'  # Set the path to the DeepFashion dataset folder
threshold = 5

file2 = open("Features DB/LTP_FEATURES.csv", "w", newline='')

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

        # Get the dimensions
        rows = img_gray.shape[0]
        cols = img_gray.shape[1]

        # Reordering vector - Essentially for getting binary strings
        reorder_vector = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # For the upper and lower LTP patterns
        ltp_upper = np.zeros_like(img_gray)
        ltp_lower = np.zeros_like(img_gray)

        # For each pixel in the image, ignoring the borders
        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                cen = img_gray[row, col]  # Get center

                # Get neighborhood - cast to double for better precision
                pixels = img_gray[row-1:row+2, col-1:col+2].astype(float)

                # Get ranges and determine LTP
                out_LTP = np.zeros((3, 3), dtype=int)
                low = cen - threshold
                high = cen + threshold
                out_LTP[pixels < low] = -1
                out_LTP[pixels > high] = 1
                out_LTP[(pixels >= low) & (pixels <= high)] = 0

                # Get upper and lower patterns
                upper = out_LTP.copy()
                upper[upper == -1] = 0
                upper = upper.flatten()[reorder_vector]

                lower = out_LTP.copy()
                lower[lower == 1] = 0
                lower[lower == -1] = 1
                lower = lower.flatten()[reorder_vector]

                # Convert to a binary character string, then use bin2dec to get the decimal representation
                upper_bitstring = ''.join(str(bit) for bit in upper)
                ltp_upper[row, col] = int(upper_bitstring, 2)

                lower_bitstring = ''.join(str(bit) for bit in lower)
                ltp_lower[row, col] = int(lower_bitstring, 2)

        # Compute LTP histograms
        hist_ltp_upper = cv2.calcHist([ltp_upper.astype(np.float32)], [0], None, [256], [0, 256])
        hist_ltp_lower = cv2.calcHist([ltp_lower.astype(np.float32)], [0], None, [256], [0, 256])

        # Concatenate upper and lower LTP histograms
        concatenated_histogram = np.concatenate((hist_ltp_upper, hist_ltp_lower), axis=0)
        concatenated_histogram = concatenated_histogram.flatten()

        # Write the concatenated histogram and image path to the CSV file
        writer = csv.writer(file2)
        writer.writerow(np.append(concatenated_histogram, image_path))

        pbar.update(1)

pbar.close()
file2.close()

print("LTP features saved to LTP_FEATURES.csv")