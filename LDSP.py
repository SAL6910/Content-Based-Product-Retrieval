import cv2
import numpy as np
import csv
import os

def compute_ldsp_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = img_gray.shape

    # Threshold the image
    img_gray[img_gray > 130] = 0

    ldsp = np.zeros((height, width), np.uint8)

    def calculateMask(img, mask, x, y):
        total = np.sum(img[x - 1: x + 2, y - 1: y + 2] * mask)
        return total

    def calculateLDSP(p1, p2):
        diff = p1 - p2
        rotation = 1 if diff < 0 else 0
        if abs(diff) == 1 and rotation == 1:
            val = 0
        elif abs(diff) == 1 and rotation == 0:
            val = 1
        elif abs(diff) == 2 and rotation == 1:
            val = 2
        elif abs(diff) == 2 and rotation == 0:
            val = 3
        else:
            val = 4
        return val

    def ldspValue(img, x, y):
        edge = []
        masks = [
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.float32),
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.float32),
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float32),
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.float32),
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.float32),
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.float32),
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.float32),
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.float32)
        ]

        for mask in masks:
            edge.append(calculateMask(img, mask, x, y))

        sortedEdge = np.sort(edge)
        p1 = edge.index(sortedEdge[-1])
        p2 = edge.index(sortedEdge[-2])
        s = calculateLDSP(p1, p2)

        if s == 4:
            ldspVal = img[x][y]
        else:
            ldspVal = 4 * p1 + s

        return ldspVal

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            ldsp[i][j] = ldspValue(img_gray, i, j)

    ldsp_features = np.histogram(ldsp, bins=range(65))[0]

    return ldsp_features

dataset_path = "Deep_Fashion_Dataset - Feature/img"  # Replace with the path to your dataset

csv_path = "Features DB/LDSP_FEATURES.csv"  # Path to save the CSV file
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                print("Processing image:", image_path)
                features = compute_ldsp_features(image_path)
                # Append the image_path to the features list
                features_with_path = list(features) + [image_path]

                writer.writerow(features_with_path)

print("LDSP features saved to:", csv_path)
