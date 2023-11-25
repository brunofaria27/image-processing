import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import cv2
import csv

def calculate_area(image, black_color=[0, 0, 0]):
    non_black_pixels = np.argwhere(np.all(np.array(image) != black_color, axis=-1))
    return len(non_black_pixels)

def calculate_compactness(area, contour):
    perimeter = cv2.arcLength(contour, True)
    compactness = (perimeter ** 2) / (4 * np.pi * area)
    return compactness

def calculate_eccentricity(contour):
    ellipse = cv2.fitEllipse(contour)
    major_axis = max(ellipse[1])
    minor_axis = min(ellipse[1])
    eccentricity = np.sqrt(1 - (minor_axis ** 2) / (major_axis ** 2))
    return eccentricity

def extract_features(segmented_images, contours_final_segmentation, cells_ids):
    data = {'Class': [], 'Area': [], 'Compactness': [], 'Eccentricity': []}

    for i in range(0, len(segmented_images)):
        classfification_csv = search_bethesda_system_cell_id(cells_ids[i])
        area = calculate_area(segmented_images[i])
        gray_image = cv2.cvtColor(np.array(segmented_images[i]), cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        compactness = calculate_compactness(area, contour)
        eccentricity = calculate_eccentricity(contour)

        data['Class'].append(classfification_csv)
        data['Area'].append(area)
        data['Compactness'].append(compactness)
        data['Eccentricity'].append(eccentricity)
    features_df = pd.DataFrame(data)
    features_df.to_csv('csv_characterization/features.csv', index=False)
    return features_df


def plot_scatterplot(features_df):
    unique_classes = features_df['Class'].unique()
    class_palette = {class_label: 'black' if class_label == 'Negative for intraepithelial lesion' else sns.color_palette()[i] for i, class_label in enumerate(unique_classes)}

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Area', y='Compactness', hue='Class', data=features_df, palette=class_palette, s=80)
    plt.title('Scatterplot of Area vs Compactness')
    plt.xlabel('Area')
    plt.ylabel('Compactness')
    plt.legend()
    plt.show()

def search_bethesda_system_cell_id(cell_id):
    result = str()
    with open('data/classifications.csv', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        
        for row in csv_reader:
            if row['cell_id'] == str(cell_id):
                result = row['bethesda_system']
    return result