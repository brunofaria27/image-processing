import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import cv2
import csv

# TODO: Tentar ver se o cálculo do perimetro e da excentricidade estão corretos.

def calculate_area(image, black_color=[0, 0, 0]):
    non_black_pixels = np.argwhere(np.all(np.array(image) != black_color, axis=-1))
    return len(non_black_pixels)

def calculate_compactness(area, contour):
    perimeter = cv2.arcLength(contour, True)
    compactness = (perimeter ** 2) / (4 * np.pi * area)
    return compactness

def calculate_eccentricity(image, black_color=[0, 0, 0]):
    non_black_pixels = np.argwhere(np.all(np.array(image) != black_color, axis=-1))
    moments = cv2.moments(non_black_pixels)
    Ixx = moments['mu20']
    Iyy = moments['mu02']
    Ixy = moments['mu11']
    a = (Ixx + Iyy) / 2 + np.sqrt(4 * Ixy**2 + (Ixx - Iyy)**2) / 2
    b = (Ixx + Iyy) / 2 - np.sqrt(4 * Ixy**2 + (Ixx - Iyy)**2) / 2
    eccentricity = np.sqrt(1 - b/a)
    return eccentricity

def extract_features(segmented_images, contours_final_segmentation, cells_ids):
    data = {'Class': [], 'Area': [], 'Compactness': [], 'Eccentricity': []}

    for i in range(0, len(segmented_images)):
        classfification_csv = search_bethesda_system_cell_id(cells_ids[i])
        contour = contours_final_segmentation[i]
        area = calculate_area(segmented_images[i])
        compactness = calculate_compactness(area, contour)
        eccentricity = calculate_eccentricity(segmented_images[i])

        if math.isnan(eccentricity):
            eccentricity = 0

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