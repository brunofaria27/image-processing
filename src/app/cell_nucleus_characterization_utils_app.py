import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
import csv

from PIL import Image

def calculate_area(image, black_color=[0, 0, 0]):
    non_black_pixels = np.argwhere(np.all(np.array(image) != black_color, axis=-1))
    return len(non_black_pixels)

def calculate_compactness(area, contour):
    perimeter = cv2.arcLength(contour, True)
    compactness = (4 * np.pi * area) / (perimeter ** 2)
    return compactness

def calculate_eccentricity(contour):
    try:
        ellipse = cv2.fitEllipse(contour)
        major_axis = max(ellipse[1])
        minor_axis = min(ellipse[1])
        eccentricity = np.sqrt(1 - (minor_axis ** 2) / (major_axis ** 2))
        return eccentricity
    except Exception:
        return 0

def extract_features_binary(segmented_images, cells_ids):
    data = {'Class': [], 'Area': [], 'Compactness': [], 'Eccentricity': []}

    for i in range(0, len(segmented_images)):
        classfication_csv = search_bethesda_system_cell_id(cells_ids[i])

        if classfication_csv != 'Negative for intraepithelial lesion':
            classfication_csv = 'Others'

        area = calculate_area(segmented_images[i])
        gray_image = cv2.cvtColor(np.array(segmented_images[i]), cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        compactness = calculate_compactness(area, contour)
        eccentricity = calculate_eccentricity(contour)

        data['Class'].append(classfication_csv)
        data['Area'].append(area)
        data['Compactness'].append(compactness)
        data['Eccentricity'].append(eccentricity)

    features_df = pd.DataFrame(data)
    features_df.to_csv('csv_characterization/features_binary.csv', index=False)
    return features_df

def extract_features_multiclass(segmented_images, cells_ids):
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
    features_df.to_csv('csv_characterization/features_multiclass.csv', index=False)
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

def load_data(path_name):
    data = {}
    for _class in os.listdir(path_name):
        folder_path = os.path.join(path_name, _class)
        if os.path.isdir(folder_path):
            images = []
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if file_path.lower().endswith(('.png', '.jpg')):
                    with Image.open(file_path) as image:
                        images.append(np.array(image))
            data[_class] = images
    return data

def load_data_all(path_name, data_dict):
    data = {}
    for _class in os.listdir(path_name):
        folder_path = os.path.join(path_name, _class)
        if os.path.isdir(folder_path):
            images = []
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if file_path.lower().endswith(('.png', '.jpg')):
                    with Image.open(file_path) as image:
                        images.append(np.array(image))
                        data_dict['ID'].append(os.path.splitext(filename)[0]) # Esse novo load foi necessário para ter os ID's das imagens
            data[_class] = images
    return data

def process_dataset(data, data_dict):
    for class_name, images in data.items():
        for image in images:
            area = calculate_area(image)
            gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea)
            compactness = calculate_compactness(area, contour)
            eccentricity = calculate_eccentricity(contour)

            data_dict['Area'].append(area)
            data_dict['Compactness'].append(compactness)
            data_dict['Eccentricity'].append(eccentricity)
            data_dict['Class'].append(class_name)

def extract_all_images():
    train_data_path_binary = 'separate-bin-dataset-segmented/train'
    test_data_path_binary = 'separate-bin-dataset-segmented/test'

    train_data_path_multiclass = 'separate-dataset-segmented/train'
    test_data_path_multiclass = 'separate-dataset-segmented/test'

    data_path_all_multiclass = 'segmented-images'
    data_path_all_binary = 'separate-bin-segmented'

    train_data_binary = load_data(train_data_path_binary)
    test_data_binary = load_data(test_data_path_binary)

    train_data_multiclass = load_data(train_data_path_multiclass)
    test_data_multiclass = load_data(test_data_path_multiclass)

    all_data_dict_multiclass = {'ID': [], 'Class': [], 'Area': [], 'Compactness': [], 'Eccentricity': []}
    all_data_dict_binary = {'ID': [], 'Class': [], 'Area': [], 'Compactness': [], 'Eccentricity': []}

    data_all_multiclass = load_data_all(data_path_all_multiclass, all_data_dict_multiclass)
    data_all_binary = load_data_all(data_path_all_binary, all_data_dict_binary)

    test_data_dict_binary = {'Class': [], 'Area': [], 'Compactness': [], 'Eccentricity': []}
    train_data_dict_binary = {'Class': [], 'Area': [], 'Compactness': [], 'Eccentricity': []}

    test_data_dict_multiclass = {'Class': [], 'Area': [], 'Compactness': [], 'Eccentricity': []}
    train_data_dict_multiclass = {'Class': [], 'Area': [], 'Compactness': [], 'Eccentricity': []}

    process_dataset(train_data_binary, train_data_dict_binary)
    process_dataset(test_data_binary, test_data_dict_binary)

    process_dataset(train_data_multiclass, train_data_dict_multiclass)
    process_dataset(test_data_multiclass, test_data_dict_multiclass)

    process_dataset(data_all_multiclass, all_data_dict_multiclass)
    process_dataset(data_all_binary, all_data_dict_binary)

    train_df_binary = pd.DataFrame(train_data_dict_binary)
    test_df_binary = pd.DataFrame(test_data_dict_binary)

    train_df_multiclass = pd.DataFrame(train_data_dict_multiclass)
    test_df_multiclass = pd.DataFrame(test_data_dict_multiclass)

    all_df_multiclass = pd.DataFrame(all_data_dict_multiclass)
    all_df_binary = pd.DataFrame(all_data_dict_binary)

    train_df_binary.to_csv('csv_characterization/train_features_binary.csv', index=False)
    test_df_binary.to_csv('csv_characterization/test_features_binary.csv', index=False)

    train_df_multiclass.to_csv('csv_characterization/train_features_multiclass.csv', index=False)
    test_df_multiclass.to_csv('csv_characterization/test_features_multiclass.csv', index=False)

    all_df_multiclass.to_csv('csv_characterization/all_features_multiclass.csv', index=False)
    all_df_binary.to_csv('csv_characterization/all_features_binary.csv', index=False)
    return train_df_binary, train_df_multiclass, all_df_multiclass, all_df_binary