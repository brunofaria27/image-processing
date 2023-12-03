import os
import cv2
import numpy as np
import pandas as pd
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
                        data_dict['ID'].append(os.path.splitext(filename)[0])
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

# Load the data
train_data_path_binary = '../separate-bin-dataset/train'
test_data_path_binary = '../separate-bin-dataset/test'

train_data_path_multiclass = '../separate-dataset/train'
test_data_path_multiclass = '../separate-dataset/test'

data_path_all_multiclass = '../segmented-images'
data_path_all_binary = '../separate-bin-segmented'

train_data_binary = load_data(train_data_path_binary)
test_data_binary = load_data(test_data_path_binary)

train_data_multiclass = load_data(train_data_path_multiclass)
test_data_multiclass = load_data(test_data_path_multiclass)

all_data_dict_multiclass = {'ID': [], 'Class': [], 'Area': [], 'Compactness': [], 'Eccentricity': []}
all_data_dict_binary = {'ID': [], 'Class': [], 'Area': [], 'Compactness': [], 'Eccentricity': []}

data_all_multiclass = load_data_all(data_path_all_multiclass, all_data_dict_multiclass)
data_all_binary = load_data_all(data_path_all_binary, all_data_dict_binary)

# Create a DataFrame to store the results
test_data_dict_binary = {'Class': [], 'Area': [], 'Compactness': [], 'Eccentricity': []}
train_data_dict_binary = {'Class': [], 'Area': [], 'Compactness': [], 'Eccentricity': []}

test_data_dict_multiclass = {'Class': [], 'Area': [], 'Compactness': [], 'Eccentricity': []}
train_data_dict_multiclass = {'Class': [], 'Area': [], 'Compactness': [], 'Eccentricity': []}

# Populate the training dataset binary
process_dataset(train_data_binary, train_data_dict_binary)
process_dataset(test_data_binary, test_data_dict_binary)

# Populate the training dataset multiclass
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

