import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from PIL import Image
    
def display_all_images(data, num_images_to_display=10):
    for folder, images in data.items():
        print(f"Folder: {folder}")
        num_images = len(images)
        
        for i in range(0, num_images, num_images_to_display):
            fig, axes = plt.subplots(1, min(num_images_to_display, num_images - i), figsize=(15, 5))
            
            if num_images_to_display == 1:
                axes = [axes]
                
            for j, ax in enumerate(axes):
                if j + i < num_images:
                    ax.imshow(images[i + j], cmap='gray')
                    ax.axis('off')
            
            plt.show()

def plot_single_image_and_original(segmented_image, original_image):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title('Segmented Image')
    plt.axis('off')

    plt.show()

# Funções para guardar as imagens segmentadas em uma pasta
def create_folders_classes(classes):
        try:
            for class_name in classes:
                class_dir = os.path.join('../segmented-images/', class_name)
                os.makedirs(class_dir, exist_ok=True)
            print('Diretórios criados com sucesso.')
            return True
        except Exception as e:
            raise Exception(f'Erro ao criar diretórios: {str(e)}')

def write_segmented_images(image_data, segmented_images, output_dir):
    for folder, images in image_data.items():
        segmented_folder = os.path.join(output_dir, folder)
        os.makedirs(segmented_folder, exist_ok=True)

        for image, segmented_image in zip(images, segmented_images[folder]):
            image_name = os.path.splitext(os.path.basename(image.filename))[0]
            output_filename = f"{image_name}_segmented.jpg"
            output_path = os.path.join(segmented_folder, output_filename)
            segmented_image = Image.fromarray(segmented_image)
            segmented_image.save(output_path)
    print(f'Todas as imagens carregadas para {output_dir}')

def separate_dataset(dataset_path, output_path, percentage_train=0.8):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for _class in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, _class)
        if os.path.isdir(class_path):
            files_class = os.listdir(class_path)

            random.shuffle(files_class)
            index_train = int(len(files_class) * percentage_train)
            train_data = files_class[:index_train]
            test_data = files_class[index_train:]

            train_path = os.path.join(output_path, 'train', _class)
            test_path = os.path.join(output_path, 'test', _class)
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)

            for _file in train_data:
                src_path = os.path.join(class_path, _file)
                dest_path = os.path.join(train_path, _file)
                shutil.copy2(src_path, dest_path)

            for _file in test_data:
                src_path = os.path.join(class_path, _file)
                dest_path = os.path.join(test_path, _file)
                shutil.copy2(src_path, dest_path)

        # Count and print the number of files in each folder
        num_train_files = len(os.listdir(train_path))
        num_test_files = len(os.listdir(test_path))
        print(f'Class {_class}: {num_train_files} files in training set, {num_test_files} files in test set')
    print(f'Images separated into training ({percentage_train}) and testing sets in the directory {output_path}')

def count_files_in_folders(folder_path):
    num_files = len(os.listdir(folder_path))
    return num_files

def separate_negative_to_others_dataset(dataset_path, output_path, percentage_train=0.8):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    negative_class = "Negative for intraepithelial lesion"
    
    for _class in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, _class)
        if os.path.isdir(class_path):
            files_class = os.listdir(class_path)

            # Separate files into training and testing sets
            random.shuffle(files_class)
            index_train = int(len(files_class) * percentage_train)
            train_data = files_class[:index_train]
            test_data = files_class[index_train:]

            # Create directories for training and testing sets
            if _class == negative_class:
                train_path = os.path.join(output_path, 'train', negative_class)
                test_path = os.path.join(output_path, 'test', negative_class)
            else:
                train_path = os.path.join(output_path, 'train', 'Others')
                test_path = os.path.join(output_path, 'test', 'Others')

            os.makedirs(train_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)

            # Copy files to training and testing sets
            for _file in train_data:
                src_path = os.path.join(class_path, _file)
                dest_path = os.path.join(train_path, _file)
                shutil.copy2(src_path, dest_path)

            for _file in test_data:
                src_path = os.path.join(class_path, _file)
                dest_path = os.path.join(test_path, _file)
                shutil.copy2(src_path, dest_path)

        # Count and print the number of files in each folder for Negative and Others classes
        num_train_negative_files = count_files_in_folders(train_path)
        num_test_negative_files = count_files_in_folders(test_path)
        
        if _class == negative_class:
            print(f'Class {negative_class}: {num_train_negative_files} files in training set, {num_test_negative_files} files in test set')
        else:
            num_train_others_files = count_files_in_folders(train_path)
            num_test_others_files = count_files_in_folders(test_path)
            print(f'Class Others: {num_train_others_files} files in training set, {num_test_others_files} files in test set')
    print(f'Images separated into training ({percentage_train}) and testing sets for Negative class and Others in the directory {output_path}')