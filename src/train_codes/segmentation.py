# Imports
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

from PIL import Image

from utils import separate_dataset, separate_negative_to_others_dataset, write_segmented_images, create_folders_classes
from balance_dataset import augment_images

def read_images(dir_name):
    base_dir = dir_name

    image_data = {}

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        
        if os.path.isdir(folder_path):
            images = []
            
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                
                if file_path.lower().endswith(('.png', '.jpg')):
                    image = Image.open(file_path)
                    images.append(image)
            
            image_data[folder] = images
    return image_data

def get_colors_around_center_pixel(image, radius=5):
    height, width = image.size
    center_x, center_y = width // 2, height // 2

    colors = []

    for x in range(center_x - radius, center_x + radius + 1):
        for y in range(center_y - radius, center_y + radius + 1):
            if 0 <= x < width and 0 <= y < height:
                pixel_color = image.getpixel((x, y))
                colors.append(pixel_color)

    return colors

def hsv_color_similarity(color1, color2):
    hue_diff = abs(color1[0] - color2[0])
    sat_diff = abs(color1[1] - color2[1])
    val_diff = abs(color1[2] - color2[2])
    return hue_diff + sat_diff + val_diff

def region_growing(image, seed, colors, threshold, visited):
    width, height = image.size
    stack = [seed]
    region = []
    target_colors = set(colors)  # Usamos um conjunto para verificação mais eficiente

    while stack:
        x, y = stack.pop()
        if not visited[x, y]:
            visited[x, y] = True
            pixel_color = image.getpixel((x, y))
            if any(hsv_color_similarity(pixel_color, target_color) <= threshold for target_color in target_colors):
                region.append((x, y))

                neighbors_8 = [
                    (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
                    (x - 1, y), (x + 1, y),
                    (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)
                ]

                neighbors_4 = [
                    (x, y - 1),
                    (x - 1, y),
                    (x + 1, y),
                    (x, y + 1)
                ]
                
                stack.extend((n for n in neighbors_4 if 0 <= n[0] < width and 0 <= n[1] < height))

    return region

def contour_segmented_images(segmented_image):
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)
    contours_per_image, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours_per_image, key=cv2.contourArea)
    return largest_contour

def apply_mask(image_data, contours):
    mask = np.zeros_like(image_data, dtype=np.uint8)
    cv2.drawContours(mask, [contours], 0, (255, 255, 255), thickness=cv2.FILLED)
    segmented_image_with_contour = cv2.bitwise_and(np.array(image_data), mask)
    return segmented_image_with_contour


####################################
#            PROCESSES             #
####################################

def process_segmentation(image_data):
    segmented_images = {}

    for folder, images in image_data.items():
        segmented_image_list = []

        for image in images:
            center_colors = get_colors_around_center_pixel(image, radius=4)
            seed = (image.width // 2, image.height // 2)
            threshold = 30  # Ajuste este valor de acordo com a similaridade desejada
            visited = np.zeros((image.width, image.height), dtype=bool)
            region = region_growing(image, seed, center_colors, threshold, visited)

            # Crie uma nova imagem com o fundo branco e depois adicione a região segmentada
            segmented_image = Image.new('RGB', image.size, (255, 255, 255))
            for pixel in region:
                segmented_image.putpixel(pixel, image.getpixel(pixel))
            segmented_image_list.append(segmented_image)
            # plot_single_image_and_original(segmented_image, image) # Plotar as imagens conforme for segmentando para ver se está sendo certo
        segmented_images[folder] = segmented_image_list
    return segmented_images

def process_contours(segmented_images, image_data):
    final_segmentation = {}
    for folder, images in segmented_images.items():
        final_segmentation_list = []

        num_images = len(images)
        for i in range(0, num_images):
            array_image = np.array(segmented_images.get(folder)[i])
            contours = contour_segmented_images(array_image)
            final_segmentation_image = apply_mask(image_data.get(folder)[i], contours)
            # casting_image = Image.fromarray(final_segmentation_image) # Caso precise
            final_segmentation_list.append(final_segmentation_image)
        final_segmentation[folder] = final_segmentation_list
    return final_segmentation

def main():
    print('Lendo imagens originais...')
    image_data = read_images('../images-processed/')

    print('Segmentando imagens originais...')
    segmented_images = process_segmentation(image_data)

    print('Fazendo contorno e máscara da imagem segmentada...')
    final_segmentation = process_contours(segmented_images, image_data)
    CLASSES = ['Negative for intraepithelial lesion', 'ASC-US', 'ASC-H', 'LSIL', 'HSIL', 'SCC']

    print('Criando pastas para guardar as imagens segmentadas...')
    create_folders_classes(CLASSES)

    print('Guardando imagens segmentadas...')
    write_segmented_images(image_data, final_segmentation, '../segmented-images/')

    dataset_path = '../segmented-images'
    output_path = '../separate-dataset'
    output_path_bin = '../separate-bin-dataset'
    output_augumented_path = '../augmented-images'

    print('Aumentando os datasets aplicando rotacoes e espelhamentos')
    # Aumenta a quantidade de dados em classes != Negative
    augment_images(dataset_path, output_augumented_path)

    print('Separando dataset em treino e teste...')
    separate_dataset(output_augumented_path, output_path, target_images_per_class=650, percentage_train=0.8)

    print('Separando dataset em treino e teste binario...')
    separate_negative_to_others_dataset(output_augumented_path, output_path_bin, target_images_per_class=3000, percentage_train=0.8)

if __name__ == "__main__":
    main()