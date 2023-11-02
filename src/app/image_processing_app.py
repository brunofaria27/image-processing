import os
import csv
import cv2
import numpy as np

# Variáveis globais
IMAGE_DIR = './images/'
CSV_FILE = './data/classifications.csv'

def process_image(image_filename, n_size, cell_id=None):
    if image_filename is not None:
        image_path = os.path.join(IMAGE_DIR, image_filename)
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                if cell_id is not None:
                    cropped_images = cut_single_cell(image_filename, image, cell_id, n_size)
                else:
                    cropped_images = cut_all_cells(image_filename, image, n_size)
    return cropped_images

def cut_single_cell(image_filename, image, cell_id, n_size):
    cropped_images = []
    with open(CSV_FILE, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            if row['image_filename'] == image_filename and int(row['cell_id']) == int(cell_id):
                print('achou')
                nucleus_x = int(row['nucleus_x'])
                nucleus_y = int(row['nucleus_y'])

                x_center = nucleus_x
                y_center = nucleus_y

                x1 = x_center - n_size // 2
                x2 = x1 + n_size
                y1 = y_center - n_size // 2
                y2 = y1 + n_size

                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > image.shape[1]:
                    x2 = image.shape[1]
                if y2 > image.shape[0]:
                    y2 = image.shape[0]

                sub_image = image[y1:y2, x1:x2]

                if sub_image.shape[0] != n_size or sub_image.shape[1] != n_size:
                    temp_image = np.full((n_size, n_size, 3), 255, dtype=np.uint8)

                    dx = n_size - sub_image.shape[1]
                    dy = n_size - sub_image.shape[0]

                    temp_image[dy:dy + sub_image.shape[0], dx:dx + sub_image.shape[1]] = sub_image
                    sub_image = temp_image

                cropped_images.append(sub_image)
    return cropped_images

def cut_all_cells(image_filename, image, n_size):
    cropped_images = []
    with open(CSV_FILE, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            if row['image_filename'] == image_filename:
                nucleus_x = int(row['nucleus_x'])
                nucleus_y = int(row['nucleus_y'])

                x_center = nucleus_x
                y_center = nucleus_y

                x1 = x_center - n_size // 2
                x2 = x1 + n_size
                y1 = y_center - n_size // 2
                y2 = y1 + n_size

                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 > image.shape[1]:
                    x2 = image.shape[1]
                if y2 > image.shape[0]:
                    y2 = image.shape[0]

                sub_image = image[y1:y2, x1:x2]

                if sub_image.shape[0] != n_size or sub_image.shape[1] != n_size:
                    temp_image = np.full((n_size, n_size, 3), 255, dtype=np.uint8)

                    dx = n_size - sub_image.shape[1]
                    dy = n_size - sub_image.shape[0]

                    temp_image[dy:dy + sub_image.shape[0], dx:dx + sub_image.shape[1]] = sub_image
                    sub_image = temp_image
                cropped_images.append(sub_image)
    return cropped_images
