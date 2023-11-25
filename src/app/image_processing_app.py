import os
import csv
import cv2
import numpy as np

# Variáveis globais
IMAGE_DIR = './images/'
CSV_FILE = './data/classifications.csv'

def process_image(image_filename, n_size, cell_id=None):
    cropped_images = []
    if image_filename is not None:
        image_path = os.path.join(IMAGE_DIR, image_filename)
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            with open(CSV_FILE, 'r') as csvfile:
                csv_reader = csv.DictReader(csvfile)
                if image is not None:
                    cropped_images = cut_cells(image_filename, image, cell_id, n_size, csv_reader, id=cell_id)
    return cropped_images

def cut_cells(image_filename, image, cell_id, n_size, csv_reader, id=None):
    cropped_images = []
    for row in csv_reader:
        if id is None:
            if row['image_filename'] == image_filename:
                nucleus_x = int(row['nucleus_x'])
                nucleus_y = int(row['nucleus_y'])
                sub_image = cut_cell_process(image, nucleus_x, nucleus_y, n_size)
                cropped_images.append(sub_image)
        else:
            if row['image_filename'] == image_filename and int(row['cell_id']) == int(id):
                nucleus_x = int(row['nucleus_x'])
                nucleus_y = int(row['nucleus_y'])
                sub_image = cut_cell_process(image, nucleus_x, nucleus_y, n_size)
                cropped_images.append(sub_image)
    return cropped_images

def cut_cell_process(image, x_center, y_center, n_size):
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

        temp_image[dy:dy + sub_image.shape[0],
                   dx:dx + sub_image.shape[1]] = sub_image
        sub_image = temp_image
    return sub_image
