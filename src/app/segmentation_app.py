import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image


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
    # Usamos um conjunto para verificação mais eficiente
    target_colors = set(colors)

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

                stack.extend((n for n in neighbors_4 if 0 <=
                             n[0] < width and 0 <= n[1] < height))
    return region


def contour_segmented_images(segmented_image):
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)
    contours_per_image, _ = cv2.findContours(
        inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours_per_image, key=cv2.contourArea)
    return largest_contour


def apply_mask(image_data, contours):
    mask = np.zeros_like(image_data, dtype=np.uint8)
    cv2.drawContours(mask, [contours], 0,
                     (255, 255, 255), thickness=cv2.FILLED)
    segmented_image_with_contour = cv2.bitwise_and(np.array(image_data), mask)
    return segmented_image_with_contour


def process_segmentation(image_data):
    segmented_images = []

    for image in image_data:
        center_colors = get_colors_around_center_pixel(image, radius=4)
        seed = (image.width // 2, image.height // 2)
        threshold = 30
        visited = np.zeros((image.width, image.height), dtype=bool)
        region = region_growing(image, seed, center_colors, threshold, visited)
        segmented_image = Image.new('RGB', image.size, (255, 255, 255))
        for pixel in region:
            segmented_image.putpixel(pixel, image.getpixel(pixel))
        segmented_images.append(segmented_image)
    return segmented_images


def process_contours(segmented_images, image_data):
    final_segmentation = []

    for i in range(0, len(segmented_images)):
        array_image = np.array(segmented_images[i])
        contours = contour_segmented_images(array_image)
        final_segmentation_image = apply_mask(image_data[i], contours)
        final_segmentation.append(final_segmentation_image)
    return final_segmentation


def main_process_segmentation(image_data):
    segmented_images = process_segmentation(image_data)
    final_segmentation = process_contours(segmented_images, image_data)
    return final_segmentation
