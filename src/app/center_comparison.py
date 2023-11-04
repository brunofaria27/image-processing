import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

class CenterComparison:
    def __init__(self, window, window_title, distances_to_original_center, segmented_images):
        self.window = window
        self.window.title(window_title)

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        initial_width = int(screen_width * 0.2)
        initial_height = int(screen_height * 0.4)
        x = (screen_width - initial_width) // 2
        y = (screen_height - initial_height) // 2
        self.window.geometry(f"{initial_width}x{initial_height}+{x}+{y}")

        self.distances_to_original_center = distances_to_original_center
        self.segmented_images = segmented_images

        self.canvas = tk.Canvas(self.window)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Increase the width of the vertical scrollbar
        self.scrollbar = tk.Scrollbar(self.window, orient=tk.VERTICAL, command=self.canvas.yview, relief='ridge', width=20)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.frame, anchor='nw')

        self.image_labels = []
        self.load_images_with_info()

        self.frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox('all'))

        self.window.mainloop()

    def load_images_with_info(self):
        for i, image_data in enumerate(self.distances_to_original_center):
            image_id = image_data['image_id']
            distance = image_data['distance_to_nucleus']
            coords = image_data['coords']

            image_with_centers = self.convert_with_points(self.segmented_images[i], coords)
            img = Image.fromarray(cv2.cvtColor(image_with_centers, cv2.COLOR_BGR2RGB))
            img = img.resize((300, 300))

            label = tk.Label(self.frame, image=None, text=f"ID: {image_id} - Distance: {distance}", compound="top", font=("Helvetica", 14))
            label.img = ImageTk.PhotoImage(img)
            label.config(image=label.img)
            label.grid(row=i, column=0, padx=10, pady=10)

            self.image_labels.append(label)

    def convert_with_points(self, segmented_image, coords):
        center_og = segmented_image.shape
        center_og_x = center_og[1] // 2
        center_og_y = center_og[0] // 2
        coord_x = coords[0]
        coord_y = coords[1]
        image_with_centers = segmented_image.copy()
        image_with_centers[coord_y, coord_x] = [0, 0, 255]
        image_with_centers[center_og_y, center_og_x] = [0, 255, 0]
        return image_with_centers