import os
import csv
import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, image_dir, output_dir, image_size):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.image_size = image_size
    
    def create_folders_classes(self, classes):
        try:
            for class_name in classes:
                class_dir = os.path.join(self.output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
            print('Diretórios criados com sucesso.')
            return True
        except Exception as e:
            raise Exception(f'Erro ao criar diretórios: {str(e)}')

    def processing_images(self, csv_name):
        with open(csv_name, 'r') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            self.processing_image_scale(csv_reader=csv_reader)

    def processing_image_scale(self, csv_reader):
        try:
            for row in csv_reader:
                image_id = int(row['image_id'])
                image_filename = row['image_filename']
                cell_id = int(row['cell_id'])
                bethesda_system = row['bethesda_system']
                nucleus_x = int(row['nucleus_x'])
                nucleus_y = int(row['nucleus_y'])

                image_path = os.path.join(self.image_dir, image_filename)

                if os.path.exists(image_path):
                    image = cv2.imread(image_path)

                    if image is not None:
                        x_center = nucleus_x
                        y_center = nucleus_y

                        x1 = x_center - self.image_size // 2
                        x2 = x1 + self.image_size
                        y1 = y_center - self.image_size // 2
                        y2 = y1 + self.image_size

                        if x1 < 0:
                            x1 = 0
                        if y1 < 0:
                            y1 = 0
                        if x2 > image.shape[1]:
                            x2 = image.shape[1]
                        if y2 > image.shape[0]:
                            y2 = image.shape[0]

                        sub_image = image[y1:y2, x1:x2]

                        if sub_image.shape[0] != self.image_size or sub_image.shape[1] != self.image_size:
                            temp_image = np.full((self.image_size, self.image_size, 3), 255, dtype=np.uint8)
                            
                            # Ajusta a posição do preenchimento com base na direção em que a imagem fugiu dos limites
                            dx = self.image_size - sub_image.shape[1]
                            dy = self.image_size - sub_image.shape[0]
                            
                            temp_image[dy:dy+sub_image.shape[0], dx:dx+sub_image.shape[1]] = sub_image
                            sub_image = temp_image

                        class_dir = os.path.join(self.output_dir, bethesda_system)
                        output_filename = f"{cell_id}.png"
                        output_path = os.path.join(class_dir, output_filename)
                        cv2.imwrite(output_path, sub_image)
        except Exception as e:
            raise Exception(f'Erro ao criar sub imagens: {str(e)}')

def main():
    # Diretórios
    IMAGE_DIR = 'E:\GitHub Projects\image-processing\src\images'
    OUTPUT_DIR = 'E:\GitHub Projects\image-processing\src\images-processed'
    CSV_FILE = 'E:\GitHub Projects\image-processing\src\data\classifications.csv'

    # Classes
    CLASSES = ['Negative for intraepithelial lesion', 'ASC-US', 'ASC-H', 'LSIL', 'HSIL', 'SCC']
    
    image_processor = ImageProcessor(image_dir=IMAGE_DIR, output_dir=OUTPUT_DIR, image_size=100)
    image_processor.create_folders_classes(classes=CLASSES)
    image_processor.processing_images(csv_name=CSV_FILE)

if __name__ == "__main__":
    main()