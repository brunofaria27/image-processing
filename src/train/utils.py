import os
import matplotlib.pyplot as plt

from PIL import Image

# Funções para plotar as imagens 
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
