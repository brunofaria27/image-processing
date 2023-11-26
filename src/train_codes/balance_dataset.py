from PIL import Image
import os

def augment_images(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    classes = os.listdir(input_path)

    for class_name in classes:
        class_path = os.path.join(input_path, class_name)
        output_class_path = os.path.join(output_path, class_name)

        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        files = os.listdir(class_path)

        for file in files:
            original_path = os.path.join(class_path, file)
            original_image = Image.open(original_path)

            original_save_path = os.path.join(output_class_path, f'{file.split(".")[0]}_ID_original.png')
            original_image.save(original_save_path)

            if class_name != 'Negative for intraepithelial lesion':
                save_augmented_images(original_image, output_class_path, file)

def save_augmented_images(original_image, output_path, file_name):
    for angle in [90, 180, 270]:
        rotated_image = original_image.rotate(angle)
        save_path = os.path.join(output_path, f'{file_name.split(".")[0]}_ID_{angle}.png')
        rotated_image.save(save_path)

        mirrored_rotated_image = rotated_image.transpose(Image.FLIP_LEFT_RIGHT)
        mirrored_rotated_save_path = os.path.join(output_path, f'{file_name.split(".")[0]}_ID_{angle}_mirror.png')
        mirrored_rotated_image.save(mirrored_rotated_save_path)

    mirrored_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)
    save_path = os.path.join(output_path, f'{file_name.split(".")[0]}_ID_mirror.png')
    mirrored_image.save(save_path)

    mirrored_rotated_image = original_image.rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
    save_path = os.path.join(output_path, f'{file_name.split(".")[0]}_ID_90_mirror.png')
    mirrored_rotated_image.save(save_path)

    mirrored_rotated_image = original_image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
    save_path = os.path.join(output_path, f'{file_name.split(".")[0]}_ID_180_mirror.png')
    mirrored_rotated_image.save(save_path)

    mirrored_rotated_image = original_image.rotate(270).transpose(Image.FLIP_LEFT_RIGHT)
    save_path = os.path.join(output_path, f'{file_name.split(".")[0]}_ID_270_mirror.png')
    mirrored_rotated_image.save(save_path)