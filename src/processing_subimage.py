import os
import csv
import cv2

# Diretórios
IMAGE_DIR = 'D:\GitHub\image-processing\images'
OUTPUT_DIR = 'D:\GitHub\image-processing\images-processed'
CSV_FILE = 'D:\GitHub\image-processing\dataset\classifications.csv'

# Classes
CLASSES = ['Negative for intraepithelial lesion', 'ASC-US', 'ASC-H', 'LSIL', 'HSIL', 'SCC']

def createFoldersClasses(classes: list, output_dir: str) -> bool or Exception:
    """
    Criação dos sub-diretórios com o nome de classe.
    """
    try:
        for class_name in classes:
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
        print('Diretórios criados com sucesso.')
        return True
    except Exception as e:
        raise Exception(f'Erro ao criar diretórios: {str(e)}')

def processingImages(csv_name: str, image_dir: str, output_dir: str, image_size: int) -> None:
    """
    Leitura do arquivo e chamada de funções de tratamento.
    Utiliza-se:
        - DictReader: Ler o CSV como dicionário, onde os nomes das colunas são usados como chaves.
    """
    with open(csv_name, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        processingImageScale(csv_reader=csv_reader, image_dir=image_dir, output_dir=output_dir, image_size=image_size)

def processingImageScale(csv_reader: str, image_dir: str, output_dir: str, image_size: int) -> None or Exception:
    """
    Função usada para percorrer todas as linhas do csv e criar as imagens 100x100 a partir das coordenadas dos núcleos.
    Tratamento de erros ocorrem para ver se a imagem e válida ou se o arquivo dela e suportado.
    """
    try:
        for row in csv_reader:
            image_id = int(row['image_id'])
            image_filename = row['image_filename']
            cell_id = int(row['cell_id'])
            bethesda_system = row['bethesda_system']
            nucleus_x = int(row['nucleus_x'])
            nucleus_y = int(row['nucleus_y'])

            image_path = os.path.join(image_dir, image_filename)

            # Trata a condição de não ter todas images disponiveis
            if os.path.exists(image_path):
                image = cv2.imread(image_path)

                if image is not None:
                    x_center = nucleus_x
                    y_center = nucleus_y

                    x1 = max(0, x_center - image_size // 2)
                    x2 = min(image.shape[1], x_center + image_size // 2)
                    y1 = max(0, y_center - image_size // 2)
                    y2 = min(image.shape[0], y_center + image_size // 2)
                    
                    if x1 < x2 and y1 < y2: # Garantir recorte correto
                        sub_image = image[y1:y2, x1:x2]  # Recorte da imagem

                        class_dir = os.path.join(output_dir, bethesda_system)
                        output_filename = f"{cell_id}.png"
                        output_path = os.path.join(class_dir, output_filename)
                        cv2.imwrite(output_path, sub_image)
                    else:
                        print(f'Coord: {x_center}, {y_center} -> {image_path}')
    except Exception as e:
        raise Exception(f'Erro ao criar sub imagens: {str(e)}')

def main():
    createFoldersClasses(classes=CLASSES, output_dir=OUTPUT_DIR)
    processingImages(csv_name=CSV_FILE, image_dir=IMAGE_DIR, output_dir=OUTPUT_DIR, image_size=100)

main()