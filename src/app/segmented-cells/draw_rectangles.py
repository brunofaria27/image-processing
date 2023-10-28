import cv2
import pandas as pd
import json

def draw_rectangles(image_name, csv_path, image_folder, segmented_folder, N):
    # Ler o arquivo CSV
    df = pd.read_csv(csv_path)
    # Filtrar o DataFrame pelo nome da imagem
    df = df[df['image_filename'] == image_name]
    # Agrupar as linhas por image_id
    grouped = df.groupby('image_id')
    for image_id, group in grouped:
        # Carregar a imagem correspondente
        img = cv2.imread(f'{image_folder}/{group["image_filename"].iloc[0]}')
        cell_data = []  # Para armazenar os dados das células

        for idx, row in group.iterrows():
            # Obter as coordenadas do núcleo da célula
            x, y = row['nucleus_x'], row['nucleus_y']
            # Desenhar um retângulo colorido em torno da célula
            cv2.rectangle(img, (x - N // 2, y - N // 2), (x + N // 2, y + N // 2), (0, 255, 0), 2)

            # Adicionar os dados da célula à lista
            cell_data.append({
                'x': x,
                'y': y,
                'label': idx  # Usar o índice como rótulo
            })

            # Escrever o rótulo diretamente na imagem
            cv2.putText(img, str(idx), (x - N // 2, y - N // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Salvar a imagem com os retângulos desenhados
        cv2.imwrite(f'{segmented_folder}/{image_name}_cells_segmented.png', img)

        # Imprimir os dados no console em formato JSON
        cell_data_json = json.dumps(cell_data, indent=4)
        print(cell_data_json)

# Chame a função para processar a imagem
draw_rectangles('2cefdbf695da71852337ae3557ccdd38.png', '../../data/classifications.csv', '../../fodas', 'segmented-gui-cells', 100)
