import cv2
import pandas as pd
import json

def draw_rectangles(image_name, csv_path, image_folder, segmented_folder, N):
    df = pd.read_csv(csv_path)
    df = df[df['image_filename'] == image_name]
    grouped = df.groupby('image_id')
    
    for image_id, group in grouped:
        img = cv2.imread(f'{image_folder}/{group["image_filename"].iloc[0]}')
        cell_data = []

        for idx, row in group.iterrows():
            x, y = row['nucleus_x'], row['nucleus_y']
            cv2.rectangle(img, (x - N // 2, y - N // 2), (x + N // 2, y + N // 2), (255, 0, 0), 2)

            cell_data.append({
                'x': x,
                'y': y,
                'label': idx  # Usar o índice como rótulo
            })

            cv2.putText(img, str(idx), (x - N // 2, y - N // 2 - 10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.75, (255, 0, 0), 2)

        cv2.imwrite(f'{segmented_folder}/{image_name}_cells_segmented.png', img)

        return img

# draw_rectangles('2cefdbf695da71852337ae3557ccdd38.png', '../data/classifications.csv', '../images', 'segmented-gui-cells', 100)
