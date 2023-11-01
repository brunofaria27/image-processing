import cv2
import pandas as pd
import json

def draw_rectangles(image_name, csv_path, image_folder, N):
    df = pd.read_csv(csv_path)
    df = df[df['image_filename'] == image_name]
    grouped = df.groupby('image_id')
    
    for image_id, group in grouped:
        img = cv2.cvtColor(cv2.imread(f'{image_folder}/{image_name}'), cv2.COLOR_BGR2RGB)
        cell_data = []

        for idx, row in group.iterrows():
            x, y = row['nucleus_x'], row['nucleus_y']
            cv2.rectangle(img, (x - N // 2, y - N // 2), (x + N // 2, y + N // 2), (255, 0, 0), 2)

            cell_data.append({
                'x': x,
                'y': y,
                'label': idx
            })

            cv2.putText(img, str(idx), (x - N // 2, y - N // 2 - 10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255, 0, 0), 2)
        return img, cell_data
