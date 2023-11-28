import csv
import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

def load_and_preprocess_image(image):
    img = cv2.resize(image, (100, 100))
    img = img / 255.0  # Normalize the image
    img_array = np.expand_dims(img, axis=0)
    return img_array

def classify_image_binary(model, image_path):
    img_array = load_and_preprocess_image(image_path)
    prediction = model.predict(img_array)[0, 0]
    class_name = "Positive" if prediction > 0.5 else "Negative"
    confidence = max(prediction, 1 - prediction) * 100
    return class_name, confidence

def classify_image_multiclass(model, image):
    img_array = load_and_preprocess_image(image)
    predictions = model.predict(img_array)[0]
    class_index = np.argmax(predictions)
    confidence = predictions[class_index] * 100
    
    # Map class indices to class names
    class_names = ['ASC-H', 'ASC-US', 'HSIL', 'LSIL', 'Negative', 'SCC']
    class_name = class_names[class_index]

    return class_name, confidence

def process_resnet_binary(segmented_images, ids):
    model_path = 'train_codes/ai_models/my_model_binary_resnet.h5'
    model = load_model(model_path)
    true_classes = search_true_classes(ids)

    results = []
    for image, true_class in zip(segmented_images, true_classes):
        class_name, confidence = classify_image_binary(model, image)
        results.append([true_class, class_name, confidence])

    columns = ['True Class', 'Predicted Class', 'Confidence']
    results_df = pd.DataFrame(results, columns=columns)
    return results_df

def process_resnet_multiclass(segmented_images, ids):
    model_path = 'train_codes/ai_models/my_model_multiclass_resnet.h5'
    model = load_model(model_path)
    true_classes = search_true_classes(ids)
    
    results = []
    for img, true_class in zip(segmented_images, true_classes):
        class_name, confidence = classify_image_multiclass(model, img)
        results.append([true_class, class_name, confidence])

    columns = ['True Class', 'Predicted Class', 'Confidence']
    results_df = pd.DataFrame(results, columns=columns)
    return results_df

def search_true_classes(ids):
    true_classes = []

    with open('data/classifications.csv', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        
        for row in csv_reader:
            for id in ids:
                if row['cell_id'] == str(id):
                    true_classes.append(row['bethesda_system'])
    return true_classes