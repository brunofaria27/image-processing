import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

def load_and_preprocess_image(image):
    img_resized = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def classify_image_binary(model, image):
    img_array = load_and_preprocess_image(image)
    prediction = model.predict(img_array)[0, 0]
    class_name = "Positive" if prediction > 0.5 else "Negative"
    confidence = max(prediction, 1 - prediction) * 100
    return class_name, confidence

def classify_image_multiclass(model, image):
    pass

def process_resnet_binary(segmented_images):
    model_path = 'train_codes/ai_models/my_model_binary_resnet.h5'
    model = load_model(model_path)

    results = []
    for image in segmented_images:
        class_name, confidence = classify_image_binary(model, image)
        results.append([class_name, confidence])

    columns = ['Predicted Class', 'Confidence']
    results_df = pd.DataFrame(results, columns=columns)
    return results_df

def process_resnet_multiclass(segmented_images):
    model_path = 'train_codes/ai_models/my_model_binary_multiclass.h5'
    model = load_model(model_path)

    results = []
    for image in segmented_images:
        class_name, confidence = classify_image_multiclass(model, image)
        results.append([class_name, confidence])

    columns = ['Predicted Class', 'Confidence']
    results_df = pd.DataFrame(results, columns=columns)
    return results_df