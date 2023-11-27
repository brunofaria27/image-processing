import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def classify_image(model, image_path):
    img_array = load_and_preprocess_image(image_path)
    prediction = model.predict(img_array)[0, 0]
    class_name = "Positive" if prediction > 0.5 else "Negative"
    confidence = max(prediction, 1 - prediction) * 100
    return [(None, class_name, confidence)]

def main():
    model_path = 'ai_models/my_model_binary_resnet.h5'
    model = load_model(model_path)

    image_path = '../images-processed/Negative for intraepithelial lesion/823.png'
    predictions = classify_image(model, image_path)

    print("Predictions:")
    for _, class_name, confidence in predictions:
        print(f"{class_name}: {confidence:.2f}%")

if __name__ == "__main__":
    main()
