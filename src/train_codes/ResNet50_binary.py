import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import confusion_matrix

train_data_dir = '../separate-bin-dataset/train'
test_data_dir = '../separate-bin-dataset/test'

base_model = ResNet50(weights='imagenet', include_top=False)

layer = base_model.output
layer = GlobalAveragePooling2D()(layer)
prediction = Dense(1, activation='sigmoid')(layer)

model = Model(inputs=base_model.input, outputs=prediction)
for nos in base_model.layers:
    nos.trainable = False

model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

train_data = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_data.flow_from_directory(
    train_data_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary'
)

start_time = time.time()
model.fit(train_generator, epochs=150)
end_time = time.time()

execution_time = end_time - start_time
print('Tempo de execução: ', execution_time)

model.save('ai_models/my_model_binary_resnet.h5')

test_data_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_data_gen.flow_from_directory(
    test_data_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

test_labels = test_generator.classes

predictions = model.predict(test_generator)
predicted_classes = np.where(predictions > 0.5, 1, 0)

# Calculate metrics
accuracy = accuracy_score(test_labels, predicted_classes)
precision = precision_score(test_labels, predicted_classes)
recall = recall_score(test_labels, predicted_classes)
f1 = f1_score(test_labels, predicted_classes)

cm = confusion_matrix(test_labels, predicted_classes)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print('Matriz de confusão: ', cm)
print('Acurácia: ', accuracy)
print('Precisão: ', precision)
print('Recall: ', recall)
print('F1 score: ', f1)
print('Sensibilidade: ', sensitivity)
print('Especificidade: ', specificity)
