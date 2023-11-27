import time
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

# TODO: Tentar arrumar

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_generator):
        self.train_accuracy = []
        self.test_generator = test_generator
        self.test_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_accuracy.append(logs.get('accuracy'))

        # Calculate test accuracy
        test_predictions = self.model.predict(self.test_generator)
        test_predicted_classes = np.argmax(test_predictions, axis=1)
        test_accuracy = accuracy_score(self.test_generator.classes, test_predicted_classes)
        self.test_accuracies.append(test_accuracy)
        print(f'Test Accuracy after Epoch {epoch + 1}: {test_accuracy}')

train_data_dir = '../separate-dataset/train'
test_data_dir = '../separate-dataset/test'

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# Freeze all layers except the last few
for layer in base_model.layers:
    layer.trainable = False

layer = base_model.output
layer = GlobalAveragePooling2D()(layer)
layer = Dense(256, activation='relu')(layer)
prediction = Dense(6, activation='softmax')(layer)

model = Model(inputs=base_model.input, outputs=prediction)

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

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
    class_mode='categorical',
    shuffle=True
)

test_data_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_data_gen.flow_from_directory(
    test_data_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

custom_callback = CustomCallback(test_generator)  # Used for plot graphs
start_time = time.time()
model.fit(train_generator, epochs=50, callbacks=[custom_callback])
end_time = time.time()

execution_time = end_time - start_time
print('Tempo de execução: ', execution_time)

model.save('ai_models/my_model_multiclass_resnet.h5')

test_data_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_data_gen.flow_from_directory(
    test_data_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_labels = test_generator.classes

predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Calculate metrics
accuracy = accuracy_score(test_labels, predicted_classes)
precision = precision_score(test_labels, predicted_classes, average='weighted')
recall = recall_score(test_labels, predicted_classes, average='weighted')
f1 = f1_score(test_labels, predicted_classes, average='weighted')

cm = confusion_matrix(test_labels, predicted_classes)

# Plot and save learning curve
plt.plot(custom_callback.train_accuracy, label='Train Accuracy')
plt.plot(custom_callback.test_accuracies, label='Test Accuracy')
plt.title('Training and Test Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('graphs/accuracy_curve_multiclass.png')
plt.show()

# Plot and save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('graphs/confusion_matrix_multiclass.png')
plt.show()

print(f'Classes: {test_labels}')
print('Matriz de confusão: ', cm)
print('Acurácia: ', accuracy)
print('Precisão: ', precision)
print('Recall: ', recall)
print('F1 score: ', f1)
