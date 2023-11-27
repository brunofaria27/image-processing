import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
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
        test_predicted_classes = np.where(test_predictions > 0.5, 1, 0)
        test_accuracy = accuracy_score(self.test_generator.classes, test_predicted_classes)
        self.test_accuracies.append(test_accuracy)
        print(f'Test Accuracy after Epoch {epoch + 1}: {test_accuracy}')

train_data_dir = '../separate-bin-dataset/train'
test_data_dir = '../separate-bin-dataset/test'

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

for nos in base_model.layers:
    nos.trainable = False

layer = base_model.output
layer = GlobalAveragePooling2D()(layer)
layer = Dense(256, activation='relu')(layer)
prediction = Dense(1, activation='sigmoid')(layer)

model = Model(inputs=base_model.input, outputs=prediction)

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

test_data_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_data_gen.flow_from_directory(
    test_data_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

custom_callback = CustomCallback(test_generator)  # Used for plot graphs
start_time = time.time()
model.fit(train_generator, epochs=100, callbacks=[custom_callback])
end_time = time.time()

execution_time = end_time - start_time
print('Tempo de execução: ', execution_time)

model.save('ai_models/my_model_binary_resnet.h5')

test_labels = test_generator.classes
class_names = list(test_generator.class_indices.keys())
print(class_names)

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

# Plot and save learning curve
plt.plot(custom_callback.train_accuracy, label='Train Accuracy')
plt.plot(custom_callback.test_accuracies, label='Test Accuracy')
plt.title('Training and Test Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('graphs/accuracy_curve_binary.png')
plt.show()

# Plot and save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('graphs/confusion_matrix_binary.png')
plt.show()

print(f'Classes: {test_labels}')
print('Matriz de confusão: ', cm)
print('Acurácia: ', accuracy)
print('Precisão: ', precision)
print('Recall: ', recall)
print('F1 score: ', f1)
print('Sensibilidade: ', sensitivity)
print('Especificidade: ', specificity)
