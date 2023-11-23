import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

# Function to create a ResNet50 model
def create_resnet_model(img_size, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    for layer in base_model.layers:
        layer.trainable = False

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Function to train the model
def train_model(model, train_generator, validation_generator, epochs, model_checkpoint):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[model_checkpoint]
    )

    return history

# Set the path to your data directories
train_data_dir = '../separate-dataset/train'
test_data_dir = '../separate-dataset/test'

# Image size that ResNet50 expects
img_size = (100, 100)

# Batch size
batch_size = 32

# Number of classes
num_classes = 6

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Create ResNet50 model
model = create_resnet_model(img_size, num_classes)

# Create model checkpoint to save the best weights during training
model_checkpoint = ModelCheckpoint('trained_model/ResNet50_best.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min', verbose=1)

# Train the model
history = train_model(model, train_generator, test_generator, epochs=10, model_checkpoint=model_checkpoint)

# Save the entire model
model.save('trained_model/ResNet50.h5')

# Plot the training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.jpg')

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(model, test_generator):
    # Get the true labels
    true_labels = test_generator.classes

    # Get class indices
    class_indices = test_generator.class_indices

    # Make predictions
    predictions = model.predict(test_generator)
    predicted_labels = np.argmax(predictions, axis=1)

    # Plot the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_indices.keys(), yticklabels=class_indices.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('MATRIX.jpg')

    # Print classification report
    print("Classification Report:\n", classification_report(true_labels, predicted_labels, target_names=class_indices.keys()))

plot_confusion_matrix(model, test_generator)