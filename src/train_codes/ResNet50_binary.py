import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

train_data_dir = '../separate-bin-dataset/train'
test_data_dir = '../separate-bin-dataset/test'
img_width, img_height = 100, 100
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# 1. Primeira Rede Binária
model_binary = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

for layer in model_binary.layers:
    layer.trainable = False

x = model_binary.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions_binary = Dense(1, activation='sigmoid')(x)

model_binary = Model(inputs=model_binary.input, outputs=predictions_binary)
model_binary.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento
history_binary = model_binary.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size)

# Avaliação
y_pred_binary = model_binary.predict(test_generator)
y_pred_binary = (y_pred_binary > 0.5).astype(int)
acc_binary = accuracy_score(test_generator.classes, y_pred_binary)
conf_matrix_binary = confusion_matrix(test_generator.classes, y_pred_binary)

# Plote os gráficos de aprendizado
plt.plot(history_binary.history['accuracy'], label='Training Accuracy')
plt.plot(history_binary.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy1.jpg')

# 2. Segunda Rede usando ResNet50 com fine-tuning
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

for layer in base_model.layers:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model_finetuned = Model(inputs=base_model.input, outputs=predictions)
model_finetuned.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento
history_finetuned = model_finetuned.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size)

# Avaliação
y_pred_finetuned = model_finetuned.predict(test_generator)
y_pred_finetuned = (y_pred_finetuned > 0.5).astype(int)
acc_finetuned = accuracy_score(test_generator.classes, y_pred_finetuned)
conf_matrix_finetuned = confusion_matrix(test_generator.classes, y_pred_finetuned)

# Plote os gráficos de aprendizado
plt.plot(history_finetuned.history['accuracy'], label='Training Accuracy')
plt.plot(history_finetuned.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy2.jpg')

# Imprima acurácia e matrizes de confusão
print("Binary Model Accuracy:", acc_binary)
print("Binary Model Confusion Matrix:")
print(conf_matrix_binary)


print("\nFine-tuned Model Accuracy:", acc_finetuned)
print("Fine-tuned Model Confusion Matrix:")
print(conf_matrix_finetuned)
