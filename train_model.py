import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

#Rutas
dataset_dir = r'D:\Documentos\UISRAEL\concursoInnovacion\resized_images'
batch_size = 32
img_height, img_width = 150, 150

#Preprocesamiento y aumentación solo en entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

#Dataset
train_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

valid_gen = valid_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

#Modelo: MobileNetV2 con pesos preentrenados en ImageNet
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Congelamos para ahorrar recursos

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

#Checkpoint para guardar el mejor modelo
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

#Entrenamiento
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=valid_gen,
    callbacks=[checkpoint],
    workers=4  # paralelismo en CPU
)

#Gráfico de accuracy
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.legend()
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.title('Precisión del Modelo')
plt.show()

print('✅ ¡Entrenamiento completado! Modelo guardado como best_model.h5')
