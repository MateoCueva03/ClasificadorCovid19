import tensorflow as tf

# Carga modelo Keras entrenado
model = tf.keras.models.load_model('best_model.h5')

# Convierte a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guarda modelo TFLite
with open('best_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("✅ Modelo convertido y guardado como best_model.tflite")
