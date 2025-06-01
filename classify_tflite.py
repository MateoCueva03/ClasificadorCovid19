import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Rutas y configuración
MODEL_PATH = 'best_model.tflite'  # modelo TFLite generado
INPUT_SIZE = (150, 150)            # mismo tamaño usado en entrenamiento
CLASS_NAMES = ['COVID-19', 'Normal', 'Viral Pneumonia']  # orden según clases entrenadas

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_path, input_size=INPUT_SIZE):
    """Carga imagen, cambia tamaño y normaliza."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(input_size)
        img_array = np.array(img, dtype=np.float32) / 255.0  # normalizar
        img_array = np.expand_dims(img_array, axis=0)  # añadir batch dimension
        return img_array
    except Exception as e:
        print(f"Error al cargar/preprocesar imagen {image_path}: {e}")
        return None

def predict(interpreter, image):
    """Ejecuta inferencia con TFLite y devuelve clase y probabilidad."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Asignar entrada
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Obtener resultados
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probs = np.squeeze(output_data)
    class_idx = np.argmax(probs)
    confidence = probs[class_idx]
    return class_idx, confidence

def classify_folder(interpreter, folder_path):
    """Clasifica todas las imágenes en una carpeta."""
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(folder_path, filename)
            img = preprocess_image(full_path)
            if img is not None:
                class_idx, confidence = predict(interpreter, img)
                class_name = CLASS_NAMES[class_idx]
                results.append((filename, class_name, confidence))
                print(f"✔ {filename} --> {class_name} ({confidence:.2f})")
    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clasificador TFLite para COVID-19, Neumonía viral y Normal")
    parser.add_argument("--input", type=str, required=True, help="Carpeta con imágenes a clasificar")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Ruta del modelo TFLite")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Modelo TFLite no encontrado en {args.model}. Primero genera el modelo con el script de conversión.")
        exit(1)

    if not os.path.isdir(args.input):
        print(f"Carpeta de entrada no encontrada o inválida: {args.input}")
        exit(1)

    print("Cargando modelo TFLite...")
    interpreter = load_tflite_model(args.model)

    print(f"Clasificando imágenes en carpeta: {args.input}")
    resultados = classify_folder(interpreter, args.input)

    # Guardar resultados a archivo CSV
    import csv
    output_csv = "classification_results.csv"
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Archivo', 'Clase_predicha', 'Confianza'])
        writer.writerows(resultados)

    print(f"✅ Clasificación completada. Resultados guardados en {output_csv}")
