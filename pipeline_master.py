import os
import subprocess

#Paso 1: Redimensionar imágenes
print("✅ Paso 1: Redimensionando imágenes...")
subprocess.run(["python", "resize_images.py"], check=True)

#Paso 2: Entrenar modelo
print("\n✅ Paso 2: Entrenando modelo...")
subprocess.run(["python", "train_model.py"], check=True)

#Paso 3: Convertir modelo a TFLite
print("\n✅ Paso 3: Convirtiendo modelo a TFLite...")
subprocess.run(["python", "convertirH5.py"], check=True)

#Paso 4: Clasificar carpeta específica
folder_to_classify = input("\n📂 Ingresa la ruta de la carpeta con imágenes a clasificar: ").strip()
if os.path.isdir(folder_to_classify):
    print("✅ Paso 4: Clasificando imágenes...")
    subprocess.run(["python", "classify_tflite.py", "--input", folder_to_classify], check=True)
else:
    print("❌ La carpeta no existe o no es válida.")

print("\n🎉 ¡Pipeline completo!")
