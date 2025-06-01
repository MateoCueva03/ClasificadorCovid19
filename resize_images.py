import os
import cv2

# Rutas de entrada (ajusta a tu disco D:)
input_folder = r'D:\Documentos\UISRAEL\concursoInnovacion\COVID-19_Radiography_Dataset'
output_folder = r'D:\Documentos\UISRAEL\concursoInnovacion\resized_images'

# Clases que procesaremos
CLASSES = ['COVID', 'Normal', 'Viral Pneumonia']

# Crea carpetas de salida si no existen
for cls in CLASSES:
    os.makedirs(os.path.join(output_folder, cls), exist_ok=True)

# Recorre cada clase y procesa las imágenes
for cls in CLASSES:
    input_path = os.path.join(input_folder, cls, 'images')  # Añadido subcarpeta images
    output_path = os.path.join(output_folder, cls)
    images = os.listdir(input_path)

    for img_name in images:
        img_path = os.path.join(input_path, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            # Redimensiona a 150x150 px
            resized_img = cv2.resize(img, (150, 150))
            cv2.imwrite(os.path.join(output_path, img_name), resized_img)
            print(f'Redimensionada: {img_name}')
        else:
            print(f'No se pudo leer: {img_path}')

print('🎉 ¡Proceso completo! Las imágenes redimensionadas están en:', output_folder)
