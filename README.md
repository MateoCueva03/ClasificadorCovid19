# ClasificadorCovid19

## Descripción del Proyecto
Este proyecto desarrolla un clasificador automático de imágenes médicas para detectar COVID-19, pulmón normal y neumonía viral a partir de radiografías de tórax. La solución está basada en un modelo de aprendizaje profundo utilizando la arquitectura MobileNetV2 preentrenada en ImageNet. Se implementa un pipeline completo que incluye:

- Preprocesamiento (redimensionado de imágenes)

- Entrenamiento del modelo con aumentación de datos

- Conversión a formato TensorFlow Lite para inferencia eficiente

- Clasificación de imágenes usando el modelo TFLite

- Interfaz gráfica para pruebas rápidas

El proyecto se enfoca en cumplir con los estándares de funcionalidad, eficiencia, mantenibilidad y seguridad para el procesamiento y clasificación de imágenes médicas, lo cual es crucial para sistemas de apoyo clínico.

### Contenido del Repositorio
resize_images.py : Script para redimensionar y organizar las imágenes del dataset.

train_model.py : Entrena el modelo MobileNetV2 con los datos preprocesados.

convertirH5.py : Convierte el modelo Keras (.h5) a TensorFlow Lite (.tflite).

classify_tflite.py : Script para clasificar imágenes con el modelo TFLite.

pipeline_master.py : Pipeline que automatiza la ejecución de los pasos anteriores.

app_classifier.py : Interfaz gráfica en Tkinter para seleccionar imágenes y obtener predicciones.

README.md : Documentación del proyecto (este archivo).

classification_results.csv : Archivo de resultados generado tras clasificar un conjunto de imágenes.

### Requisitos
- Python 3.7 o superior

- TensorFlow 2.x

- OpenCV (opencv-python)

- Pillow

- Matplotlib

- NumPy

Instalación rápida de dependencias:

```bash
pip install tensorflow opencv-python pillow matplotlib numpy
```

Uso
1. Preprocesamiento: Redimensionar imágenes
```bash
python resize_images.py
```

Redimensiona todas las imágenes a 150x150 píxeles y las organiza en carpetas por clase.

2. Entrenamiento del modelo
```bash
python train_model.py
```
Entrena el modelo MobileNetV2 usando los datos redimensionados. Se generan gráficos de precisión y se guarda el mejor modelo (best_model.h5).

3. Conversión a TensorFlow Lite

```bash
python convertirH5.py
```
Convierte el modelo Keras al formato .tflite para inferencias rápidas en dispositivos con recursos limitados.

5. Clasificación de imágenes con TFLite
```bash
python classify_tflite.py --input ruta/a/carpeta_con_imagenes
```
Clasifica todas las imágenes en la carpeta indicada y guarda los resultados en classification_results.csv.

6. Ejecución automática con Pipeline
```bash
python pipeline_master.py
```
Ejecuta automáticamente todos los pasos anteriores de forma secuencial.

7. Interfaz gráfica para clasificación (opcional)
```bash
python app_classifier.py
```
Permite seleccionar imágenes y obtener predicciones en una ventana sencilla.

### Resultados esperados
Modelos entrenados capaces de clasificar correctamente radiografías en las tres categorías.

Archivo CSV con resultados detallados de clasificación.

Gráficos que muestran evolución de la precisión durante el entrenamiento.

Modelo TFLite eficiente para inferencias rápidas.

Atributos de Calidad
Atributo	Implementación en el proyecto
Funcionalidad	Clasificación precisa con MobileNetV2
Eficiencia	Uso de modelo TFLite para reducir tiempo de inferencia
Mantenibilidad	Código modular y documentado en scripts separados
Seguridad	No se exponen datos sensibles, solo imágenes de prueba

### Comparativa Tecnológica: Hadoop vs. Dask vs. Pandas

| Tecnología | Ventajas                                                    | Desventajas                                 |
| ---------- | ----------------------------------------------------------- | ------------------------------------------- |
| Hadoop     | Escalabilidad masiva, manejo de big data                    | Complejidad, sobrecarga para datos pequeños |
| Dask       | Paralelización en Python, fácil integración                 | Menos maduro que Hadoop, requiere tuning    |
| Pandas     | Muy fácil de usar, excelente para datos medianos y pequeños | No escala bien para big data                |


Este proyecto usa principalmente Pandas para la manipulación sencilla y TensorFlow para ML, considerando el tamaño manejable del dataset.

### Video Explicativo
