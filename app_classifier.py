import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import random

class CovidClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de COVID-19")
        self.root.geometry("500x600")

        self.label_title = tk.Label(root, text="Clasificador de COVID-19", font=("Arial", 20, "bold"))
        self.label_title.pack(pady=10)

        self.btn_select = tk.Button(root, text="Seleccionar Imagen", command=self.select_image)
        self.btn_select.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.btn_predict = tk.Button(root, text="Predecir", command=self.simulate_prediction, state=tk.DISABLED)
        self.btn_predict.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png")])
        if file_path:
            # Cargar y mostrar la imagen
            img = Image.open(file_path)
            img = img.resize((300, 300))
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo

            # Habilitar botón de predicción
            self.btn_predict.config(state=tk.NORMAL)

    def simulate_prediction(self):
        # Simulación de predicción aleatoria
        classes = ["COVID-19", "Normal", "Neumonía Viral"]
        predicted_class = random.choice(classes)
        confidence = round(random.uniform(0.5, 1.0), 2)

        self.result_label.config(text=f"Predicción: {predicted_class}\nConfianza: {confidence * 100:.0f}%")
        messagebox.showinfo("Resultado", f"La predicción es:\n{predicted_class}\nConfianza: {confidence * 100:.0f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = CovidClassifierApp(root)
    root.mainloop()

