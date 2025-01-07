import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import cv2

class ImageSimilarityFinder:
    def __init__(self, root):
        self.root = root
        self.root.title("Buscador de Imágenes Similares")
        
        # Cargar datos
        try:
            self.data = pd.read_csv("imagenes_clustered_normalized.csv")
            self.data.columns = self.data.columns.str.strip()
        except FileNotFoundError:
            print("El archivo 'imagenes_clustered_normalized.csv' no se encontró.")
            exit()
            
        # Crear interfaz
        self.create_widgets()
        
    def create_widgets(self):
        # Botón para cargar imagen
        self.btn_cargar = tk.Button(self.root, text="Cargar Imagen", command=self.cargar_imagen)
        self.btn_cargar.pack(pady=10)
        
        # Label para mostrar la imagen cargada
        self.lbl_imagen_original = tk.Label(self.root)
        self.lbl_imagen_original.pack(pady=10)
        
        # Frame para imágenes similares
        self.frame_similares = tk.Frame(self.root)
        self.frame_similares.pack(pady=10)
        
    def cargar_imagen(self):
        # Abrir diálogo para seleccionar archivo
        ruta_archivo = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        
        if ruta_archivo:
            # Procesar imagen cargada
            try:
                # Mostrar imagen original
                img = Image.open(ruta_archivo)
                img.thumbnail((150, 150))
                img_tk = ImageTk.PhotoImage(img)
                self.lbl_imagen_original.configure(image=img_tk)
                self.lbl_imagen_original.image = img_tk
                
                # Extraer características de la imagen cargada
                caracteristicas = self.extraer_caracteristicas(ruta_archivo)
                
                # Obtener el nombre completo del archivo de entrada
                nombre_archivo_entrada = os.path.basename(ruta_archivo)
                
                # Encontrar imágenes similares
                self.mostrar_similares(caracteristicas, nombre_archivo_entrada)
                
            except Exception as e:
                tk.messagebox.showerror("Error", f"Error al procesar la imagen: {str(e)}")
    
    def extraer_caracteristicas(self, ruta_imagen):
        # Cargar y procesar imagen
        img = cv2.imread(ruta_imagen)
        img = cv2.resize(img, (224, 224))  # Redimensionar a tamaño estándar
        img = img.flatten() / 255.0  # Normalizar
        return img
    
    def mostrar_similares(self, caracteristicas_query, nombre_archivo_entrada):
        # Limpiar frame de imágenes similares
        for widget in self.frame_similares.winfo_children():
            widget.destroy()
            
        # Obtener características de todas las imágenes en el dataset
        imagenes_similares = []
        for idx, row in self.data.iterrows():
            try:
                ruta_imagen = os.path.join("imagenes", str(row['clave']))
                
                # Obtener el nombre del archivo actual
                nombre_archivo_actual = os.path.basename(ruta_imagen)
                
                # Omitir la imagen si tiene el mismo nombre que la de entrada
                if nombre_archivo_actual == nombre_archivo_entrada:
                    continue
                    
                caracteristicas = self.extraer_caracteristicas(ruta_imagen)
                
                # Calcular distancia
                distancia = euclidean_distances([caracteristicas_query], [caracteristicas])[0][0]
                imagenes_similares.append((distancia, ruta_imagen))
            except:
                continue
        
        # Ordenar por similitud y mostrar las top 5
        imagenes_similares.sort(key=lambda x: x[0])
        for distancia, ruta in imagenes_similares[:5]:
            try:
                img = Image.open(ruta)
                img.thumbnail((100, 100))
                img_tk = ImageTk.PhotoImage(img)
                lbl = tk.Label(self.frame_similares, image=img_tk)
                lbl.image = img_tk
                lbl.pack(side=tk.LEFT, padx=5)
                
                # Mostrar puntaje de similitud
                tk.Label(self.frame_similares, 
                        text=f"Similitud: {(1 - distancia/100):.2%}").pack(side=tk.LEFT)
            except Exception as e:
                print(f"Error al mostrar imagen similar {ruta}: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSimilarityFinder(root)
    root.mainloop()