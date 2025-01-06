import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
import os

# carga CSV
try:
    data = pd.read_csv("clustering_file_name.csv")
    data.columns = data.columns.str.strip()  
    print(data.head())  
    print(data.info())  

except FileNotFoundError:
    print("El archivo 'clustering_file_name.csv' no se encontró.")
    exit()
except Exception as e:
    print(f"Error al cargar el archivo CSV: {e}")
    exit()

# Verificacion
if 'Cluster_Normalized' not in data.columns or 'Cluster_Normalized' not in data.columns or 'clave' not in data.columns:
    print("El archivo CSV no contiene las columnas necesarias: 'Cluster_Normalized', 'Cluster_Normalized', 'clave'.")
    exit()

def mostrar_imagenes():
    clase_seleccionada = combo_clase.get()

  
    filtro = data[
        (data['Cluster_Normalized'] == int(clase_seleccionada))
]
    print(data['Cluster_Normalized'].unique())  


    # Limpiar imágenes anteriores
    for widget in frame_imagenes.winfo_children():
        widget.destroy()

    # Mostrar imágenes filtradas
    for _, fila in filtro.iterrows():
        # Verificar ruta de la imagen
        ruta_imagen = os.path.join("path/to/your/images/folder", str(fila['clave'])) 
        print(f"Intentando cargar: {ruta_imagen}")

        try:
            img = Image.open(ruta_imagen)
            img.thumbnail((100, 100))  # Redimensionar
            img_tk = ImageTk.PhotoImage(img)
            lbl_img = tk.Label(frame_imagenes, image=img_tk)
            lbl_img.image = img_tk  #Borrar cuando vuelva a pedir imagenes
            lbl_img.pack(side=tk.LEFT, padx=5, pady=5)
        except FileNotFoundError:
            print(f"No se encontró la imagen: {ruta_imagen}")
        except Exception as e:
            print(f"Error al cargar la imagen '{ruta_imagen}': {e}")
            
            
# Ventana principal
root = tk.Tk()
root.title("Visor de Imágenes por Clases")


# Dropdown para clases
ttk.Label(root, text="Selecciona una Clase:").pack()
combo_clase = ttk.Combobox(root, values=data['Cluster_Normalized'].unique().tolist())
combo_clase.pack()

# Botón para mostrar imagenes
btn_mostrar = tk.Button(root, text="Mostrar Imágenes", command=mostrar_imagenes)
btn_mostrar.pack(pady=10)

# Frame para mostrar imagenes
frame_imagenes = tk.Frame(root)
frame_imagenes.pack()

# Ejecutar la aplicación
root.mainloop()
