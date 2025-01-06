
import os
import csv
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

def preprocess_image(image_path, output_path):
    # Abre la imagen
    img = Image.open(image_path)
    img = img.convert("RGB")
    
    # Aumenta el contraste de los elementos que no son [0, 0, 0]
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(3)  # Ajusta el factor de contraste según sea necesario
    
    # Aplica un filtro de suavizado para eliminar ruido
    img = img.filter(ImageFilter.SMOOTH)
    
    # Convierte la imagen a escala de grises
    img = ImageOps.grayscale(img)
    
    # Aplica una dilatación más fuerte (más veces)
    for _ in range(14):  # Ajusta el número de iteraciones según sea necesario
        img = img.filter(ImageFilter.MaxFilter(3))  # Dilatación
    
    # Aplica una erosión (filtro de serie)
    for _ in range(5):  # Ajusta el número de iteraciones según sea necesario
        img = img.filter(ImageFilter.MinFilter(3))  # Erosión
    
    # Binariza la imagen
    threshold = 128
    img = img.point(lambda p: p > threshold and 255)
    
    # Guarda la imagen preprocesada
    img.save(output_path)
    
    return img

def analyze_image(img, threshold=30):
    pixels = img.load()
    
    # Inicializa variables
    objects = []
    visited = set()
    
    # Función auxiliar para realizar DFS y encontrar propiedades del objeto
    def dfs(x, y):
        stack = [(x, y)]
        area = 0
        perimeter = 0
        min_x, max_x = x, x
        min_y, max_y = y, y
        intensity_sum = 0
        
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            area += 1
            intensity_sum += pixels[cx, cy]
            
            min_x = min(min_x, cx)
            max_x = max(max_x, cx)
            min_y = min(min_y, cy)
            max_y = max(max_y, cy)
            
            for nx, ny in [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]:
                if 0 <= nx < img.width and 0 <= ny < img.height:
                    if pixels[nx, ny] > threshold and (nx, ny) not in visited:
                        stack.append((nx, ny))
                    elif pixels[nx, ny] <= threshold:
                        perimeter += 1
        
        angle_of_inclination = (max_y - min_y) / (max_x - min_x + 1e-5)
        texture = area / (perimeter + 1e-5)
        intensity_avg = intensity_sum // area
        
        return {
            "area": area,
            "perimeter": perimeter,
            "angle_of_inclination": angle_of_inclination,
            "texture": texture,
            "intensity": intensity_avg
        }
    
    # Itera sobre cada píxel para encontrar objetos
    for x in range(img.width):
        for y in range(img.height):
            if pixels[x, y] > threshold and (x, y) not in visited:
                obj_props = dfs(x, y)
                objects.append(obj_props)
    
    return objects

def process_images(folder_path, output_folder_path, threshold=30):  # Cambiar el valor predeterminado aquí si es necesario
    results = []

    for image_name in os.listdir(folder_path):
        if image_name.endswith(".png") or image_name.endswith(".jpg"):
            image_path = os.path.join(folder_path, image_name)
            
            # Define la ruta de salida para la imagen preprocesada
            output_image_path = os.path.join(output_folder_path, f"preprocessed_{image_name}")
            
            # Preprocesa la imagen y guarda una copia de la imagen modificada
            img = preprocess_image(image_path, output_image_path)
            
            # Analiza la imagen preprocesada
            objects = analyze_image(img, threshold=threshold)

            for idx, obj in enumerate(objects, start=1):
                results.append({
                    "image_number": image_name,
                    "object_number": idx,
                    "area": obj["area"],
                    "perimeter": obj["perimeter"],
                    "angle_of_inclination": obj["angle_of_inclination"],
                    "texture": obj["texture"],
                    "intensity": obj["intensity"]
                })

    return results

def save_to_csv(results, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_number", "object_number", "area", "perimeter", "angle_of_inclination", "texture", "intensity"])

        for result in results:
            writer.writerow([
                result["image_number"],
                result["object_number"],
                result["area"],
                result["perimeter"],
                result["angle_of_inclination"],
                result["texture"],
                result["intensity"]
            ])

# Define la ruta de la carpeta que contiene las imágenes y el nombre del archivo CSV de salida
folder_path = "imagenes"
output_folder_path = "./finaldataset"
output_file = "imagenes.csv"

# Crea la carpeta de salida si no existe
os.makedirs(output_folder_path, exist_ok=True)

# Procesa las imágenes y guarda los resultados en un archivo CSV
threshold_value = 30  # Ajusta este valor según la intensidad del fondo
results = process_images(folder_path, output_folder_path, threshold=threshold_value)
save_to_csv(results, output_file)

print(f"Análisis completo. Resultados guardados en {output_file}. Las imágenes preprocesadas se han guardado en {output_folder_path}.")


