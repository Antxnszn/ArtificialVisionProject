import pandas as pd
from sklearn.cluster import KMeans

# Cargar el dataset
dataset = pd.read_csv("imagenes.csv")

# Seleccionar las características para el clustering
X_dataset = dataset[['area', 'perimeter', 'angle_of_inclination', 'texture', 'intensity']]

# Inicializar el modelo K-means con 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)

# Ajustar el modelo a los datos
kmeans.fit(X_dataset)

# Predecir los clusters
clusters = kmeans.predict(X_dataset)

# Agregar los clusters al dataset original
dataset['Cluster'] = clusters

# Guardar los resultados en un nuevo archivo CSV
output_path = "clusteredImages.csv"
dataset.to_csv(output_path, index=False)

print(f"Los resultados con los clusters asignados se han guardado en: {output_path}")
