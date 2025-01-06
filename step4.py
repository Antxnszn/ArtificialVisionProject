import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Cargar el dataset
dataset = pd.read_csv("imagenes.csv")

# Seleccionar las características para el clustering
X_dataset = dataset[['area', 'perimeter', 'angle_of_inclination', 'texture', 'intensity']]

### 1. Versión Normalizada ###
# Escalar los datos
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_dataset)

# Inicializar y ajustar el modelo K-means con datos normalizados
kmeans_normalized = KMeans(n_clusters=5, random_state=42)
clusters_normalized = kmeans_normalized.fit_predict(X_normalized)

# Agregar los clusters al dataset original
dataset['Cluster_Normalized'] = clusters_normalized

# Guardar los resultados de la versión normalizada en un nuevo archivo CSV
output_path_normalized = "imagenes_clustered_normalized.csv"
dataset.to_csv(output_path_normalized, index=False)

print(f"Resultados con datos normalizados guardados en: {output_path_normalized}")


### 2. Versión con PCA ###
# Reducir la dimensionalidad a 2 componentes principales
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_normalized)

# Inicializar y ajustar el modelo K-means con datos reducidos por PCA
kmeans_pca = KMeans(n_clusters=5, random_state=42)
clusters_pca = kmeans_pca.fit_predict(X_pca)

# Crear nuevas columnas con las componentes principales y los clusters
dataset['PCA_Component_1'] = X_pca[:, 0]
dataset['PCA_Component_2'] = X_pca[:, 1]
dataset['Cluster_PCA'] = clusters_pca

# Guardar los resultados de la versión con PCA en un nuevo archivo CSV
output_path_pca = "imagenes_clustered_pca.csv"
dataset.to_csv(output_path_pca, index=False)

print(f"Resultados con PCA guardados en: {output_path_pca}")
