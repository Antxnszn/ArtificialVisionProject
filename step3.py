import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
dataset = pd.read_csv("imagenes.csv")

# Seleccionar las características para el clustering
X_dataset = dataset[['area', 'perimeter', 'angle_of_inclination', 'texture', 'intensity']]

# Escalar los datos para mejorar el rendimiento del clustering y 1NN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dataset)

# Inicializar el modelo K-means con 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Asignar los clusters al dataset
dataset['Cluster'] = kmeans.labels_

# Dividir los datos en entrenamiento y prueba (70%-30%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, dataset['Cluster'], test_size=0.3, stratify=dataset['Cluster'], random_state=42
)

# Inicializar y entrenar el clasificador 1NN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Realizar predicciones para el conjunto de prueba
y_pred = knn.predict(X_test)

# Crear un DataFrame para guardar las predicciones del 1NN
test_results = pd.DataFrame(X_test, columns=['area', 'perimeter', 'angle_of_inclination', 'texture', 'intensity'])
test_results['Cluster_Pred'] = y_pred

# Guardar las predicciones del 1NN en un nuevo archivo CSV
output_path_knn = "imagenes_knn_results.csv"
test_results.to_csv(output_path_knn, index=False)

# Resultados
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Precisión con 1NN después del clustering:", accuracy)
print("Matriz de Confusión:\n", conf_matrix)
print(f"Resultados del 1NN guardados en: {output_path_knn}")
