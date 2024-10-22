from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import time
import pandas as pd  # Importar pandas para guardar el CSV

# Directorio principal del dataset
dataset_dir = 'CamVid_bal/'
# capa = 'fc2'
capa = 'block5_conv1'

start_time = time.time()

# Cargar el modelo VGG16 con las capas totalmente conectadas
base_model = VGG16(weights='imagenet', include_top=True)

# Crear un modelo que tenga como salida la capa 'fc2' (fc7 en algunos contextos)
model = Model(inputs=base_model.input, outputs=base_model.get_layer(capa).output)

# Resumen del modelo
# model.summary()

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Redimensionar la imagen a 224x224
    img_array = image.img_to_array(img)  # Convertir la imagen a un array numpy
    img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión para representar el lote de imágenes
    img_array = preprocess_input(img_array)  # Preprocesar la imagen (normalización y otras operaciones)
    return img_array

def extract_features(img_path, model):
    img_array = preprocess_image(img_path)
    features = model.predict(img_array)  # Predecir las características
    return features

# Extraer características de todas las imágenes .png en todas las subcarpetas de categorías
all_features = []
image_labels = []  # Guardar las etiquetas de las categorías

# Recorrer las carpetas de cada categoría
for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)
    
    # Verificar si es una carpeta
    if os.path.isdir(category_path):
        # Recorrer todas las imágenes .png en la carpeta de la categoría
        image_paths = [os.path.join(category_path, f) for f in os.listdir(category_path) if f.endswith('.png')]
        
        for img_path in image_paths:
            features = extract_features(img_path, model)  # Extraer las características
            all_features.append(features)  # Guardar las características
            image_labels.append(category)  # Guardar la etiqueta (nombre de la categoría)

# Aplanar las características para que se puedan usar en modelos posteriores
all_features_flattened = [features.flatten() for features in all_features]

# Convertir las características y etiquetas en arrays de numpy
all_features_flattened = np.array(all_features_flattened)
image_labels = np.array(image_labels)

print("Características extraídas: ", all_features_flattened.shape)
print("Etiquetas de las imágenes: ", image_labels.shape)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Tiempo de convolucion transcurrido: {elapsed_time:.2f} segundos")

# Escalar los datos
scaler = StandardScaler()
X = scaler.fit_transform(all_features_flattened) 
Y = image_labels

start_time = time.time()

# Crear un DataFrame de pandas con las características y etiquetas
df = pd.DataFrame(X)
df['label'] = Y  # Añadir la columna de etiquetas

# Guardar el DataFrame como un archivo CSV en el mismo directorio del dataset
nombre = 'features_dataset_'+ capa +'.csv'
csv_path = os.path.join(dataset_dir, nombre)
df.to_csv(csv_path, index=False)

print(f"El dataset ha sido guardado en: {csv_path}")

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Tiempo de guardado transcurrido: {elapsed_time:.2f} segundos")
