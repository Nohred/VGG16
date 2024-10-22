import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
import time

ruta = 'CamVid_bal/'

dataset = 'features_dataset_fc2'
# dataset = 'features_dataset_block5_conv1'

def apply_pca_lda(df, pca_components=5, lda_components=5):
    X = df.iloc[:, :-1]  # Todas las columnas menos la última
    y = df.iloc[:, -1]   # La última columna

    # Si las etiquetas son categóricas, las codificamos
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Aplicar PCA
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X)

    # Aplicar LDA
    lda = LDA(n_components=lda_components)
    X_lda = lda.fit_transform(X_pca, y)

    # Convertir a DataFrame con etiquetas
    pca_df = pd.DataFrame(X_pca)
    lda_df = pd.DataFrame(X_lda)
    pca_df['Label'] = y
    lda_df['Label'] = y

    return pca_df, lda_df

def reduce_dimensions(input_csv, pca_components=10, lda_components=5):
    print("Leyendo dataset...")
    start = time.time()

    # Leer el CSV completo
    data = pd.read_csv(input_csv)

    print(f"Dataset leído. Shape: {data.shape}")

    # Aplicar PCA y LDA al dataset completo
    pca_results, lda_results = apply_pca_lda(data, pca_components=pca_components, lda_components=lda_components)

    end = time.time()
    print(f"Dataset leído y procesado en {(end - start) / 60:.2f} minutos.")

    # Guardar los resultados
    print("Guardando resultados...")
    pca_output = ruta + dataset + '_pca.csv'
    lda_output = ruta + dataset + '_lda.csv'

    pca_results.to_csv(pca_output, index=False)
    lda_results.to_csv(lda_output, index=False)

    print(f'Reducción completada. Archivos guardados: {pca_output}, {lda_output}')

# Uso del programa
input_csv_file = ruta + dataset + '.csv'
reduce_dimensions(input_csv_file)