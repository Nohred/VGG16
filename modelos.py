import warnings
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

# Ignorar todos los warnings
warnings.filterwarnings("ignore")

# Ruta genérica del archivo CSV
# dataset = 'features_dataset_block4_conv1'
# dataset = 'features_dataset_block4_conv1'
ruta = 'CamVid_bal/features_dataset_block5_conv1_pca.csv'  # Cambia esta ruta según sea necesario

# Leer el archivo CSV
tabla = pd.read_csv(ruta)

# División de los datos en X (características) e Y (etiquetas)
X = tabla.iloc[:, :-1].values  # Todas las columnas excepto la última
Y = tabla.iloc[:, -1].values    # Solo la última columna

# Definición de métricas personalizadas para validación cruzada
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted')
}

# ---------------- DECLARACION DE LOS CLASIFICADORES ----------------

# LDA
lda = LDA(solver='svd')

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,        # Número de árboles en el bosque
    max_depth=10,            # Profundidad máxima de los árboles
    min_samples_split=10,    # Número mínimo de muestras requeridas para dividir un nodo
    min_samples_leaf=4,      # Número mínimo de muestras requeridas en una hoja
    random_state=42          # Fijar la semilla para reproducibilidad
)

# AdaBoost
base_estimator = DecisionTreeClassifier(
    max_depth=3,             # Profundidad máxima del árbol base
    min_samples_split=10,    # Número mínimo de muestras requeridas para dividir un nodo
    min_samples_leaf=4,      # Número mínimo de muestras requeridas en una hoja
    random_state=42          # Fijar la semilla para reproducibilidad
)
ab = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=50,         # Número de árboles en el bosque
    learning_rate=0.3,       # Tasa de aprendizaje
    random_state=42          # Fijar la semilla para reproducibilidad
)

# ---------------- EVALUACIÓN DE LOS CLASIFICADORES ----------------


# Evaluación con validación cruzada
scores_lda = cross_validate(lda, X, Y, cv=3, scoring=scoring)
scores_rf = cross_validate(rf, X, Y, cv=3, scoring=scoring)
scores_adaboost = cross_validate(ab, X, Y, cv=3, scoring=scoring)

# Resultados Promedios
print('Resultados Promedios:')
print(f'LDA - Accuracy: {np.mean(scores_lda["test_accuracy"]):.2f}, Precision: {np.mean(scores_lda["test_precision"]):.2f}, Recall: {np.mean(scores_lda["test_recall"]):.2f}')
print(f'Bosque Aleatorio - Accuracy: {np.mean(scores_rf["test_accuracy"]):.2f}, Precision: {np.mean(scores_rf["test_precision"]):.2f}, Recall: {np.mean(scores_rf["test_recall"]):.2f}')
print(f'AdaBoost - Accuracy: {np.mean(scores_adaboost["test_accuracy"]):.2f}, Precision: {np.mean(scores_adaboost["test_precision"]):.2f}, Recall: {np.mean(scores_adaboost["test_recall"]):.2f}')

# Función para generar y mostrar la matriz de confusión para cada modelo
def plot_confusion_matrix(model, X, Y, title):
    from sklearn.model_selection import cross_val_predict
    Y_pred = cross_val_predict(model, X, Y, cv=5)
    C_matrix = confusion_matrix(Y, Y_pred)
    plt.xlabel("True Target")
    plt.ylabel("Predicted Target")
    disp = ConfusionMatrixDisplay(confusion_matrix=C_matrix)
    disp.plot()
    plt.title(title)
    plt.show()

def plot_learning_curve(model, X, Y, title):
    train_sizes, train_scores, test_scores = learning_curve(model, X, Y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label="Training score", color="r")
    plt.plot(train_sizes, test_scores_mean, label="Cross-validation score", color="g")
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# Generar y mostrar las matrices de confusión y curvas de aprendizaje
plot_confusion_matrix(ab, X, Y, "Confusion Matrix - Ada Boost")
plot_learning_curve(ab, X, Y, "Learning Curve - Ada Boost")
plot_confusion_matrix(rf, X, Y, "Confusion Matrix - Random Forest")
plot_learning_curve(rf, X, Y, "Learning Curve - Random Forest")
plot_confusion_matrix(lda, X, Y, "Confusion Matrix - LDA")
plot_learning_curve(lda, X, Y, "Learning Curve - LDA")
