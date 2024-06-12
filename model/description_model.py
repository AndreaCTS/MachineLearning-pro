import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def load_mnist_data():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Aplanar las imágenes
    # Redimensionar las imágenes a 28x28, aplanar y normalizar 
    X_train = X_train.reshape(-1, 28*28).astype('float32')/255.0
    X_test = X_test.reshape(-1, 28*28).astype('float32')/255.0
    return X_train, X_test, y_train, y_test

    
# Cargar el dataset MNIST
X_train, X_test, y_train, y_test = load_mnist_data()

best_svm_grid = joblib.load("model_archivos/svm_digit_classifier.pkl")

best_svm = best_svm_grid.best_estimator_

# Mostrar los parámetros del kernel polinomial
coef0 = best_svm.coef0
degree = best_svm.degree
gamma = best_svm.gamma
C = best_svm.C

print(f'Coef0 (r): {coef0}')
print(f'Degree (d): {degree}')
print(f'Gamma: {gamma}')
print(f'C: {C}')

# Descripción matemática de la función del SVM
print(f"La función del SVM es: ({gamma} * <x, x'> + {coef0})^{degree}")


y_predPoly = best_svm.predict(X_test)
# Generar y visualizar la matriz de confusión
cm = confusion_matrix(y_test, y_predPoly)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Mostrar el reporte de clasificación
report = classification_report(y_test, y_predPoly)
print("\nReporte de clasificación:\n", report)