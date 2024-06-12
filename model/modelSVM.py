import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Función para cargar y preprocesar el dataset MNIST
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


# Visualizar las imágenes con sus predicciones
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    plt.title("preprocesamiento")
    plt.axis('off')
plt.tight_layout()
plt.show()

#Ajusta los hiperparámetros del modelo SVM utilizando GridSearchCV.
param_grid = {
            'C': [1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear','rbf', 'poly']
        }


# Ajustar el modelo SVM con kernel polinomial
grid_svm = GridSearchCV(SVC(), param_grid, refit=True, cv=5, n_jobs=-1)
grid_svm.fit(X_train, y_train)
print(f'Mejores hiperparámetros: {grid_svm.best_params_}')
print(f'Mejor puntuación: {grid_svm.best_score_}')
y_predPoly = grid_svm.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_predPoly)}')



# Guardar el modelo entrenado y el scaler
joblib.dump(grid_svm, "model_archivos/svm_digit_classifier.pkl")
#joblib.dump(scaler, "scaler_28x28.pkl")
