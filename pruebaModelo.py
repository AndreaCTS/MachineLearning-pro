import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib

# Cargar el modelo SVM y el escalador
clf = joblib.load("svm_digit_classifier.pkl")

# Función para cargar y preprocesar el dataset MNIST
def load_mnist_data():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Redimensionar las imágenes a 28x28 y aplanar
    X_train = X_train.reshape(-1, 28*28).astype('float32')
    X_test = X_test.reshape(-1, 28*28).astype('float32')
    return X_train, X_test, y_train, y_test

# Cargar el dataset MNIST
X_train, X_test, y_train, y_test = load_mnist_data()

print()

# Estandarizar las características
#X_test = scaler.transform(X_test)

# Seleccionar algunas imágenes de prueba al azar
num_images = 10
indices = np.random.choice(len(X_test), num_images, replace=False)
sample_images = X_test[indices]
sample_labels = y_test[indices]

# Predecir con el modelo SVM
predictions = clf.predict(sample_images)

# Visualizar las imágenes con sus predicciones
plt.figure(figsize=(10, 5))
for i in range(num_images):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {predictions[i]}\nTrue: {sample_labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
