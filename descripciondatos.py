import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib


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

print("tamaño x train: ",X_train.shape)
print("tamaño x test: ",X_test.shape)