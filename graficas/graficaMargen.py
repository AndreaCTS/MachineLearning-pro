import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Cargar los datos MNIST
def load_mnist_data():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Aplanar las imágenes
    X_train = X_train.reshape(-1, 28*28).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28*28).astype('float32') / 255.0
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_mnist_data()

# Reducir las dimensiones a 2D con PCA
pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)


class_of_interest = 0
y_train_binary = (y_train == class_of_interest).astype(int)

svm_binary = SVC(C=10, kernel='poly')
svm_binary.fit(X_train_reduced, y_train_binary)

# Función para visualizar el margen suave en 2D
def plot_soft_margin(clf, X, y):
    print(f"Tamaño de X_reduced: {X.shape}")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    
    print(f"Tamaño de xx: {xx.shape}, Tamaño de yy: {yy.shape}")
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    print(f"Tamaño de grid: {grid.shape}")
    
    Z = clf.decision_function(grid)
    print(f"Tamaño de Z antes de reshaping: {Z.shape}")
    Z = Z.reshape(xx.shape)
    print(f"Tamaño de Z después de reshaping: {Z.shape}")
    
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    
    # Dibujar los márgenes suaves
    plt.contour(xx, yy, Z, levels=[-1, 1], linewidths=2, colors='green', linestyles='dashed')
    
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap=plt.cm.Paired)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Soft Margin for Class {class_of_interest} in 2D PCA Space')
    plt.show()

# Visualizar el margen suave en 2D
plot_soft_margin(svm_binary, X_train_reduced, y_train_binary)