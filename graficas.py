import joblib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
import tensorflow as tf
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

# Cargar el modelo SVM
svm = joblib.load("svm_digit_classifier.pkl")

# Cargar los datos MNIST
def load_mnist_data():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Aplanar las imágenes
    X_train = X_train.reshape(-1, 28*28).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28*28).astype('float32') / 255.0
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_mnist_data()

# Reducir la dimensionalidad a 3D usando PCA
pca = PCA(n_components=3)
X_train_3D = pca.fit_transform(X_train)
X_test_3D = pca.transform(X_test)

# Entrenar un SVM en el espacio 3D reducido
svm_3D = SVC(kernel='poly', degree=3, C=10)
svm_3D.fit(X_train_3D, y_train)

def plot_hyperplane_only(clf, X):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 50),
                             np.linspace(y_min, y_max, 50),
                             np.linspace(z_min, z_max, 50))
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    Z = clf.decision_function(grid)
    Z = Z.reshape(xx.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Decision Boundary in 3D PCA Space')
    
    # Visualizar el límite de decisión
    ax.plot_surface(xx[:, :, 0], yy[:, :, 0], Z[:, :, 0], alpha=0.3, cmap='coolwarm')
    
    plt.show()

# Visualizar el límite de decisión en 3D
plot_hyperplane_only(svm_3D, X_test_3D)
