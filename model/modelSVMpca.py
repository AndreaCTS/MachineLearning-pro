import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from tensorflow.keras.datasets import mnist
import tensorflow as tf

# Cargar los datos MNIST
def load_mnist_data():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Aplanar las imágenes
    X_train = X_train.reshape(-1, 28*28).astype('float32')/255.0
    X_test = X_test.reshape(-1, 28*28).astype('float32')/255.0
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_mnist_data()


# Reducir la dimensionalidad a 3D usando PCA
pca = PCA(n_components=2)
X_train_3D = pca.fit_transform(X_train)
X_test_3D = pca.transform(X_test)

# Entrenar un SVM en el espacio 3D reducido
svm_3D = SVC(kernel='poly', C=10, gamma='scale')
svm_3D.fit(X_train_3D, y_train)

joblib.dump(svm_3D, "model_archivos/svm_digit_classifier_PCA2.pkl")
np.save("model_archivos/test2.npy", X_test_3D)
np.save("model_archivos/test.npy", y_test)


