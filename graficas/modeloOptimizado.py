import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV

class SVMClassifier:
    def __init__(self, class_of_interest=1):
        """
        Inicializa la clase SVMClassifier con los parámetros especificados.

        Args:
            class_of_interest (int): Clase de interés para la clasificación binaria.
        """
        self.class_of_interest = class_of_interest
        self.pca = PCA(n_components=3)
        self.svm = joblib.load("model_archivos/svm_digit_classifier_PCA3.pkl")

    def load_mnist_data(self):
        """
        Carga el conjunto de datos MNIST y lo preprocesa.

        Returns:
            X_train, X_test (numpy.ndarray): Conjuntos de entrenamiento y prueba de las imágenes.
            y_train, y_test (numpy.ndarray): Conjuntos de entrenamiento y prueba de las etiquetas.
        """
        mnist = tf.keras.datasets.mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(-1, 28*28).astype('float32') / 255.0
        X_test = X_test.reshape(-1, 28*28).astype('float32') / 255.0
        return X_train, X_test, y_train, y_test

    def preprocess_data(self, X_train, X_test):
        """
        Reduce las dimensiones de los datos de entrenamiento y prueba utilizando PCA.

        Args:
            X_train (numpy.ndarray): Conjunto de datos de entrenamiento.
            X_test (numpy.ndarray): Conjunto de datos de prueba.

        Returns:
            X_train_reduced, X_test_reduced (numpy.ndarray): Datos reducidos a 3 dimensiones.
        """
        X_train_reduced = self.pca.fit_transform(X_train)
        X_test_reduced = self.pca.transform(X_test)
        return X_train_reduced, X_test_reduced

    def train_model(self, X_train, y_train):
        """
        Entrena el modelo SVM utilizando los datos de entrenamiento.

        Args:
            X_train (numpy.ndarray): Datos de entrenamiento reducidos.
            y_train (numpy.ndarray): Etiquetas de entrenamiento.

        Returns:
            y_train_binary (numpy.ndarray): Etiquetas de entrenamiento binarizadas para la clase de interés.
        """
        y_train_binary = (y_train == self.class_of_interest).astype(int)
        self.svm.fit(X_train, y_train_binary)
        return y_train_binary

    def evaluate_model(self, X_test, y_test):
        """
        Evalúa el modelo SVM utilizando los datos de prueba y muestra las métricas de rendimiento.

        Args:
            X_test (numpy.ndarray): Datos de prueba reducidos.
            y_test (numpy.ndarray): Etiquetas de prueba.
        """
        y_test_binary = (y_test == self.class_of_interest).astype(int)
        y_pred = self.svm.predict(X_test)
        accuracy = accuracy_score(y_test_binary, y_pred)
        report = classification_report(y_test_binary, y_pred)
        cm = confusion_matrix(y_test_binary, y_pred)

        print("\n\t\t\t\t--------- METRICAS ---------")
        print(f'\nAccuracy: {accuracy}')
        print(f'\nClassification Report:\n{report}')
        print(f'\nConfusion Matrix:\n{cm}\n\n')

    def plot_soft_margin(self, X, y):
        """
        Visualiza el margen suave de la clasificación SVM en un espacio 2D PCA.

        Args:
            X (numpy.ndarray): Datos reducidos.
            y (numpy.ndarray): Etiquetas binarizadas.
        """
        pca_2d = PCA(n_components=2)
        X_2d = pca_2d.fit_transform(X)

        print(f"Tamaño de X_reduced: {X_2d.shape}")

        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))

        print(f"Tamaño de xx: {xx.shape}, Tamaño de yy: {yy.shape}")

        grid = np.c_[xx.ravel(), yy.ravel()]

        print(f"Tamaño de grid: {grid.shape}")

        Z = self.svm.decision_function(pca_2d.inverse_transform(grid))
        print(f"Tamaño de Z antes de reshaping: {Z.shape}")
        Z = Z.reshape(xx.shape)
        print(f"Tamaño de Z después de reshaping: {Z.shape}")

        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')

        # Dibujar los márgenes suaves
        plt.contour(xx, yy, Z, levels=[-1, 1], linewidths=2, colors='green', linestyles='dashed')

        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=30, edgecolor='k', cmap=plt.cm.Paired)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'Soft Margin for Class {self.class_of_interest} in 2D PCA Space')
        save_fig("soft_margin_2D")
        plt.show()

    def plot_decision_boundary_3D(self, X):
        """
        Visualiza el límite de decisión en un espacio 3D PCA.

        Args:
            X (numpy.ndarray): Datos reducidos a 3 dimensiones.
        """
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

        xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 30),
                                np.linspace(y_min, y_max, 30),
                                np.linspace(z_min, z_max, 30))
        grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        Z = self.svm.decision_function(grid)
        Z = Z.reshape(xx.shape)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'Decision Boundary for class  {self.class_of_interest} in 3D PCA Space')
        
        # Visualizar el límite de decisión
        ax.plot_surface(xx[:, :, 0], yy[:, :, 0], Z[:, :, 0], alpha=0.3, cmap='coolwarm')
        save_fig("decision_boundary_3D")
        plt.show()

    def plot_class_distribution_3D(self, X_3d, y):
        """
        Visualiza la distribución de puntos de las 10 clases en el espacio 3D PCA.

        Args:
            X_3d (numpy.ndarray): Datos reducidos a 3 dimensiones.
            y (numpy.ndarray): Etiquetas.
        """

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y, cmap=plt.get_cmap("tab10"), s=10, alpha=0.7)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('Class Distribution in 3D PCA Space')
        fig.legend(handles=scatter.legend_elements()[0], labels=list(range(10)))
        save_fig("distribution__points_3D")
        plt.show()

if __name__ == "__main__":

    IMAGES_PATH = ""

    def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
        path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)


    IMAGES_PATH = Path() / "graficas" / "por_clase" 
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    svm_classifier = SVMClassifier(class_of_interest=0)
        
    X_train, X_test, y_train, y_test = svm_classifier.load_mnist_data()
    X_train_reduced, X_test_reduced = svm_classifier.preprocess_data(X_train, X_test)
        
    y_train_binary = svm_classifier.train_model(X_train_reduced, y_train)
        
    svm_classifier.evaluate_model(X_test_reduced, y_test)

    svm_classifier.plot_soft_margin(X_train_reduced, y_train_binary)
    svm_classifier.plot_decision_boundary_3D(X_train_reduced)
    #svm_classifier.plot_class_distribution_3D(X_train_reduced,y_train)


    


