import cv2
import numpy as np
import joblib
import mediapipe as mp
import matplotlib.pyplot as plt

class RoiRecognition:
    def __init__(self):
        """
        Inicializa la clase RoiRecognition cargando el modelo SVM.
        """
        self.clf = joblib.load("model_archivos/svm_digit_classifier.pkl")

    def preprocess_image(self, image):
        """
        Preprocesa la imagen ROI para invertir los colores, normalizar,
        aplicar un desenfoque Gaussiano y convertirla a binaria.

        Args:
            image (numpy.ndarray): Imagen de la región de interés (ROI).

        Returns:
            numpy.ndarray: Imagen preprocesada de 28x28 píxeles.
        """
        # Invertir los colores de la imagen (blanco a negro y viceversa)
        inverted_roi = cv2.bitwise_not(image)
        # Normalizar la imagen para que los valores de los píxeles estén entre 0 y 255
        normalized_image = cv2.normalize(inverted_roi, None, 0, 255, cv2.NORM_MINMAX)
        # Aplicar un desenfoque Gaussiano para suavizar la imagen
        blurred_image = cv2.GaussianBlur(normalized_image, (3, 3), 0)
        # Convertir la imagen a binaria utilizando un umbral de 128
        _, binary_image = cv2.threshold(blurred_image, 128, 255, cv2.THRESH_BINARY)
        # Preprocesar la imagen binaria
        return self.preprocess_binary_image(binary_image)

    def preprocess_binary_image(self, image):
        """
        Preprocesa la imagen binaria recortando, redimensionando y aplicando
        operaciones morfológicas para preparar la imagen para el modelo SVM.

        Args:
            image (numpy.ndarray): Imagen binaria.

        Returns:
            numpy.ndarray: Imagen preprocesada de 28x28 píxeles.
        """
        # Encontrar las coordenadas de los píxeles blancos en la imagen binaria
        x_coords, y_coords = np.where(image > 0)
        if len(x_coords) == 0 or len(y_coords) == 0:
            return None
        # Determinar los límites de la región blanca en la imagen
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        if x_min == x_max or y_min == y_max:
            return None
        # Agregar un margen alrededor de la región blanca
        margin = 50
        x_min = max(x_min - margin, 0)
        x_max = min(x_max + margin, image.shape[1])
        y_min = max(y_min - margin, 0)
        y_max = min(y_max + margin, image.shape[0])
        # Recortar la imagen a la región blanca con el margen añadido
        cropped_image = image[y_min:y_max, x_min:x_max]
        # Redimensionar la imagen a 20x20 píxeles
        resized_image = cv2.resize(cropped_image, (20, 20), interpolation=cv2.INTER_AREA)
        # Colocar la imagen redimensionada en el centro de una imagen en blanco de 28x28 píxeles
        final_image = np.zeros((28, 28), dtype=np.uint8)
        final_image[4:24, 4:24] = resized_image
        # Aplicar operaciones morfológicas para cerrar los pequeños agujeros en la imagen
        kernel = np.ones((2, 2), np.uint8)
        final_image = cv2.morphologyEx(final_image, cv2.MORPH_CLOSE, kernel)
        # Ecualizar el histograma de la imagen para mejorar el contraste
        final_image = cv2.equalizeHist(final_image)
        # Convertir la imagen a binaria nuevamente utilizando un umbral de 128
        _, final_image = cv2.threshold(final_image, 128, 255, cv2.THRESH_BINARY)
        return final_image

    def predict(self, image):
        """
        Predice el dígito a partir de la imagen ROI preprocesada.

        Args:
            image (numpy.ndarray): Imagen de la región de interés (ROI).

        Returns:
            int: Dígito predicho.
        """
        processed_image = self.preprocess_image(image)
        if processed_image is not None:
            processed_image = processed_image.reshape(1, -1)
            return self.clf.predict(processed_image)[0]
        return None

    def run(self):
        """
        Inicia la captura de video y el reconocimiento de dígitos en la región de interés (ROI) en tiempo real.
        """
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convertir la imagen a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Aplicar un desenfoque Gaussiano para suavizar la imagen
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Convertir la imagen a binaria utilizando un umbral de 128
            _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
            # Encontrar los contornos en la imagen binaria
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            offset = 300  # Aumenta el tamaño del offset
            center_region = (center_x - offset, center_y - offset, center_x + offset, center_y + offset)
            best_contour = None
            best_area = 0
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                # Verificar si el contorno está dentro de la región central y si cumple con los criterios de tamaño
                if (center_region[0] < x < center_region[2] and center_region[1] < y < center_region[3] and 
                    30 < w < 200 and 30 < h < 200 and area > best_area):
                    best_contour = cnt
                    best_area = area
            if best_contour is not None:
                x, y, w, h = cv2.boundingRect(best_contour)
                roi = gray[y:y+h, x:x+w]
                processed_digit = self.preprocess_image(roi)
                if processed_digit is not None:
                    # Mostrar la imagen segmentada y preprocesada
                    cv2.imshow('Segmented and Preprocessed Image', cv2.resize(processed_digit, (280, 280), interpolation=cv2.INTER_AREA))
                    roi_digits = processed_digit.reshape((1, -1))
                    number_poly = self.clf.predict(roi_digits)
                    # Dibujar un rectángulo alrededor de la región de interés y mostrar el dígito predicho
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f'{int(number_poly)}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Mostrar la imagen con los dígitos y la trayectoria
            cv2.imshow('Frame with Digits and Trajectory', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
