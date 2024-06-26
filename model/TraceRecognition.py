import cv2
import numpy as np
import joblib
import mediapipe as mp
import matplotlib.pyplot as plt

class TraceRecognition:
    def __init__(self):
        """
        Inicializa la clase TraceRecognition cargando el modelo SVM,
        configurando MediaPipe Hands y la captura de video.
        """
        self.clf = joblib.load("model_archivos/svm_digit_classifier.pkl")
        self.trajectory = []
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.cap = cv2.VideoCapture(0)

    def smooth_trajectory(self, trajectory, alpha=0.75):
        """
        Suaviza la trayectoria utilizando un filtro de media móvil exponencial.

        Args:
            trajectory (list): Lista de coordenadas (x, y) de la trayectoria.
            alpha (float): Factor de suavizado.

        Returns:
            list: Lista de coordenadas suavizadas.
        """
        smoothed = []
        if len(trajectory) > 1:
            smoothed.append(trajectory[0])
            for i in range(1, len(trajectory)):
                prev = smoothed[-1]
                curr = trajectory[i]
                smoothed.append((int(prev[0] * alpha + curr[0] * (1 - alpha)), int(prev[1] * alpha + curr[1] * (1 - alpha))))
        else:
            smoothed = trajectory
        return smoothed

    def interpolate_trajectory(self, trajectory):
        """
        Interpola la trayectoria para agregar puntos adicionales entre las coordenadas.

        Args:
            trajectory (list): Lista de coordenadas (x, y) de la trayectoria.

        Returns:
            list: Lista de coordenadas interpoladas.
        """
        interpolated = []
        for i in range(1, len(trajectory)):
            prev = trajectory[i - 1]
            curr = trajectory[i]
            interpolated.append(prev)
            # Agrega puntos adicionales entre las coordenadas para suavizar la trayectoria
            for t in np.linspace(0, 1, num=10):
                interpolated.append((int(prev[0] * (1 - t) + curr[0] * t), int(prev[1] * (1 - t) + curr[1] * t)))
        interpolated.append(trajectory[-1])
        return interpolated

    def preprocess_trajectory(self, trajectory, frame_shape):
        """
        Preprocesa la trayectoria para crear una imagen adecuada para la clasificación del modelo SVM.

        Args:
            trajectory (list): Lista de coordenadas (x, y) de la trayectoria.
            frame_shape (tuple): Forma del marco de la imagen capturada.

        Returns:
            numpy.ndarray: Imagen preprocesada de 28x28 píxeles.
        """
        if not trajectory:
            print("Trajectory is empty")
            return None

        # Crear una imagen en blanco
        image = np.zeros((480, 640), dtype=np.uint8)
        # Dibujar la trayectoria interpolada en la imagen
        for (x, y) in self.interpolate_trajectory(trajectory):
            cv2.circle(image, (x, y), 5, 255, -1)
        
        # Encontrar los límites de la trayectoria
        x_coords, y_coords = zip(*trajectory)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Asegurarse de que los límites son válidos
        if x_min == x_max or y_min == y_max:
            return None
        
        # Recortar la imagen al tamaño de la trayectoria con un margen
        margin = 10
        x_min = max(x_min - margin, 0)
        x_max = min(x_max + margin, image.shape[1])
        y_min = max(y_min - margin, 0)
        y_max = min(y_max + margin, image.shape[0])
        cropped_image = image[y_min:y_max, x_min:x_max]
        
        # Redimensionar la imagen a 20x20 píxeles
        resized_image = cv2.resize(cropped_image, (20, 20), interpolation=cv2.INTER_AREA)
        
        # Colocar la imagen redimensionada en el centro de una imagen en blanco de 28x28 píxeles
        final_image = np.zeros((28, 28), dtype=np.uint8)
        final_image[4:24, 4:24] = resized_image
        
        # Aplicar operaciones morfológicas y normalización
        kernel = np.ones((2, 2), np.uint8)
        final_image = cv2.morphologyEx(final_image, cv2.MORPH_CLOSE, kernel)
        final_image = cv2.GaussianBlur(final_image, (3, 3), 0)
        final_image = cv2.equalizeHist(final_image)
        return final_image
    
    def predict(self, trajectory, frame_shape):
        """
        Predice el dígito a partir de la trayectoria preprocesada.

        Args:
            trajectory (list): Lista de coordenadas (x, y) de la trayectoria.
            frame_shape (tuple): Forma del marco de la imagen capturada.

        Returns:
            int: Dígito predicho.
        """
        input_img = self.preprocess_trajectory(trajectory, frame_shape)
        if input_img is not None:
            input_img_for_model = input_img.reshape(1, -1)
            digit = self.clf.predict(input_img_for_model)
            return digit[0]
        return None

    def run(self):
        """
        Inicia la captura de video y el reconocimiento de trazos en tiempo real.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Invertir la imagen horizontalmente para una mejor experiencia del usuario
            frame = cv2.flip(frame, 1)
            
            # Convertir la imagen a RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar la imagen para encontrar las manos
            result = self.hands.process(rgb_frame)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Obtener las coordenadas del índice del dedo
                    index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    h, w, _ = frame.shape
                    cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                    
                    # Dibujar el círculo en la punta del dedo índice
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
                    
                    # Agregar la coordenada a la trayectoria
                    self.trajectory.append((cx, cy))
                    
                    # Suavizar la trayectoria y dibujarla en la imagen
                    smoothed_trajectory = self.smooth_trajectory(self.trajectory)
                    for i in range(1, len(smoothed_trajectory)):
                        if smoothed_trajectory[i - 1] is None or smoothed_trajectory[i] is None:
                            continue
                        cv2.line(frame, smoothed_trajectory[i - 1], smoothed_trajectory[i], (0, 255, 0), 2)
            
            # Mostrar la imagen con la trayectoria
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1) & 0xFF
            
            # Si el usuario presiona 'p', procesar la trayectoria y predecir el dígito
            if key == ord('p'):
                if self.trajectory:
                    input_img = self.preprocess_trajectory(self.trajectory, frame.shape)
                    if input_img is not None:
                        input_img_for_model = input_img.reshape(1, -1)
                        digit = self.clf.predict(input_img_for_model)
                        print(f'Predicted Digit: {digit[0]}')
                        
                        # Guardar la imagen del rastro
                        trajectory_img = np.zeros((480, 640), dtype=np.uint8)
                        for (x, y) in self.trajectory:
                            cv2.circle(trajectory_img, (x, y), 5, 255, -1)
                        cv2.imwrite('trajectory.png', trajectory_img)
                        
                        # Mostrar la imagen del trazo y la imagen que se envía al modelo
                        cv2.imshow('Trace Image', cv2.resize(trajectory_img, (280, 280), interpolation=cv2.INTER_AREA))
                        cv2.imshow('Model Input Image', (input_img * 255).astype(np.uint8))
                        
                        # Visualizar el preprocesamiento detallado
                        plt.figure(figsize=(5, 5))
                        plt.imshow(input_img.reshape(28, 28), cmap='gray')
                        plt.title(f'Predicted: {digit[0]}')
                        plt.axis('off')
                        plt.show()
                        
                        # Limpiar la trayectoria
                        self.trajectory = []
                    else:
                        print("Invalid trajectory, try again.")
            elif key == ord('c'):
                # Limpiar la trayectoria
                self.trajectory = []
                print("Trazo eliminado. Empieza de nuevo.")
            elif key == ord('q'):
                break
        
        # Liberar la captura de video y cerrar las ventanas
        self.cap.release()
        cv2.destroyAllWindows()

