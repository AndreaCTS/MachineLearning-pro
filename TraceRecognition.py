import cv2
import numpy as np
import joblib
import mediapipe as mp
import matplotlib.pyplot as plt


class TraceRecognition():

    def __init__(self):
        self.clf = joblib.load("svm_digit_classifier.pkl")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.trajectory = []


    def smooth_trajectory(self,trajectory, alpha=0.75):
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

    def interpolate_trajectory(self,trajectory):
        interpolated = []
        for i in range(1, len(trajectory)):
            prev = trajectory[i - 1]
            curr = trajectory[i]
            interpolated.append(prev)
            # Interpolar puntos adicionales
            for t in np.linspace(0, 1, num=10):
                interpolated.append((int(prev[0] * (1 - t) + curr[0] * t), int(prev[1] * (1 - t) + curr[1] * t)))
        interpolated.append(trajectory[-1])
        return interpolated

    def preprocess_trajectory(self,trajectory, frame_shape):
        # Crear una imagen en blanco de 640x480 para capturar la resolución completa
        image = np.zeros((480, 640), dtype=np.uint8)
        
        # Dibujar la trayectoria en la imagen con un círculo de tamaño moderado
        for (x, y) in trajectory:
            cv2.circle(image, (x, y), 5, 255, -1)  # Reducir el tamaño del círculo a 5

        # Encontrar los límites del trazo
        x_coords, y_coords = zip(*trajectory)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Asegurarse de que los límites son válidos
        if x_min == x_max or y_min == y_max:
            return None

        # Recortar la imagen al tamaño del trazo con un pequeño margen
        margin = 10
        x_min = max(x_min - margin, 0)
        x_max = min(x_max + margin, image.shape[1])
        y_min = max(y_min - margin, 0)
        y_max = min(y_max + margin, image.shape[0])

        cropped_image = image[y_min:y_max, x_min:x_max]

        # Redimensionar la imagen recortada a 20x20 píxeles (MNIST deja un margen de 4 píxeles)
        resized_image = cv2.resize(cropped_image, (20, 20), interpolation=cv2.INTER_AREA)

        # Crear una imagen en blanco de 28x28 píxeles
        final_image = np.zeros((28, 28), dtype=np.uint8)

        # Colocar la imagen redimensionada en el centro de la imagen 28x28
        final_image[4:24, 4:24] = resized_image

        # Aplicar operaciones morfológicas para rellenar huecos y suavizar
        kernel = np.ones((2,2), np.uint8)
        final_image = cv2.morphologyEx(final_image, cv2.MORPH_CLOSE, kernel)
        final_image = cv2.GaussianBlur(final_image, (3, 3), 0)
        final_image = cv2.normalize(final_image, 0, 255, cv2.NORM_MINMAX)
        # Perform histogram equalization
        final_image = cv2.equalizeHist(final_image)

        return final_image

    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Invertir la imagen horizontalmente
            frame = cv2.flip(frame, 1)
            
            # Convertir a espacio de color RGB para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar la imagen para encontrar las manos
            result = self.hands.process(rgb_frame)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Obtener las coordenadas del índice del dedo
                    index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    h, w, _ = frame.shape
                    cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                    
                    # Dibujar el centro en el frame
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
                    self.trajectory.append((cx, cy))
                    
                    # Dibujar el rastro del dedo en el frame
                    smoothed_trajectory = self.smooth_trajectory(self.trajectory)
                    for i in range(1, len(smoothed_trajectory)):
                        if smoothed_trajectory[i - 1] is None or smoothed_trajectory[i] is None:
                            continue
                        cv2.line(frame, smoothed_trajectory[i - 1], smoothed_trajectory[i], (0, 255, 0), 2)
            
            # Mostrar el frame con el rastro del dedo
            cv2.imshow('Frame', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Si el usuario presiona 'p', procesar la trayectoria y predecir el dígito
            if key == ord('p'):
                if self.trajectory:
                    input_img = self.preprocess_trajectory(self.trajectory, frame.shape)
                    if input_img is not None:
                        input_img_for_model = input_img.reshape(1, -1)   # aplanar para el modelo SVM
                        print("tamaño de la imagen : ",input_img_for_model.shape)
                        print("vector image: ",input_img_for_model)
                        #input_img_for_model = scaler.transform(input_img_for_model)  # Estandarizar
                        digit = self.clf.predict(input_img_for_model)
                        print(f'Predicted Digit: {digit[0]}')
                        
                        # Guardar la imagen del rastro
                        trajectory_img = np.zeros((480, 640), dtype=np.uint8)
                        for (x, y) in self.trajectory:
                            cv2.circle(trajectory_img, (x, y), 5, 255, -1)
                        cv2.imwrite('trajectory.png', trajectory_img)
                        
                        # Mostrar la imagen del trazo
                        cv2.imshow('Trace Image', cv2.resize(trajectory_img, (280, 280), interpolation=cv2.INTER_AREA))

                        # Mostrar la imagen que se envía al modelo
                        cv2.imshow('Model Input Image', (input_img * 255).astype(np.uint8))  # Desnormalizar para mostrar

                        # Visualizar el preprocesamiento detallado
                        plt.figure(figsize=(5, 5))
                        plt.imshow(input_img.reshape(28, 28), cmap='gray')
                        plt.title(f'Predicted: {digit[0]}')
                        plt.axis('off')
                        plt.show()

                        self.trajectory = []
                    else:
                        print("Invalid trajectory, try again.")
            
            # Si el usuario presiona 'c', borrar la trayectoria y empezar de nuevo
            elif key == ord('c'):
                self.trajectory = []
                print("Trazo eliminado. Empieza de nuevo.")
            
            # Romper el bucle con la tecla 'q'
            elif key == ord('q'):
                break

        # Liberar la captura y cerrar las ventanas
        cap.release()
        cv2.destroyAllWindows()
