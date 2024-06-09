import cv2
import numpy as np
import joblib
import mediapipe as mp
import matplotlib.pyplot as plt

# Load the classifier
clf = joblib.load("svm_digit_classifier.pkl")

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Iniciar la captura de video
cap = cv2.VideoCapture(0)
trajectory = []

# Variable de estado para el modo de predicción
mode = 'trace'  # 'trace' o 'roi'

def smooth_trajectory(trajectory, alpha=0.75):
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

def preprocess_trajectory(trajectory, frame_shape):
    # Crear una imagen en blanco de 640x480 para capturar la resolución completa
    image = np.zeros((480, 640), dtype=np.uint8)
    
    # Dibujar la trayectoria en la imagen
    for (x, y) in trajectory:
        cv2.circle(image, (x, y), 3, 255, -1)  # Reducir el tamaño del círculo a 3

    return preprocess_image(image)

def preprocess_image(image):
    # Encontrar los límites del trazo
    x_coords, y_coords = np.where(image > 0)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Asegurarse de que los límites son válidos
    if x_min == x_max or y_min == y_max:
        return None

    # Recortar la imagen al tamaño del trazo con un margen más amplio
    margin = 20
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
    kernel = np.ones((2, 2), np.uint8)
    final_image = cv2.morphologyEx(final_image, cv2.MORPH_CLOSE, kernel)
    
    # Normalización y ecualización del histograma
    final_image = cv2.normalize(final_image, None, 0, 255, cv2.NORM_MINMAX)
    final_image = cv2.equalizeHist(final_image)

    # Aplicar binarización para asegurar que todos los blancos sean iguales
    _, final_image = cv2.threshold(final_image, 128, 255, cv2.THRESH_BINARY)

    return final_image

def preprocess_roi_image(roi):
    # Invertir los colores de la imagen de ROI (fondo blanco y número negro)
    inverted_roi = cv2.bitwise_not(roi)
    
    # Normalizar la imagen
    normalized_image = cv2.normalize(inverted_roi, None, 0, 255, cv2.NORM_MINMAX)
    
    # Aplicar un filtro Gaussiano para suavizar la imagen y reducir el ruido
    blurred_image = cv2.GaussianBlur(normalized_image, (3, 3), 0)
    
    # Binarización
    _, binary_image = cv2.threshold(blurred_image, 128, 255, cv2.THRESH_BINARY)
    
    return preprocess_image(binary_image)

# Función para preprocesar y predecir en tiempo real
def preprocess_and_predict(frame, clf):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Aplicar un filtro Gaussiano para suavizar la imagen y reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    offset = 100  # Tamaño de la región central
    center_region = (center_x - offset, center_y - offset, center_x + offset, center_y + offset)

    best_contour = None
    best_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Verificar si el contorno está dentro de la región central y es el más grande
        if (center_region[0] < x < center_region[2] and center_region[1] < y < center_region[3] and 
            30 < w < 200 and 30 < h < 200 and area > best_area):
            best_contour = cnt
            best_area = area

    if best_contour is not None:
        x, y, w, h = cv2.boundingRect(best_contour)
        roi = gray[y:y+h, x:x+w]
        
        # Preprocesar la imagen de la ROI
        processed_digit = preprocess_roi_image(roi)

        if processed_digit is not None:
            # Mostrar la imagen segmentada y preprocesada en tamaño más grande
            cv2.imshow('Segmented and Preprocessed Image', cv2.resize(processed_digit, (280, 280), interpolation=cv2.INTER_AREA))

            # Visualizar el preprocesamiento detallado
            roi_digits = processed_digit.reshape((1, -1))
            number_poly = clf.predict(roi_digits)

            plt.figure(figsize=(5, 5))
            plt.imshow(processed_digit, cmap='gray')
            plt.title(f'Predicted with picture process: {number_poly}')
            plt.axis('off')
            plt.show()
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'{int(number_poly)}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Invertir la imagen horizontalmente
    frame = cv2.flip(frame, 1)
    
    # Convertir a espacio de color RGB para MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar la imagen para encontrar las manos
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Obtener las coordenadas del índice del dedo
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            
            # Dibujar el centro en el frame
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
            trajectory.append((cx, cy))
            
            # Dibujar el rastro del dedo en el frame
            smoothed_trajectory = smooth_trajectory(trajectory)
            for i in range(1, len(smoothed_trajectory)):
                if smoothed_trajectory[i - 1] is None or smoothed_trajectory[i] is None:
                    continue
                cv2.line(frame, smoothed_trajectory[i - 1], smoothed_trajectory[i], (0, 255, 0), 2)
    
    # Mostrar el frame con el rastro del dedo
    cv2.imshow('Frame', frame)
    
    key = cv2.waitKey(1) & 0xFF

    # Manejar el menú de selección
    if key == ord('m'):
        if mode == 'trace':
            mode = 'roi'
            print("Switched to ROI mode")
        else:
            mode = 'trace'
            print("Switched to Trace mode")
    
    # Si el usuario presiona 'p', procesar la trayectoria y predecir el dígito
    if key == ord('p'):
        if mode == 'trace':
            if trajectory:
                input_img = preprocess_trajectory(trajectory, frame.shape)
                if input_img is not None:
                    input_img_for_model = input_img.reshape(1, -1)   # aplanar para el modelo SVM
                    digit = clf.predict(input_img_for_model)
                    print(f'Predicted Digit: {digit[0]}')
                    
                    # Guardar la imagen del rastro
                    trajectory_img = np.zeros((480, 640), dtype=np.uint8)
                    for (x, y) in trajectory:
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

                    trajectory = []
                else:
                    print("Invalid trajectory, try again.")
        elif mode == 'roi':
            frame_with_digits = preprocess_and_predict(frame, clf)
            # Mostrar el frame con el rastro del dedo y los dígitos detectados y clasificados
            cv2.imshow('Frame with Digits and Trajectory', frame_with_digits)
    
    # Si el usuario presiona 'c', borrar la trayectoria y empezar de nuevo
    elif key == ord('c'):
        trajectory = []
        print("Trazo eliminado. Empieza de nuevo.")
    
    # Romper el bucle con la tecla 'q'
    elif key == ord('q'):
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
