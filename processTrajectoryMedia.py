import cv2
import numpy as np
import joblib
import mediapipe as mp
import matplotlib.pyplot as plt

# Load the classifier
clf = joblib.load("svm_digit_classifier.pkl")
scaler = joblib.load("scaler_28x28.pkl")

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Iniciar la captura de video
cap = cv2.VideoCapture(0)
trajectory = []

def preprocess_trajectory(trajectory, frame_shape):
    # Crear una imagen en blanco de 640x480 para capturar la resolución completa
    image = np.zeros((480, 640), dtype=np.uint8)
    
    # Dibujar la trayectoria en la imagen
    for (x, y) in trajectory:
        cv2.circle(image, (x, y), 5, 255, -1)

    # Encontrar los límites del trazo
    x_coords, y_coords = zip(*trajectory)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Asegurarse de que los límites son válidos
    if x_min == x_max or y_min == y_max:
        return None

    # Recortar la imagen al tamaño del trazo
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Redimensionar la imagen recortada a 20x20 píxeles (MNIST deja un margen de 4 píxeles)
    resized_image = cv2.resize(cropped_image, (20, 20), interpolation=cv2.INTER_AREA)

    # Crear una imagen en blanco de 28x28 píxeles
    final_image = np.zeros((28, 28), dtype=np.uint8)

    # Colocar la imagen redimensionada en el centro de la imagen 28x28
    final_image[4:24, 4:24] = resized_image

    # Normalizar la imagen
    normalized_image = final_image 

    return normalized_image

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
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            trajectory.append((cx, cy))
            
            # Dibujar el rastro del dedo en el frame
            for i in range(1, len(trajectory)):
                if trajectory[i - 1] is None or trajectory[i] is None:
                    continue
                cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)
    
    # Mostrar el frame con el rastro del dedo
    cv2.imshow('Frame', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Si el usuario presiona 'p', procesar la trayectoria y predecir el dígito
    if key == ord('p'):
        if trajectory:
            input_img = preprocess_trajectory(trajectory, frame.shape)
            if input_img is not None:
                input_img_for_model = input_img.reshape(1, -1)   # aplanar para el modelo SVM
                print("tamaño de la imagen : ",input_img_for_model.shape)
                print("vector image: ",input_img_for_model)
                #input_img_for_model = scaler.transform(input_img_for_model)  # Estandarizar
                #print("escalado: ",input_img_for_model)
                digit = clf.predict(input_img_for_model)
                print(f'Predicted Digit: {digit}')
                
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
