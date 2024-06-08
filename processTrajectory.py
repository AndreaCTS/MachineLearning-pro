import cv2
import numpy as np
import joblib

# Load the classifier
clf = joblib.load("svm_digit_classifier.pkl")
scaler = joblib.load("scaler_28x28.pkl")

# Iniciar la captura de video
cap = cv2.VideoCapture(0)
trajectory = []

def preprocess_trajectory(trajectory, frame_shape):
    # Crear una imagen en blanco de 280x280 para mayor resolución
    image = np.zeros((280, 280), dtype=np.uint8)
    
    # Escalar la trayectoria a la imagen de 280x280
    for (x, y) in trajectory:
        x_bin = int(x * 280 / frame_shape[1])
        y_bin = int(y * 280 / frame_shape[0])
        if x_bin < 280 and y_bin < 280:
            image[y_bin, x_bin] = 255

    # Redimensionar la imagen a 28x28
    small_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    return small_image.reshape(1, -1) / 255.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir a espacio de color HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Definir rango de color para la detección (ajustar según el color del guante)
    lower_color = np.array([0, 100, 100])
    upper_color = np.array([10, 255, 255])
    
    # Crear una máscara
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Encontrar el contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Aproximar el contorno para simplificarlo
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Encontrar el punto más alto (y mínimo) como la punta del dedo
        topmost = tuple(approx[approx[:, :, 1].argmin()][0])
        cx, cy = topmost
        
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
            input_img = scaler.transform(input_img)  # Estandarizar


            #######Revisar que se pasa al modelo
            digit = clf.predict(input_img)
            print(f'Predicted Digit: {digit[0]}')
            
            # Guardar la imagen del rastro
            trajectory_img = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            for (x, y) in trajectory:
                cv2.circle(trajectory_img, (x, y), 5, 255, -1)
            cv2.imwrite('trajectory.png', trajectory_img)
            
            # Mostrar la imagen segmentada
            ###IMagen mal segmentada o si esta bien pero no se ve un carajo
            segment_img = input_img.reshape(28, 28) * 255
            cv2.imshow('Segmented Image', segment_img)
            trajectory = []
    
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
