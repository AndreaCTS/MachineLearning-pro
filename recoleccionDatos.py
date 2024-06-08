import cv2
import numpy as np
import os

# Crear directorios para guardar las imágenes de los dígitos
for i in range(10):
    os.makedirs(f'digits/{i}', exist_ok=True)

def capture_digits():
    cap = cv2.VideoCapture(0)
    current_digit = 0
    img_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir a escala de grises y aplicar desenfoque
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                roi = thresh[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                cv2.imshow('Digit', roi_resized)
                
                # Guardar la imagen con etiqueta actual
                img_name = f"digits/{current_digit}/digit_{img_counter}.png"
                cv2.imwrite(img_name, roi_resized)
                img_counter += 1
        
        cv2.imshow('Frame', frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('n'):
            current_digit = (current_digit + 1) % 10
            print(f"Capturando dígitos: {current_digit}")

    cap.release()
    cv2.destroyAllWindows()

capture_digits()
