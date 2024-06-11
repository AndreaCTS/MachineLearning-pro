import unittest
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.RoiRecognition import RoiRecognition
from model.TraceRecognition import TraceRecognition


class TestRecognition(unittest.TestCase):
    def setUp(self):
        self.roi_recognition = RoiRecognition()
        self.trace_recognition = TraceRecognition()

    def test_roi_recognition(self):
        # Lista de imágenes de prueba
        test_images = []
        for digit in ['7','0']:
            test_image = np.ones((200, 200), dtype=np.uint8) * 255  # Fondo blanco más grande
            cv2.putText(test_image, digit, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 10)  # Dígito más grande en negro
            test_images.append((digit, test_image))

        for digit, test_image in test_images:
            # Mostrar la imagen de prueba
            cv2.imshow(f"Test Image (ROI) - {digit}", test_image)
            cv2.imwrite(f"test_image_roi_{digit}.png", test_image)
            cv2.waitKey(0)

            # Predecir el dígito utilizando la clase RoiRecognition
            predicted_digit = self.roi_recognition.predict(test_image)
            self.assertIsNotNone(predicted_digit)
            print(f'Predicted digit (ROI) for {digit}: {predicted_digit}')



    def test_trace_recognition(self):
        # Crear una trayectoria de prueba
        test_trajectory = [
            (30, 20), (35, 20), (40, 20), (45, 20), (50, 20), (55, 20), (60, 20),
            (60, 25), (60, 30), (55, 35), (50, 40), (45, 45), (40, 50), (35, 55), (30, 60)
        ]

        # Procesar la trayectoria
        processed_image = self.trace_recognition.preprocess_trajectory(test_trajectory, (480, 640))
        self.assertIsNotNone(processed_image, "Preprocessed image is None")

        # Mostrar la imagen procesada
        if processed_image is not None:
            cv2.imshow("Processed Image (Trace)", processed_image)
            cv2.imwrite("processed_image_trace.png", processed_image)
            cv2.waitKey(0)

        # Predecir el dígito utilizando la clase TraceRecognition
        predicted_digit = self.trace_recognition.predict(test_trajectory, (480, 640))
        self.assertIsNotNone(predicted_digit, "Predicted digit is None")
        print(f'Predicted digit (Trace): {predicted_digit}')


if __name__ == '__main__':
    unittest.main()
