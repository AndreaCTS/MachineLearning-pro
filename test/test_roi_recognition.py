# tests/test_roi_recognition.py

import unittest
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.RoiRecognition import RoiRecognition

class TestRoiRecognition(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.roi_recognition = RoiRecognition()
    
    def test_preprocess_image(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.putText(image, '5', (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255), 5)
        preprocessed = self.roi_recognition.preprocess_image(image)
        self.assertEqual(preprocessed.shape, (28, 28))
    
    def test_preprocess_binary_image(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.putText(image, '5', (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255), 5)
        preprocessed = self.roi_recognition.preprocess_binary_image(image)
        self.assertEqual(preprocessed.shape, (28, 28))

if __name__ == '__main__':
    unittest.main()
