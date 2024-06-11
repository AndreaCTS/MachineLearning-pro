# tests/test_trace_recognition.py

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.TraceRecognition import TraceRecognition

class TestTraceRecognition(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.trace_recognition = TraceRecognition()
    
    def test_smooth_trajectory(self):
        trajectory = [(0, 0), (10, 10), (20, 20)]
        smoothed = self.trace_recognition.smooth_trajectory(trajectory)
        self.assertEqual(len(smoothed), len(trajectory))
    
    def test_interpolate_trajectory(self):
        trajectory = [(0, 0), (10, 10)]
        interpolated = self.trace_recognition.interpolate_trajectory(trajectory)
        self.assertGreater(len(interpolated), len(trajectory))
    
    def test_preprocess_trajectory(self):
        trajectory = [(0, 0), (10, 10), (20, 20), (30, 30)]
        frame_shape = (480, 640)
        preprocessed = self.trace_recognition.preprocess_trajectory(trajectory, frame_shape)
        self.assertEqual(preprocessed.shape, (28, 28))

if __name__ == '__main__':
    unittest.main()
