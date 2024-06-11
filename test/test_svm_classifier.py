# tests/test_svm_classifier.py

import unittest
import numpy as np
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graficas.modeloOptimizado import SVMClassifier

class TestSVMClassifier(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.svm_classifier = SVMClassifier(class_of_interest=1)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = cls.svm_classifier.load_mnist_data()
        cls.X_train_reduced, cls.X_test_reduced = cls.svm_classifier.preprocess_data(cls.X_train, cls.X_test)
        cls.y_train_binary = cls.svm_classifier.train_model(cls.X_train_reduced, cls.y_train)
    
    def test_load_mnist_data(self):
        self.assertEqual(self.X_train.shape, (60000, 784))
        self.assertEqual(self.X_test.shape, (10000, 784))
        self.assertEqual(self.y_train.shape, (60000,))
        self.assertEqual(self.y_test.shape, (10000,))
    
    def test_preprocess_data(self):
        self.assertEqual(self.X_train_reduced.shape, (60000, 3))
        self.assertEqual(self.X_test_reduced.shape, (10000, 3))
    
    def test_train_model(self):
        unique_labels = np.unique(self.y_train_binary)
        self.assertEqual(len(unique_labels), 2)
        self.assertTrue(0 in unique_labels and 1 in unique_labels)
    
    def test_evaluate_model(self):
        self.svm_classifier.evaluate_model(self.X_test_reduced, self.y_test)
    
    def test_plot_soft_margin(self):
        self.svm_classifier.plot_soft_margin(self.X_train_reduced, self.y_train_binary)
    
    def test_plot_decision_boundary_3D(self):
        self.svm_classifier.plot_decision_boundary_3D(self.X_train_reduced)

if __name__ == '__main__':
    unittest.main()
