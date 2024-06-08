import os
import cv2
import numpy as np

def preprocess_images():
    data = []
    labels = []

    for digit in range(10):
        img_dir = f'digits/{digit}'
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            data.append(img.flatten())
            labels.append(digit)
    
    data = np.array(data)
    labels = np.array(labels)
    
    return data, labels

data, labels = preprocess_images()
