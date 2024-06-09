import cv2
import numpy as np
import joblib
import mediapipe as mp
import matplotlib.pyplot as plt

class RoiRecognition:
    def __init__(self):
        self.clf = joblib.load("svm_digit_classifier.pkl")

    def preprocess_image(self, image):
        inverted_roi = cv2.bitwise_not(image)
        normalized_image = cv2.normalize(inverted_roi, None, 0, 255, cv2.NORM_MINMAX)
        blurred_image = cv2.GaussianBlur(normalized_image, (3, 3), 0)
        _, binary_image = cv2.threshold(blurred_image, 128, 255, cv2.THRESH_BINARY)
        return self.preprocess_binary_image(binary_image)

    def preprocess_binary_image(self, image):
        x_coords, y_coords = np.where(image > 0)
        if len(x_coords) == 0 or len(y_coords) == 0:
            return None
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        if x_min == x_max or y_min == y_max:
            return None
        margin = 20
        x_min = max(x_min - margin, 0)
        x_max = min(x_max + margin, image.shape[1])
        y_min = max(y_min - margin, 0)
        y_max = min(y_max + margin, image.shape[0])
        cropped_image = image[y_min:y_max, x_min:x_max]
        resized_image = cv2.resize(cropped_image, (20, 20), interpolation=cv2.INTER_AREA)
        final_image = np.zeros((28, 28), dtype=np.uint8)
        final_image[4:24, 4:24] = resized_image
        kernel = np.ones((2, 2), np.uint8)
        final_image = cv2.morphologyEx(final_image, cv2.MORPH_CLOSE, kernel)
        final_image = cv2.normalize(final_image, None, 0, 255, cv2.NORM_MINMAX)
        final_image = cv2.equalizeHist(final_image)
        _, final_image = cv2.threshold(final_image, 128, 255, cv2.THRESH_BINARY)
        return final_image

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            offset = 100
            center_region = (center_x - offset, center_y - offset, center_x + offset, center_y + offset)
            best_contour = None
            best_area = 0
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if (center_region[0] < x < center_region[2] and center_region[1] < y < center_region[3] and 
                    30 < w < 200 and 30 < h < 200 and area > best_area):
                    best_contour = cnt
                    best_area = area
            if best_contour is not None:
                x, y, w, h = cv2.boundingRect(best_contour)
                roi = gray[y:y+h, x:x+w]
                processed_digit = self.preprocess_image(roi)
                if processed_digit is not None:
                    cv2.imshow('Segmented and Preprocessed Image', cv2.resize(processed_digit, (280, 280), interpolation=cv2.INTER_AREA))
                    roi_digits = processed_digit.reshape((1, -1))
                    number_poly = self.clf.predict(roi_digits)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f'{int(number_poly)}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow('Frame with Digits and Trajectory', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
