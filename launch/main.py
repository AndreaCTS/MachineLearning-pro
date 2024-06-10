import sys
import cv2
import os
# Agrega la ruta de la carpeta padre al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.TraceRecognition import TraceRecognition
from model.RoiRecognition import RoiRecognition

# Asegura la ruta del módulo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    while True:
        print("Select one mode:")
        print("1. Trace Recognition")
        print("2. ROI Recognition")
        print("q. Quit")
        choice = input("Enter your choice: ")
        
        if choice == '1':
            trace_recognition = TraceRecognition()
            trace_recognition.run()
        elif choice == '2':
            roi_recognition = RoiRecognition()
            roi_recognition.run()
        elif choice == 'q':
            print("Exiting...")
            sys.exit()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
