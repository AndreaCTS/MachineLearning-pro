import sys
import cv2
from TraceRecognition import TraceRecognition
from RoiRecognition import RoiRecognition

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
