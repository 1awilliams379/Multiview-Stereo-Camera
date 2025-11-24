import cv2
cap = cv2.VideoCapture(0)  # Try 0, 1, 2, etc.
if not cap.isOpened():
    print("Error: Could not open camera.")