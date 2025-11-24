import cv2

# Initialize the camera
cap = cv2.VideoCapture('http://10.0.0.107/cam-hi.jpg')  # 0 for default camera

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Capture a frame
ret, frame = cap.read()

if ret:
    # Display the captured frame
    cv2.imshow("Captured Image", frame)

    # Save the image
    cv2.imwrite("captured_image.jpg", frame)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error capturing frame")

# Release the camera
cap.release()