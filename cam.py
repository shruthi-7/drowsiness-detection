import cv2

# Try to open camera (adjust index if needed)
cap = cv2.VideoCapture(0)  # Default is usually 0, not 1

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame!")
            break

        cv2.imshow("Test", frame)
        if cv2.waitKey(1) == 27:  # ESC to exit
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Ensures window closes cleanly