# Using YOLOv11 for detection with weights named 'best.pt'

# Import necessary libraries
from ultralytics import YOLO
import mss
import numpy as np
import cv2

model = YOLO('best.pt')  # Load a YOLO model with the specified weights

# redirecting the screen output to be used as input for the model
# Capture the screen using mss
sct = mss.mss()
monitor = sct.monitors[1]  # Select the primary monitor

def screen_capture():
    screenshot = sct.grab(monitor)  # Capture the screen
    frame = np.array(screenshot)  # Convert to numpy array
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert to RGB format
    return frame

# Process the screen frames in a loop
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)  # Create a window to display results
cv2.resizeWindow("YOLO Detection", 800, 600)  # Resize the window
time = cv2.getTickCount()  # Get the current time in ticks  # Initialize time variable for FPS calculation
while True:
    frame = screen_capture()  # Capture a frame from the screen
    results = model(frame)  # Run the model on the captured frame
    annotated_frame = results[0].plot()  # Annotate the frame with detection results
    # Annote frame with fps counter
    fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - time))  # Calculate FPS
    cv2.putText(annotated_frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)  # Add FPS text to the frame
    cv2.imshow("YOLO Detection", annotated_frame)  # Show the detection results in the single window
    time = cv2.getTickCount()  # Get the current time in ticks
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("YOLO Detection", cv2.WND_PROP_VISIBLE) < 1:  # Exit on 'q' key press or window close
        break

cv2.destroyAllWindows()  # Close the window when the loop ends
