# Using YOLOv11 for detection with weights named 'best.pt'

# Import necessary libraries
from ultralytics import YOLO
import mss
import numpy as np
import cv2

# Frame interval for model processing
frame_interval = 5
# Confidence threshold for displaying labels
confidence = 0.5

model = YOLO('best.pt')  # Load a YOLO model with the specified weights

# redirecting the screen output to be used as input for the model
# Capture the screen using mss
sct = mss.mss()
monitor = sct.monitors[1]  # Select the primary monitor

# color dictionnary for each class
colors = {
    0: (255, 0, 0),  # Class 0 color
    1: (0, 255, 0),  # Class 1 color
    2: (0, 0, 255),  # Class 2 color
    3: (255, 255, 0),  # Class 3 color
    4: (255, 0, 255),  # Class 4 color
    5: (0, 255, 255),  # Class 5 color
    6: (255, 128, 0),  # Class 6 color
    7: (128, 255, 0),  # Class 7 color
    8: (0, 128, 255),  # Class 8 color
    9: (128, 0, 255),  # Class 9 color
    10: (255, 0, 128),  # Class 10 color
    11: (0, 255, 128),  # Class 11 color
    12: (128, 128, 255),  # Class 12 color
    13: (255, 128, 128)}  # Class 13 color

def screen_capture():
    screenshot = sct.grab(monitor)  # Capture the screen
    frame = np.array(screenshot)  # Convert to numpy array
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert to RGB format
    return frame

# Process the screen frames in a loop
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)  # Create a window to display results
cv2.resizeWindow("YOLO Detection", 800, 600)  # Resize the window
time = cv2.getTickCount()  # Get the current time in ticks  # Initialize time variable for FPS calculation
frame = screen_capture()  # Capture the first frame
results = model(frame, save=False)
boxes = results[0].boxes  # Only keep the bounding boxes
i = 0
while True:
    frame = screen_capture()  # Capture a frame from the screen
    #results = model(frame)  # Run the model on the captured frame
    #annotated_frame = results[0].plot()  # Annotate the frame with detection results
    if i % frame_interval == 0:
        results = model(frame,save=False)  # Run the model on the captured frame
        boxes = results[0].boxes
    # Annotate the current frame with the boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates of the bounding box
        conf = box.conf[0]  # Get the confidence score
        if conf > confidence:
            cls = int(box.cls[0])  # Get the class index
            color = colors[cls]
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # Draw the bounding box
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Add label to the frame
    fps = int(cv2.getTickFrequency() / (cv2.getTickCount() - time))  # Calculate FPS
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)  # Add FPS text to the frame
    cv2.imshow("YOLO Detection", frame)  # Show the detection results in the single window
    time = cv2.getTickCount()  # Get the current time in ticks
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("YOLO Detection", cv2.WND_PROP_VISIBLE) < 1:  # Exit on 'q' key press or window close
        break

cv2.destroyAllWindows()  # Close the window when the loop ends
