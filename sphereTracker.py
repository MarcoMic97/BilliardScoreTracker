import cv2
import torch
import numpy as np

def detect_sphere(frame):
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the color range for detecting a red sphere
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    
    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    
    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # If any contours were found
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the radius and center of the minimum enclosing circle around the largest contour
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        
        # Only proceed if the radius meets a minimum size
        if radius > 10:
            return (int(x), int(y)), int(radius)
    
    return None, None

def track_sphere():
    # Open the webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detect the sphere in the frame
        center, radius = detect_sphere(frame)
        
        if center and radius:
            # Draw the circle around the detected sphere
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Sphere Tracking', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Start tracking the sphere
track_sphere()

