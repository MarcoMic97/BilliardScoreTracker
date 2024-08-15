import cv2
import numpy as np

def detect_spheres(frame):
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the color range for detecting white spheres
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    # Define the color range for detecting dark red spheres
    lower_dark_red = np.array([160, 100, 100])
    upper_dark_red = np.array([180, 255, 255])

    # Define the color range for detecting orange spheres
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([15, 255, 255])
    
    # Threshold the HSV image to get only white, dark red, and orange colors
    white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    dark_red_mask = cv2.inRange(hsv_frame, lower_dark_red, upper_dark_red)
    orange_mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)
    
    # Find contours for all masks
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dark_red_contours, _ = cv2.findContours(dark_red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Container for detected spheres
    detected_spheres = {'white': [], 'dark_red': [], 'orange': []}
    
    # Process white contours
    for contour in white_contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if radius > 10:  # Filter out small contours
            detected_spheres['white'].append((int(x), int(y), int(radius)))
    
    # Process dark red contours
    for contour in dark_red_contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if radius > 10:
            detected_spheres['dark_red'].append((int(x), int(y), int(radius)))
    
    # Process orange contours
    for contour in orange_contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if radius > 10:
            detected_spheres['orange'].append((int(x), int(y), int(radius)))
    
    return detected_spheres

def track_spheres():
    # Open the webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)
    
    score = 0
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detect the spheres in the frame
        detected_spheres = detect_spheres(frame)
        
        # Draw the detected white spheres
        for (x, y, radius) in detected_spheres['white']:
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)  # Green circle for white spheres
        
        # Draw the detected dark red sphere
        for (x, y, radius) in detected_spheres['dark_red']:
            cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)  # Red circle for dark red spheres

        # Draw the detected orange spheres and check for collisions
        if len(detected_spheres['orange']) == 2:
            (x1, y1, r1) = detected_spheres['orange'][0]
            (x2, y2, r2) = detected_spheres['orange'][1]
            
            # Draw the orange spheres
            cv2.circle(frame, (x1, y1), r1, (0, 165, 255), 2)  # Orange circle for first orange sphere
            cv2.circle(frame, (x2, y2), r2, (0, 165, 255), 2)  # Orange circle for second orange sphere
            
            # Calculate the distance between the centers of the two orange balls
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            # Check if the balls are touching
            if distance <= r1 + r2:
                score += 1
                print(f"Score: {score}")
        
        # Display the score on the frame
        cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('Billiard Ball Tracking', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Start tracking the spheres
track_spheres()
