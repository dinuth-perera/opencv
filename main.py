import cv2
import numpy as np

def detect_pp_strap(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the HSV range for open colurs (PP strap)
    #
    lower_c = np.array([0, 50, 50])
    upper_c = np.array([30, 255, 255])
    
    # Mask out the strap color in the HSV image
    mask = cv2.inRange(hsv, lower_c, upper_c)
    
    # Apply morphological closing to fill small holes and gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    # Find contours in the mask (only external contours)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with maximum area
    best_contour = max(contours, key=cv2.contourArea, default=None)
    
    if best_contour is not None:
        # Draw a green contour around the detected pp strap
        cv2.drawContours(frame, [best_contour], -1, (0, 255, 0), 2)
        
        ## TODO: Implement quality test & image processing with training dataset
        
        return cv2.contourArea(best_contour), "Good"
    
    else:
        return 0, "Not found"

def main():
    # Open Android camera
    # cv2.VideoCapture(0) <= for defualt webcam
    cap = cv2.VideoCapture("/dev/video2")
    if not cap.isOpened():
        print("Error: Unable to open Android camera.")
        return
    
    while True:
        ret, frame = cap.read()
        ## TODO: keep needed caputre area
        
        if not ret:
            print("Error: Unable to capture frame.")
            break
        
        area, quality = detect_pp_strap(frame)
        
        # Display the output
        cv2.imshow('PP Strap Detection', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
