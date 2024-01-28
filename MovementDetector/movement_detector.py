import cv2
import pygame

def MovementDetector():
    
    #initial frame before any movement is detected
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return
    
    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while 1:
        ret, frame = cap.read()
        if not ret:
            return
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        #for detection of movement we find the difference between the prev frame and the current frame
        frame_diff = cv2.absdiff(gray, prev_gray)

        #when to detect the movement? we need a threshold
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        #find the outlines in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255,0), 2)
        
        cv2.imshow('MOVEMENT DETECTION', frame)

        prev_gray = gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
