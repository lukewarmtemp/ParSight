import cv2
import numpy as np 

cap = cv2.VideoCapture(0)  # Replace 0 with the correct device ID if necessary

if  print("Failed to open camera")
else:
    print("Camera opened successfully")
    ret, frame = cap.read()
    print(ret)
    if ret:
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)
    cap.release()not cap.isOpened():
   

