# import cv2

# # Open the camera, try using 0, 1, 2, etc.
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# if not cap.isOpened():
#     raise RuntimeError("Failed to open camera!")
# print("Camera is Open")
# # Capture a frame to verify the camera works
# ret, frame = cap.read()
# if ret:
#     print("Camera is working!")
# else:
#     print("Failed to capture frame")

# # Release the camera
# cap.release()

import sys
import cv2
import time
import numpy as np

def read_cam():
    cap = cv2.VideoCapture(0)#"nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert !  appsink")
    time.sleep(2)
    if cap.isOpened():
        cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
        while True:
            ret_val, img = cap.read()
            if ret_val == True:
                cv2.imshow('demo',img)
                cv2.waitKey(10)
            else:
                print('FAILED TO GET IMAGE')

    else:
     print("camera open failed")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    read_cam()