################################################
# Descriptions
################################################


'''
node subscribes to message with image and bbox sent by yolo_seg_node
node computes the distances and sends vision_pose and setpoint_position 
accordingly to move the amount required

** there may be issues with synch between realsense pose and rgb vision
** should be robust enough with some lag
** better at higher altitudes

1. maybe some timing thing (like frame id or that clock thing) needs to be leveraged to sync the realsense data and the segmentation data?
2. can use the confidence thing maybe if we change to pid to know how much to weigh
3. technically i don't think the frame needs to be passed over, but might be nice to have all the data here
4. this assignment thing on setpoint the directions might not be right (can also incorporate distance somehow, which doesn't have a around 0 operating point)

not done the actual landing testing and launching procedures yet but they are mapped out
'''


################################################
# Imports and Setup
################################################


# # ros imports
# import rclpy
# from rclpy.node import Node
# from std_srvs.srv import Trigger

# # custom image related ros imports
# from custom_msgs.msg import ImageWithBboxConf  # custom!
# from cv_bridge import CvBridge

# # ros imports for realsense and mavros
# from geometry_msgs.msg import PoseArray, PoseStamped, Point, Quaternion
# from nav_msgs.msg import Odometry

# # reliability imports
# from rclpy.qos import QoSProfile, QoSReliabilityPolicy
# qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)

# image related
import numpy as np

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import signal
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load YOLO Model
model = YOLO("C:/Users/Siddh/runs/detect/train6/weights/best.pt")  # Ensure correct path

#Open Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


################################################
# Class Nodes
################################################


class ADAPTED_DroneControlNode():

    def __init__(self):
        # super().__init__('drone_control_node') 

        # init class attributes to store msg: image, bbox, confidence, and validity
        self.image_data = None
        self.bbox_data = None
        self.confidence_data = None
        self.valid_bbox = False
        self.yes_bbox_got = False
        self.cut_looping = False
        self.curr_bbox = None

        # camera parameters
        self.REAL_DIAMETER_MM = 42.67  # Standard golf ball diameter in mm
        self.FOCAL_LENGTH_MM = 26      # iPhone 14 Plus main camera focal length in mm
        self.SENSOR_WIDTH_MM = 4.93     # Approximate sensor size: 5.095 mm (H) × 4.930 mm (W)
        self.DOWN_SAMPLE_FACTOR = 4    # Downsample factor used in YOLO model

        # frame parameters (updated in first frame)
        self.frame_width, self.frame_height = None, None
        self.camera_frame_center = None
        self.FOCAL_LENGTH_PIXELS = None



    def frame_input_callback(self):
        # convert ROS Image to opencv format
        ret, frame = cap.read()
        if not ret:
            # print("Error: Could not read frame.")
            self.cut_looping = True
        # frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # process the frame, then publish!
        best_bbox, frame, conf = self.run_yolo_segmentation(frame)
        # call other function
        self.image_data = frame
        self.bbox_data = best_bbox
        self.confidence_data = conf
        # check if the bounding box is valid
        if self.bbox_data != [-1, -1, -1, -1]: self.valid_bbox = True; self.yes_bbox_got = True
        else: self.valid_bbox = False
        # these are just print statements
        # print(f"Received image with bbox: {self.bbox_data} and confidence: {self.confidence_data}")
        # print(f"Valid BBox: {self.valid_bbox}")
        ########################################################
        # then we go into any image processing
        self.full_image_processing()
        return

    def run_yolo_segmentation(self, frame):
        # apply a confidence threshold
        results = model(frame, imgsz=640, conf=0.4, verbose=False)
        # initialize variables for the highest confidence detection
        best_conf, best_bbox = 0, None
        # cycle through all found bboxes
        for result in results:
            for det in result.boxes.data:
                # extract the bounding box
                x_min, y_min, x_max, y_max, conf, cls = det.tolist()
                conf = float(conf)
                # filter low confidence out
                if conf < 0: continue
                # check if this is the highest confidence so far
                if conf > best_conf:
                    best_conf = conf
                    best_bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
        # if no valid bounding box found, set it to [-1, -1, -1, -1]
        if best_bbox is None:
            best_bbox = [-1, -1, -1, -1]
            conf = -1
        
        return best_bbox, frame, conf



    ################################################
    # IMAGE PROCESSING
    ################################################

    def full_image_processing(self):
        # the first time, we set up parameters
        if self.FOCAL_LENGTH_PIXELS is None: self.first_time_setup_image_parameters()
        # safety check in case the new bbox is bad, use last
        if self.valid_bbox: 
            self.curr_bbox = self.bbox_data
        # then everytime we get the distances
        if self.yes_bbox_got:
            distance_m, offset_x_m, offset_y_m, offset_x_pixels, offset_y_pixels = self.calculate_golf_ball_metrics()
            # then based on how far off we are, instruct the drone's setpoint to move that much

            frame = self.image_data
            bbox = self.bbox_data
            conf = self.confidence_data

            # Draw Bounding Box and Distance
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            # offset_x_pixels = self.meters_to_pixels(offset_x_m, distance_m)    
            # offset_y_pixels = self.meters_to_pixels(offset_y_m, distance_m)
            label = f"Conf: {conf:.2f} | Dist: {offset_x_pixels:.1f} {offset_y_pixels:.1f}m"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Show the Frame with Detections
            cv2.imshow("Golf Ball Detection", frame)

            #  Exit Condition
            if cv2.waitKey(1) & 0xFF == ord('q'): self.cut_looping = True
            self.move_drone(offset_x_pixels, offset_y_pixels)

    def calculate_golf_ball_metrics(self):
        # unpack all the values from the bounding box and calculate the diameter
        x1, y1, x2, y2 = self.curr_bbox
        bbox_width, bbox_height = x2 - x1, y2 - y1
        # if the ball is cut off on the edges, choose the larger dimension
        if x1 <= 0 or y1 <= 0 or x2 >= self.frame_width or y2 >= self.frame_height: diameter_pixels = max(bbox_width, bbox_height)
        else: diameter_pixels = (bbox_width + bbox_height) / 2
        # compute the golf ball's center
        ball_center_x, ball_center_y = x1 + bbox_width / 2, y1 + bbox_height / 2
        image_center_x, image_center_y = self.camera_frame_center
        # compute the distance to the ball
        distance_mm = (self.REAL_DIAMETER_MM * self.FOCAL_LENGTH_PIXELS) / diameter_pixels
        distance_m = distance_mm / 1000 
        # calculate ball and image centers (in pixel coordinates)
        offset_x_pixels, offset_y_pixels = ball_center_x - image_center_x, ball_center_y - image_center_y
        offset_x_m = self.pixels_to_meters(offset_x_pixels, distance_m)
        offset_y_m = self.pixels_to_meters(offset_y_pixels, distance_m)
        return distance_m, offset_x_m, offset_y_m, offset_x_pixels, offset_y_pixels

    def move_drone(self, offset_x_pixels, offset_y_pixels,  tolerance=50):
        # If the drone is close enough to the center, just hover
        # offset_x_pixels = self.meters_to_pixels(offset_x_m, distance_m)    
        # offset_y_pixels = self.meters_to_pixels(offset_y_m, distance_m)
        if abs(offset_x_pixels) <= tolerance and abs(offset_y_pixels) <= tolerance:
            print("Move: Hovering")
            return

        if abs(offset_x_pixels) >= 4 * abs(offset_y_pixels):
            if offset_x_pixels > 0:
                print("Move: Right")
            else:
                print("Move: Left")
        elif abs(offset_y_pixels) >= 4 * abs(offset_x_pixels):
            if offset_y_pixels > 0:
                print("Move: Forward")
            else:
                print("Move: Backward")
        else:
            # Handles diagonal movements
            if offset_x_pixels > 0 and offset_y_pixels > 0:
                print("Move: Forward Right")
            elif offset_x_pixels > 0 and offset_y_pixels < 0:
                print("Move: Backward Right")
            elif offset_x_pixels < 0 and offset_y_pixels > 0:
                print("Move: Forward Left")
            elif offset_x_pixels < 0 and offset_y_pixels < 0:
                print("Move: Backward Left")


    ################################################
    # IMAGE PROCESSING HELPERS
    ################################################

    # def imgmsg_to_numpy(self, ros_image):
    #     # helper to convert ROS image to numpy array
    #     bridge = CvBridge()
    #     return bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')

    def first_time_setup_image_parameters(self):
        self.frame_height, self.frame_width, _ = self.image_data.shape
        self.camera_frame_center = (self.frame_width / 2, self.frame_height / 2)
        self.FOCAL_LENGTH_PIXELS = ((self.FOCAL_LENGTH_MM / self.SENSOR_WIDTH_MM) * self.frame_width) / self.DOWN_SAMPLE_FACTOR
        return

    def pixels_to_meters(self, pixel_offset, distance_m):
        # Compute displacement in mm, then convert to meters
        return (distance_m / self.FOCAL_LENGTH_PIXELS) * pixel_offset

    def meters_to_pixels(self, offset_m, distance_m):
        # Convert offset to mm, then compute pixel displacement
        return (offset_m * self.FOCAL_LENGTH_PIXELS) / distance_m



################################################
# MAIN EXECUTION
################################################



if __name__ == "__main__":
    parsight_node = ADAPTED_DroneControlNode()
    try:
        while True:
            parsight_node.frame_input_callback()
            if parsight_node.cut_looping:
                break
    except KeyboardInterrupt:
        # print("\n[INFO] Ctrl+C detected! Generating plots...")
        # self.generate_plots()  #  Call plotting function before exiting
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)  # Ensure clean exit

################################################
# END
################################################

