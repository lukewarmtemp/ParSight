import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from ultralytics import YOLO

import signal
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
PARSIGHT_PATH = "./src/ParSight/ParSight"
model = YOLO(PARSIGHT_PATH + "/models/best.pt")  # Ensure correct path
bridge = CvBridge()

################################################
# PARSIGHT CLASS WITH CTRL+C HANDLING
################################################
class Segment(Node):
    def __init__(self):
        super().__init__('segmentation_node') 
        
        # ROS Subscriber and Publisher nodes
        self.camera_subscriber = self.create_subscription(Image, '/camera/image_raw', self.frame_input, 1)
        self.get_logger().info('Subscribed to Camera Input!')

        # Camera Parameters
        self.REAL_DIAMETER_MM = 42.67  # Standard golf ball diameter in mm
        self.FOCAL_LENGTH_MM = 26      # iPhone 14 Plus main camera focal length in mm
        self.SENSOR_WIDTH_MM = 4.93     # Approximate sensor size: 5.095 mm (H) × 4.930 mm (W)
        self.DOWN_SAMPLE_FACTOR = 4    # Downsample factor used in YOLO model

        # Frame Parameters (Updated in first frame)
        self.frame_width, self.frame_height = None, None
        self.camera_frame_center = None
        self.FOCAL_LENGTH_PIXELS = None

        # Tracking Data
        self.distances = []
        self.offsets_x = []
        self.offsets_y = []
        self.time_steps = []
        self.time = 0

    ################################################
    # HELPER FUNCTIONS
    ################################################
    def pixels_to_meters(self, pixel_offset, distance_mm):
        """ Converts pixel displacement to meters based on focal length. """
        offset_mm = (distance_mm / self.FOCAL_LENGTH_PIXELS) * pixel_offset
        return offset_mm / 1000  # Convert mm to meters

    def calculate_golf_ball_metrics(self, bbox):
        """ Compute distance and offset from camera center. """
        x1, y1, x2, y2 = bbox
        bbox_width, bbox_height = x2 - x1, y2 - y1

        # Handle edge cases where ball is partially out of frame
        if x1 <= 0 or y1 <= 0 or x2 >= self.frame_width or y2 >= self.frame_height:
            diameter_pixels = max(bbox_width, bbox_height)
        else:
            diameter_pixels = (bbox_width + bbox_height) / 2

        # Compute Ball Center
        ball_center_x, ball_center_y = x1 + bbox_width / 2, y1 + bbox_height / 2
        image_center_x, image_center_y = self.camera_frame_center

        # Compute Distance in mm and Convert to meters
        distance_mm = (self.REAL_DIAMETER_MM * self.FOCAL_LENGTH_PIXELS) / diameter_pixels
        distance_m = distance_mm / 1000  

        # Compute Offsets in Meters
        offset_x_pixels, offset_y_pixels = ball_center_x - image_center_x, ball_center_y - image_center_y
        offset_x_m = self.pixels_to_meters(offset_x_pixels, distance_mm)
        offset_y_m = self.pixels_to_meters(offset_y_pixels, distance_mm)

        return distance_m, offset_x_m, offset_y_m

    ################################################
    # CONTROL LOOP WITH YOLO DETECTION
    ################################################

    def frame_input(self,msg):
        # Update Frame Parameters (Only once)
        if self.frame_width is None:
            self.frame_height, self.frame_width, _ = msg.height, msg.width
            self.camera_frame_center = (self.frame_width / 2, self.frame_height / 2)
            self.FOCAL_LENGTH_PIXELS = ((self.FOCAL_LENGTH_MM / self.SENSOR_WIDTH_MM) * self.frame_width) / self.DOWN_SAMPLE_FACTOR
        
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.segmentation(frame)


    def segmentation(self, frame, model):
        """ Real-time YOLO detection"""
        # ✅ Run YOLO Detection
        results = model(frame, imgsz=640, conf=0.4)  # Apply confidence threshold

        # Extract Bounding Box
        for result in results:
            for det in result.boxes.data:
                x_min, y_min, x_max, y_max, conf, cls = det.tolist()
                conf = float(conf)

                # ✅ Filter low-confidence detections
                if conf < 0.4:
                    continue  

                # Convert Bounding Box to Integers
                bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]

                # Compute Distance & Offsets
                distance_m, offset_x_m, offset_y_m = self.calculate_golf_ball_metrics(bbox)

                # ✅ Store for Post-Processing Plots
                self.distances.append(distance_m)
                self.offsets_x.append(offset_x_m)
                self.offsets_y.append(offset_y_m)
                self.time_steps.append(self.time)
                self.time += 1

                # ✅ Draw Bounding Box and Distance
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                label = f"Conf: {conf:.2f} | Dist: {distance_m:.2f}m"
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # ✅ Show the Frame with Detections
        cv2.imshow("Golf Ball Detection", frame)
    


        # except KeyboardInterrupt:
        #     print("\n[INFO] Ctrl+C detected! Generating plots...")
        #     self.generate_plots()  # ✅ Call plotting function before exiting
        #     cap.release()
        #     cv2.destroyAllWindows()
        #     sys.exit(0)  # Ensure clean exit

    ################################################
    # GENERATE PLOTS AFTER TRACKING
    ################################################
    def generate_plots(self):
        """ Plot the distance and offsets after tracking ends. """
        
        # ✅ Distance Over Time
        plt.figure()
        plt.plot(self.time_steps, self.distances, marker='o', markersize=3, color='g')
        plt.title("Distance to Golf Ball Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Distance (m)")
        plt.grid(True)
        plt.show()

        # ✅ Offset X and Y Over Time
        plt.figure()
        plt.plot(self.time_steps, self.offsets_x, marker='o', markersize=3, color='b', label="Offset X")
        plt.plot(self.time_steps, self.offsets_y, marker='o', markersize=3, color='r', label="Offset Y")
        plt.title("Offset X & Y Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Offset (m)")
        plt.grid(True)
        plt.legend()
        plt.show()

        # ✅ 3D Offset and Distance Plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.offsets_x, self.offsets_y, self.distances, marker='o', markersize=3)
        ax.set_title("3D Offset and Distance Trajectory")
        ax.set_xlabel("Offset X (m)")
        ax.set_ylabel("Offset Y (m)")
        ax.set_zlabel("Distance (m)")
        plt.show()

################################################
# MAIN EXECUTION
################################################


def main(args = None ):
    rclpy.init(args=args)
    segmentation_node = Segment()   
    try:
        rclpy.spin(segmentation_node)
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected! Generating plots...")
        segmentation_node.generate_plots()  # ✅ Call plotting function before exiting
        node.get_logger().info('Shutting down segmentation node.')
        node.stop()
    finally:
        segmentation_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
    
    
