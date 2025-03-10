################################################
# DUMMY DATA FROM VIDEOS ONLY
################################################

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

def prep_data_as_if_real_time():
    frames, indices = [], []
    with open(os.path.join(output_folder, 'output.txt'), 'r') as f_out:
        for line in f_out:
            parts = line.strip().split()
            frame_filename = parts[0]
            x1_idx, y1_idx, x2_idx, y2_idx = map(int, parts[1:])
            frame = Image.open(os.path.join(output_folder, frame_filename))
            frame = frame.convert("RGB")
            frames.append(np.array(frame))
            indices.append([x1_idx, y1_idx, x2_idx, y2_idx])
    return np.array(frames), np.array(indices)


output_folder = '/Users/felicialiu/Desktop/DEV/ROB498_DroneCapstone/segmentation/task4'
frames, bboxes = prep_data_as_if_real_time()

print("Num Frames:", len(frames))
print("Size of Frame:", frames[0].shape)
print("Num Bounding Boxes:", len(bboxes))
print(frames.shape, bboxes.shape)


################################################
# DUMMY DATA FROM VIDEOS ONLY
################################################


def plot_data(time_steps, distances, offsets_x, offsets_y):
    
    # Plot 1: Distance over time
    plt.figure()
    plt.plot(time_steps, distances, marker='o', markersize=3)
    plt.title("Distance to Golf Ball Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Distance (m)")
    plt.grid(True)
    plt.show()

    # Plot 2: 3D Offset Trajectory over Time
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(offsets_x, offsets_y, time_steps, marker='o', markersize=3)
    ax.set_title("3D Offset Trajectory Over Time")
    ax.set_xlabel("Offset X (m)")
    ax.set_ylabel("Offset Y (m)")
    ax.set_zlabel("Time Step")
    x_min, x_max = min(offsets_x), max(offsets_x)
    y_min, y_max = min(offsets_y), max(offsets_y)
    mid_x = (x_max + x_min) / 2.0
    mid_y = (y_max + y_min) / 2.0
    range_x = (x_max - x_min) / 2.0
    range_y = (y_max - y_min) / 2.0
    max_range = max(range_x, range_y)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    plt.show()

    # Plot 3: 3D Offset and Depth Trajectory
    offsets_x = np.array(offsets_x)
    offsets_y = np.array(offsets_y)
    distances = np.array(distances)
    time_steps = np.arange(len(offsets_x))  
    distances = -distances 
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(time_steps))) 
    max_range = max(offsets_x.ptp(), offsets_y.ptp(), distances.ptp()) / 2.0
    mid_x = (offsets_x.max() + offsets_x.min()) / 2.0
    mid_y = (offsets_y.max() + offsets_y.min()) / 2.0
    mid_z = (distances.max() + distances.min()) / 2.0
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(offsets_x) - 1):
        ax.plot([offsets_x[i], offsets_x[i+1]], 
                [offsets_y[i], offsets_y[i+1]], 
                [distances[i], distances[i+1]], 
                color=colors[i], linewidth=2)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_title("3D Offset and Depth Trajectory")
    ax.set_xlabel("Offset X (m)")
    ax.set_ylabel("Offset Y (m)")
    ax.set_zlabel("Depth (m) (Flipped)")
    plt.show()


################################################
# MAIN
################################################

class ParSight:

    def __init__(self, frames, bboxes):

        ################################################
        # DUMMY DATA

        self.frames = frames
        self.bboxes = bboxes
        self.time = 0

        ################################################
        # CAMERA PARAMETERS

        self.frame_height, self.frame_width, _ = self.frames[0].shape
        self.camera_frame_center = self.frame_width / 2, self.frame_height / 2

        self.REAL_DIAMETER_MM = 42.67  # Standard golf ball diameter in mm
        self.FOCAL_LENGTH_MM = 26      # iPhone 14 Plus main camera focal length in mm
        self.SENSOR_WIDTH_MM = 7.6     # Approximate sensor width in mm
        self.DOWN_SAMPLE_FACTOR = 4    # Downsample factor used in YOLOv5 model
        self.FOCAL_LENGTH_PIXELS = ((self.FOCAL_LENGTH_MM / self.SENSOR_WIDTH_MM) * self.frame_width) / self.DOWN_SAMPLE_FACTOR


    ################################################
    # CONVERSION FUNCTIONS
    ################################################

    def pixels_to_meters(self, pixel_offset, distance_mm):
        # Compute displacement in mm, then convert to meters
        offset_mm = (distance_mm / self.FOCAL_LENGTH_PIXELS) * pixel_offset
        return offset_mm / 1000  # Convert mm to meters

    def meters_to_pixels(self, offset_m, distance_mm):
        # Convert offset to mm, then compute pixel displacement
        offset_mm = offset_m * 1000
        return (offset_mm * self.FOCAL_LENGTH_PIXELS) / distance_mm

    def calculate_golf_ball_metrics(self, bbox):
        # unpack all the values from the bounding box and calculate the diameter
        x1, y1, x2, y2 = bbox
        bbox_width, bbox_height = x2 - x1, y2 - y1
        # if the ball is cut off on the edges, choose the larger dimension
        if x1 <= 0 or y1 <= 0 or x2 >= self.frame_width or y2 >= self.frame_height: diameter_pixels = max(bbox_width, bbox_height)
        else: diameter_pixels = (bbox_width + bbox_height) / 2
        ball_center_x, ball_center_y = x1 + bbox_width / 2, y1 + bbox_height / 2
        image_center_x, image_center_y = self.camera_frame_center
        # compute the distance to the ball
        distance_mm = (self.REAL_DIAMETER_MM * self.FOCAL_LENGTH_PIXELS) / diameter_pixels
        distance_m = distance_mm / 1000 
        # calculate ball and image centers (in pixel coordinates)
        offset_x_pixels, offset_y_pixels = ball_center_x - image_center_x, ball_center_y - image_center_y
        offset_x_m = self.pixels_to_meters(offset_x_pixels, distance_mm)
        offset_y_m = self.pixels_to_meters(offset_y_pixels, distance_mm)
        return distance_m, (offset_x_m, offset_y_m)


    ################################################
    # FUNCTIONS TO BE CALLED IN CONTROL LOOP 
    ################################################
    
    def rgb_camera_callback(self):
        # calls the camera and returns the frame
        return self.frames[self.time]

    def yolo_inference(self, frame):
        # calls the YOLO NN and returns the bounding box
        bbox = self.bboxes[self.time]
        # convert to center point
        return bbox, self.get_bbox_center(bbox)

    def get_bbox_center(self, bbox):
        # convert the bounding box to the center point
        x1, y1, x2, y2 = bbox
        return (x1 + x2) // 2, (y1 + y2) // 2
    
    def compute_distances(self, bbox):
        # first we calculate how far the center is from the camera frame center
        distance_m, (offset_x_m, offset_y_m) = self.calculate_golf_ball_metrics(bbox)
        return distance_m, offset_x_m, offset_y_m
        

    ################################################
    # CONTROL LOOP 
    ################################################

    def control_loop(self):
        
        # Lists to record data for plotting
        distances = []
        offsets_x = []
        offsets_y = []
        time_steps = []

        while self.time < len(self.frames):

            # call the camera and get the frame
            frame = self.rgb_camera_callback()

            # run the YOLO NN to get the bounding box
            bbox, center = self.yolo_inference(frame)

            # move the drone to center the ball
            distance_m, offset_x_m, offset_y_m = self.compute_distances(bbox)

            # print with 4 decimals
            print(f"Time: {self.time}, Distance: {distance_m:.4f} m, Offset: ({offset_x_m:.4f}, {offset_y_m:.4f}) m")

            # Record the data for plotting
            distances.append(distance_m)
            offsets_x.append(offset_x_m)
            offsets_y.append(offset_y_m)
            time_steps.append(self.time)
            # increment the time
            self.time += 1
        
        plot_data(time_steps, distances, offsets_x, offsets_y)




parsight_node = ParSight(frames, bboxes)
parsight_node.control_loop()