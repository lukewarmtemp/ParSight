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


# ros imports
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

# custom image related ros imports
from custom_msgs.msg import ImageWithBboxConf  # custom!
from cv_bridge import CvBridge

# ros imports for realsense and mavros
from geometry_msgs.msg import PoseArray, PoseStamped, Point, Quaternion
from nav_msgs.msg import Odometry

# reliability imports
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)

# image related
import numpy as np


################################################
# Class Nodes
################################################


class DroneControlNode(Node):

    def __init__(self):
        super().__init__('drone_control_node') 

        ###############
        # SERVICE CALLS

        self.srv_launch = self.create_service(Trigger, 'rob498_drone_1/comm/launch', self.callback_launch)
        self.srv_test = self.create_service(Trigger, 'rob498_drone_1/comm/test', self.callback_test)
        self.srv_land = self.create_service(Trigger, 'rob498_drone_1/comm/land', self.callback_land)
        self.srv_abort = self.create_service(Trigger, 'rob498_drone_1/comm/abort', self.callback_abort)
        print('services created')

        #######################
        # MAVROS VARIABLE SETUP

        # generally how high above the ball we fly
        self.desired_flight_height = 0.5
        self.max_searching_height = 1.5

        # for vision_pose to know where it is
        self.position = Point()
        self.orientation = Quaternion()
        self.timestamp = None
        self.frame_id = "map"

        # for setpoint_vision to know where to go
        self.set_position = Point()
        self.set_orientation = Quaternion()
        self.set_orientation.w = -1.0

        ######################
        # IMAGE VARIABLE SETUP

        # init class attributes to store msg: image, bbox, confidence, and validity
        self.image_data = None
        self.bbox_data = None
        self.confidence_data = None
        self.valid_bbox = False
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

        ############################
        # SUBSCRIBER/PUBLISHER SETUP

        # subscriber to RealSense pose data
        self.realsense_subscriber = self.create_subscription(Odometry, '/camera/pose/sample', self.realsense_callback, qos_profile)
        self.get_logger().info('Subscribing to RealSense!')

        # publisher for VisionPose topic
        self.vision_pose_publisher = self.create_publisher(PoseStamped, '/mavros/vision_pose/pose', 1)
        self.get_logger().info('Publishing to VisionPose')

        # publisher for SetPoint topic
        self.setpoint_publisher = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', qos_profile)
        self.get_logger().info('Publishing to SetPoint')

        # subscriber to the message from YOLO segmentation node
        self.camera_subscriber = self.create_subscription(ImageWithBboxConf, '/camera/image_with_bbox_conf', self.image_bbox_conf_callback, 1)
        self.get_logger().info('Subscribed to ImageWithBboxConf messages!')

        # statement to end the inits
        self.get_logger().info('Nodes All Setup and Started!')

    ################################################
    # SERVICE CALLS 
    ################################################

    def callback_launch(self, request, response):
        print('Launch Requested. Drone takes off to find the golf ball and hover overtop.')
        self.launching_procedure()
        return response

    def callback_test(self, request, response):
        print('Test Requested. Drone is ready to follow whever the ball may go.')
        self.testing_procedure()
        return response
        
    def callback_land(self, request, response):
        print('Land Requested. Drone will return to starting position where the humans are.')
        self.landing_procedure()
        return response

    def callback_abort(self, request, response):
        print('Abort Requested. Drone will land immediately due to safety considerations.')
        self.set_position.z = 0.0
        response.success = True
        response.message = "Success"
        return response

    ################################################
    # SERVICE FUNCTIONS
    ################################################

    def launching_procedure(self):
        # start by taking off and flying higher to search for the ball
        # continuously search until a valid segmentation is found
        # once the ball is detected, lower the drone to the desired height
        # center the drone over the ball
        # capture the current position for landing
        # TODO
        return

    def testing_procedure(self):
        # set the drone to continuously hover and track the ball
        # TODO
        return

    def landing_procedure(self):
        # drone will land at the captured position (back where the people are)
        # also at a lower height
        # TODO
        return

    ################################################
    # CALLBACKS
    ################################################

    def realsense_callback(self, msg):
        # get the info
        self.position = msg.pose.pose.position
        self.orientation = msg.pose.pose.orientation
        self.timestamp = self.get_clock().now().to_msg()
        # frame conversion
        self.orientation.x *= -1
        self.orientation.y *= -1
        self.orientation.z *= -1
        self.orientation.w *= -1
        # print statements to know its running
        # print(f"Position: x={self.position.x}, y={self.position.y}, z={self.position.z}")
        # print(f"Orientation: x={self.orientation.x}, y={self.orientation.y}, z={self.orientation.z}, w={self.orientation.w}")
        # print(f"Timestamp: {self.timestamp.sec}.{self.timestamp.nanosec}")
        # print(f"Frame ID: {self.frame_id}")
        # WRITE BOTH IMMEDIATELY
        self.send_vision_pose()
        self.send_setpoint()
        self.test_loop()

    def send_vision_pose(self):
        # Create a new PoseStamped message to publish to vision_pose topic
        vision_pose_msg = PoseStamped()
        vision_pose_msg.header.stamp = self.timestamp
        vision_pose_msg.header.frame_id = self.frame_id
        vision_pose_msg.pose.position = self.position
        vision_pose_msg.pose.orientation = self.orientation
        # Publish the message to the /mavros/vision_pose/pose topic
        self.vision_pose_publisher.publish(vision_pose_msg)

    def send_setpoint(self):
        # Create a new PoseStamped message to publish to setpoint topic
        setpoint_msg = PoseStamped()
        setpoint_msg.header.stamp = self.timestamp
        setpoint_msg.header.frame_id = self.frame_id
        setpoint_msg.pose.position = self.set_position
        setpoint_msg.pose.orientation = self.set_orientation
        # Publish the message to the /mavros/setpoint_position/local topic
        self.setpoint_publisher.publish(setpoint_msg)

    def image_bbox_conf_callback(self, msg):
        # unload data from the message into class attributes
        self.image_data = self.imgmsg_to_numpy(msg.image)
        self.bbox_data = msg.bbox
        self.confidence_data = msg.confidence
        # check if the bounding box is valid
        if self.bbox_data != [-1, -1, -1, -1]: self.valid_bbox = True
        else: self.valid_bbox = False
        # these are just print statements
        self.get_logger().info(f"Received image with bbox: {self.bbox_data} and confidence: {self.confidence_data}")
        self.get_logger().info(f"Valid BBox: {self.valid_bbox}")
        ########################################################
        # then we go into any image processing
        self.full_image_processing()

    ################################################
    # IMAGE PROCESSING
    ################################################

    def full_image_processing(self):
        # the first time, we set up parameters
        if self.FOCAL_LENGTH_PIXELS is None: self.first_time_setup_image_parameters()
        # safety check in case the new bbox is bad, use last
        if self.valid_bbox: self.curr_bbox = self.bbox_data
        # then everytime we get the distances
        distance_m, offset_x_m, offset_y_m = self.calculate_golf_ball_metrics()
        # then based on how far off we are, instruct the drone's setpoint to move that much
        self.move_drone(distance_m, offset_x_m, offset_y_m)

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
        return distance_m, offset_x_m, offset_y_m

    def move_drone(self, offset_x_m, offset_y_m, distance_m):
        # change the drone position based on the offset
        self.set_position.x = self.position.x + offset_x_m
        self.set_position.y = self.position.y + offset_y_m
        self.set_position.z = self.desired_flight_height
        return

    ################################################
    # IMAGE PROCESSING HELPERS
    ################################################

    def imgmsg_to_numpy(self, ros_image):
        # helper to convert ROS image to numpy array
        bridge = CvBridge()
        return bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')

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


def main(args=None):
    rclpy.init(args=args)
    data_processing_node = DroneControlNode()
    try:
        rclpy.spin(data_processing_node)
    except KeyboardInterrupt:
        data_processing_node.get_logger().info('Shutting down data processing node.')
    finally:
        data_processing_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()


################################################
# END
################################################