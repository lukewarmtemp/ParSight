import rclpy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from rclpy.node import Node
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy


qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)

class RealSense(Node):

    def __init__(self):
        super().__init__('realsense_subscriber')
        self.bridge = CvBridge()
        self.realsense_subscriber = self.create_subscription(Image, '/camera/fisheye1/image_raw', self.realsense_callback, qos_profile)
        self.get_logger().info('Subscribing to RealSense!')

    def realsense_callback(self, msg):
        print("recieved image")
        self.get_logger().info(f"Recieved Image: {msg.height}x{msg.width}, encoding={msg.encoding}")
        
        img_data = np.frombuffer(msg.data,dtype=np.uint8)
        img_reshaped = img_data.reshape(msg.height,msg.width,-1)
        cv2.imshow("Realsense Image", img_reshaped)
        cv2.waitKey(1)

def main(args = None):
    rclpy.init(args = args)
    node = RealSense()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
