################################################
# Descriptions
################################################


'''
node subscribes to camera data from rgb_camera topic
node publishes message with just the image
'''


################################################
# Imports and Setup
################################################


# ros imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# image related
import cv2
from cv_bridge import CvBridge

# other imports
import time
try:
    from queue import Queue
except ModuleNotFoundError:
    from Queue import Queue
import threading


################################################
# Class Nodes
################################################


# threaded frame reader to get frames from camera
class FrameReader(threading.Thread):

    # queues to store frames
    queues = []
    _running = True

    def __init__(self, camera, name):
        super().__init__()
        self.name = name
        self.camera = camera

    def run(self):
        while self._running:
            ret, frame = self.camera.read()
            if ret:
                # push all frames into the queue
                for queue in self.queues:
                    queue.put(frame)

    def addQueue(self, queue):
        self.queues.append(queue)

    def getFrame(self, timeout=None):
        queue = Queue(1)
        self.addQueue(queue)
        return queue.get(timeout=timeout)

    def stop(self):
        self._running = False


# ros2 camera node
class RGBCameraNode(Node):

    def __init__(self):
        super().__init__('rgb_camera_node')
        # initiate the node
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 1)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.publish_frame)  # 10 Hz (We can go up to 120 Fps)
        self.cap = cv2.VideoCapture(0)
        # self.cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        # check for camera starting
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera!")
            raise RuntimeError("Failed to open camera!")
        self.frame_reader = FrameReader(self.cap, "FrameReader")
        self.frame_reader.start()

    def publish_frame(self):
        # publishes frames as ros2 messages
        frame = self.frame_reader.getFrame()
        if frame is not None:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)

    def stop(self):
        # stop frame reader and release camera
        self.frame_reader.stop()
        self.cap.release()


# gstreamer pipeline for nvidia jetson cameras
def gstreamer_pipeline(capture_width=1280, capture_height=720,
                       display_width=640, display_height=360,
                       framerate=30, flip_method=0):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


################################################
# Main
################################################


def main(args=None):
    rclpy.init(args=args)
    node = RGBCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down camera node.')
        node.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


################################################
# END
################################################