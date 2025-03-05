# # MIT License
# # Copyright (c) 2019 JetsonHacks
# # See license
# # Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# # NVIDIA Jetson Nano Developer Kit using OpenCV
# # Drivers for the camera and OpenCV are included in the base image

# import cv2
# import time
# try:
#     from  Queue import  Queue
# except ModuleNotFoundError:
#     from  queue import  Queue

# import  threading
# import signal
# import sys


# # def signal_handler(sig, frame):
# #     print('You pressed Ctrl+C!')
# #     sys.exit(0)
# # signal.signal(signal.SIGINT, signal_handler)


# def gstreamer_pipeline(
#     capture_width=1280,
#     capture_height=720,
#     display_width=640,
#     display_height=360,
#     framerate=60,
#     flip_method=0,
# ):
#     return (
#         "nvarguscamerasrc ! "
#         "video/x-raw(memory:NVMM), "
#         "width=(int)%d, height=(int)%d, "
#         "format=(string)NV12, framerate=(fraction)%d/1 ! "
#         "nvvidconv flip-method=%d ! "
#         "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
#         "videoconvert ! "
#         "video/x-raw, format=(string)BGR ! appsink"
#         % (
#             capture_width,
#             capture_height,
#             framerate,
#             flip_method,
#             display_width,
#             display_height,
#         )
#     )

# class FrameReader(threading.Thread):
#     queues = []
#     _running = True
#     camera = None
#     def __init__(self, camera, name):
#         threading.Thread.__init__(self)
#         self.name = name
#         self.camera = camera
 
#     def run(self):
#         while self._running:
#             _, frame = self.camera.read()
#             while self.queues:
#                 queue = self.queues.pop()
#                 queue.put(frame)
    
#     def addQueue(self, queue):
#         self.queues.append(queue)

#     def getFrame(self, timeout = None):
#         queue = Queue(1)
#         self.addQueue(queue)
#         return queue.get(timeout = timeout)

#     def stop(self):
#         self._running = False

# class Previewer(threading.Thread):
#     window_name = "Arducam"
#     _running = True
#     camera = None
#     def __init__(self, camera, name):
#         threading.Thread.__init__(self)
#         self.name = name
#         self.camera = camera
    
#     def run(self):
#         self._running = True
#         while self._running:
#             cv2.imshow(self.window_name, self.camera.getFrame(2000))
#             keyCode = cv2.waitKey(16) & 0xFF
#         cv2.destroyWindow(self.window_name)

#     def start_preview(self):
#         self.start()
#     def stop_preview(self):
#         self._running = False

# class Camera(object):
#     frame_reader = None
#     cap = None
#     previewer = None

#     def __init__(self):
#         self.open_camera()

#     def open_camera(self):
#         self.cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
#         if not self.cap.isOpened():
#             raise RuntimeError("Failed to open camera!")
#         if self.frame_reader == None:
#             self.frame_reader = FrameReader(self.cap, "")
#             self.frame_reader.daemon = True
#             self.frame_reader.start()
#         self.previewer = Previewer(self.frame_reader, "")

#     def getFrame(self):
#         return self.frame_reader.getFrame()

#     def start_preview(self):
#         self.previewer.daemon = True
#         self.previewer.start_preview()

#     def stop_preview(self):
#         self.previewer.stop_preview()
#         self.previewer.join()
    
#     def close(self):
#         self.frame_reader.stop()
#         self.cap.release()

# if __name__ == "__main__":
#     camera = Camera()
#     camera.start_preview()
#     time.sleep(10)
#     camera.stop_preview()
#     camera.close()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
try:
    from queue import Queue
except ModuleNotFoundError:
    from Queue import Queue
import threading

class FrameReader(threading.Thread):
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

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.publish_frame)  # 10 Hz
        self.cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera!")
            raise RuntimeError("Failed to open camera!")
        self.frame_reader = FrameReader(self.cap, "FrameReader")
        self.frame_reader.start()

    def publish_frame(self):
        frame = self.frame_reader.getFrame()
        if frame is not None:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)

    def stop(self):
        self.frame_reader.stop()
        self.cap.release()


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=60,
    flip_method=0,
):
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


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
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
