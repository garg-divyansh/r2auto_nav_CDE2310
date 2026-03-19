from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image         # raw camera image
from geometry_msgs.msg import PoseStamped, TransformStamped

import cv2
import numpy as np

import tf2_ros
import time

"""
[FIX] 
- Is this "camera_link" defined?

[Image sampling]
- Limit resolution to ~640x480 instead of full res (8MP, 3840x2160)
- Set camera data format to NV21 (YUV420):
    - Full range luminance (8-bit)
    - 1:2 subsampled chroma U/V in both horizontal & vertical axis
    - Represented in 2D array of uint8_t: 
        - First h*w bytes: Y-plane
        - Other h*w/2 bytes: interleaved VU chroma
- Directly extract y-plane luma and reshape into 8-bit grayscale, discarding U/V values 
  instead of converting to color
- Aruco detectors work better with grayscale images due to reliance on edge detection

[Callback routine]
- Callback only extracts grayscale data & caches latest image frame in local buffer

[Aruco detection]
- Runs at fixed frequency, currently set to 20hz
- Publishes TF transform of aruco markers ("aruco_marker_{id}") relative to "camera_link"
- Ignored rotation, only publishes translation
"""

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        # Command line variables
        self.declare_parameter('verbouse', False)
        self.declare_parameter('frequency', 20)
        self.declare_parameter('marker_size', 0.05)
        self.declare_parameter('benchmark', False)
        self.verbouse = self.get_parameter('verbouse').get_parameter_value().bool_value
        self.update_frequency = self.get_parameter('frequency').get_parameter_value().integer_value
        self.marker_length = self.get_parameter('marker_size').get_parameter_value().double_value
        self.benchmark = self.get_parameter('benchmark').get_parameter_value().bool_value

        # QoS profile for camera images
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscriber to raw camera images
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile
        )

        # Callback timer for aruco detection 
        self.detect_timer = self.create_timer(
            1.0 / self.update_frequency,
            self.locate_aruco
        )

        # tf2 broadcaster for marker transforms
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Aruco dictionary and detector parameters
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters_create()

        # Load camera calibration data
        # calib_data = np.load("/home/grp5/turtlebot3_ws/src/aruco_detector/aruco_detector/camera_calib.npz")
        self.camera_matrix = np.array([
            [1292.7630777016007, 0.0, 312.5025852215214],
            [0.0, 1293.1031123142184, 241.62988786682297],
            [0.0, 0.0, 1.0]], 
            dtype=np.float64)
        
        self.dist_coeffs = np.array(
            [0.09054834883546267, 0.6555785199323255, 0.0003356103392188726, -0.0017633731422733664, 0.0], 
            dtype=np.float64)

        # Frame buffer & msg stamp
        self.frame_buffer = None
        self.msg_stamp = None

        # 3D coordinates of marker corners
        half = self.marker_length / 2
        self.obj_points = np.array([
            [-half, half, 0],
            [half, half, 0],
            [half, -half, 0],
            [-half, -half, 0]
        ], dtype=np.float32)

    """
    Callback function
    Extracts & caches y-plane from NV21 img msg
    [!Warn] Raises exception & sets cache as None if extraction fails
    """
    def image_callback(self, msg):
        try:
            # Extract y-plane luma values & cache frame, stamp
            self.frame_buffer = np.frombuffer(msg.data, dtype=np.uint8)[:msg.height * msg.width].reshape(msg.height, msg.width)
            self.msg_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().warn(f"Grayscale extraction failed: {e}")
            self.frame_buffer = None
            self.msg_stamp = None
            return

    """
    Attempt to find aruco markers

    If arucos found (ids != None): 
        Compute & publish TF transform of aruco_marker_id -> camera_link
    Otherwise:
        Does nothing
    """
    def locate_aruco(self):
        if self.frame_buffer is None or self.msg_stamp is None:
            return
        
        # Snapshot of frame & stamp
        frame = self.frame_buffer
        stamp = self.msg_stamp

        # Detect Aruco markers
        if self.benchmark:
            timestamp = time.perf_counter()
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame,
            self.dictionary,
            parameters=self.parameters
        )

        if self.benchmark:
            aruco_detect_time = time.perf_counter() - timestamp

        if ids is not None:
            for i in range(len(ids)):
                # Solve PnP for marker pose
                if self.benchmark:
                    timestamp = time.perf_counter()

                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points,
                    corners[i],
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if self.benchmark:
                    pnp_time = time.perf_counter() - timestamp
                if success:
                    # Convert rotation vector to rotation matrix
                    # rot_matrix, _ = cv2.Rodrigues(rvec)

                    # Convert rotation matrix to quaternion
                    # quat = R.from_matrix(rot_matrix).as_quat()

                    # Create TransformStamped
                    t = TransformStamped()

                    t.header.stamp = stamp
                    t.header.frame_id = "camera_link"   # parent frame
                    t.child_frame_id = f"aruco_marker_{int(ids[i][0])}"

                    # Translation
                    # Transform camera coord. sys to match turtlebot
                    t.transform.translation.x = float(tvec[2][0])
                    t.transform.translation.y = float(tvec[0][0])
                    t.transform.translation.z = -float(tvec[1][0])

                    # Rotation
                    # t.transform.rotation.x = quat[0]
                    # t.transform.rotation.y = quat[1]
                    # t.transform.rotation.z = quat[2]
                    # t.transform.rotation.w = quat[3]

                    # Broadcast
                    self.tf_broadcaster.sendTransform(t)

                    # Log info if verbouse
                    if self.verbouse:
                        self.get_logger().info(f'Marker ID:{ids[i][0]} found, x:{float(tvec[0][0])}, y:{float(tvec[1][0])}, z{float(tvec[2][0])}')
                    if self.benchmark:
                        self.get_logger().info(f"Detection ms: {aruco_detect_time * 1000}, solvePnP ms: {pnp_time * 1000}")
                elif self.verbouse:
                    self.get_logger().info(f"Marker ID: {ids[i][0]} found, solvePnP failed.")

        elif self.verbouse:
            self.get_logger().info('No marker found.')

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()