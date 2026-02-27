from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

import cv2
import numpy as np


class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        # Subscriber to camera images
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            10
        )

        # Publisher for marker pose
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            '/aruco_pose',
            10
        )

        # CvBridge for ROS <-> OpenCV conversion
        self.bridge = CvBridge()

        # Aruco dictionary and detector parameters
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()

        #loaded camera calibration data from external file
        calib_data = np.load("camera_calib.npz")

        self.camera_matrix = calib_data['camera_matrix']
        self.dist_coeffs = calib_data['dist_coeff']

        # Real-world marker size in meters, remember to change
        self.marker_length = 0.05

    def image_callback(self, msg):
        # Convert ROS compressed image to OpenCV BGR
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Detect Aruco markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame,
            self.dictionary,
            parameters=self.parameters
        )

        if ids is not None:
            # Draw markers on frame for visualization
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                half = self.marker_length / 2

                # Define 3D coordinates of marker corners
                obj_points = np.array([
                    [-half, half, 0],
                    [half, half, 0],
                    [half, -half, 0],
                    [-half, -half, 0]
                ], dtype=np.float32)

                # Solve PnP to get marker pose
                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    corners[i],
                    self.camera_matrix,
                    self.dist_coeffs
                )

                if success:
                    # Convert rotation vector to rotation matrix
                    rot_matrix, _ = cv2.Rodrigues(rvec)

                    # Convert rotation matrix to quaternion
                    quat = R.from_matrix(rot_matrix).as_quat()

                    # Create PoseStamped message
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = msg.header.stamp
                    pose_msg.header.frame_id = f"camera_link_marker_{int(ids[i][0])}"

                    # Translation (marker position in camera frame)
                    pose_msg.pose.position.x = float(tvec[0][0])
                    pose_msg.pose.position.y = float(tvec[1][0])
                    pose_msg.pose.position.z = float(tvec[2][0])

                    # Rotation (marker orientation in camera frame)
                    pose_msg.pose.orientation.x = quat[0]
                    pose_msg.pose.orientation.y = quat[1]
                    pose_msg.pose.orientation.z = quat[2]
                    pose_msg.pose.orientation.w = quat[3]

                    # Publish marker pose
                    self.pose_publisher.publish(pose_msg)

        # Optional: visualize
        #cv2.imshow("Aruco Detection", frame)
        #cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
