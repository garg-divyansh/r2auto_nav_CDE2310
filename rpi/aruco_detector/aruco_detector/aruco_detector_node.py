from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image         # raw camera image
from geometry_msgs.msg import PoseStamped, TransformStamped

import cv2
import numpy as np

import tf2_ros


class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')

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

        # Publisher for marker poses
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            '/aruco_pose',
            10
        )

        #tf2 broadcaster for marker transforms
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Aruco dictionary and detector parameters
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters_create()

        # Load camera calibration data
        calib_data = np.load("/home/grp5/turtlebot3_ws/src/aruco_detector/aruco_detector/camera_calib.npz")
        self.camera_matrix = calib_data['camera_matrix']
        self.dist_coeffs = calib_data['dist_coeff']

        # Real-world marker size in meters
        self.marker_length = 0.05

    def image_callback(self, msg):
        try:
            # NV21 (YUV420) to BGR conversion
            height = msg.height
            width = msg.width
            yuv = np.frombuffer(msg.data, dtype=np.uint8).reshape((height * 3 // 2, width))
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
            return

        # Detect Aruco markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame,
            self.dictionary,
            parameters=self.parameters
        )

        if ids is not None:
            # Draw markers on frame for visualization (optional)
            # cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                half = self.marker_length / 2

                # 3D coordinates of marker corners
                obj_points = np.array([
                    [-half, half, 0],
                    [half, half, 0],
                    [half, -half, 0],
                    [-half, -half, 0]
                ], dtype=np.float32)

                # Solve PnP for marker pose
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

                    # Broadcast TF transform
                    # Create TransformStamped
                    t = TransformStamped()

                    t.header.stamp = msg.header.stamp
                    t.header.frame_id = "camera_link"   # parent frame
                    t.child_frame_id = f"aruco_marker_{int(ids[i][0])}"

                    # Translation
                    t.transform.translation.x = float(tvec[0][0])
                    t.transform.translation.y = float(tvec[1][0])
                    t.transform.translation.z = float(tvec[2][0])

                    # Rotation
                    t.transform.rotation.x = quat[0]
                    t.transform.rotation.y = quat[1]
                    t.transform.rotation.z = quat[2]
                    t.transform.rotation.w = quat[3]

                    # Broadcast
                    self.tf_broadcaster.sendTransform(t)

        # Optional: visualize (uncomment for debugging)
        # cv2.imshow("Aruco Detection", frame)
        # cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
