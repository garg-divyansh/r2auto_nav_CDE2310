#!/usr/bin/env python3
"""
aruco_dock_node.py

Docks a robot directly in front of an ArUco marker using TF lookups.

State machine:
  SEARCH          -> rotate in place until the marker TF appears
  ALIGN_HEADING   -> rotate to face the marker's front face
  APPROACH        -> drive to the goal point (d metres in front of marker)
  FINAL_ALIGN     -> fine-tune heading to face opposite the marker normal
  DONE            -> stop and hold

Coordinate conventions (camera_link / base_link):
  x = forward, y = left, z = up
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
import tf2_ros
import numpy as np
from scipy.spatial.transform import Rotation
from enum import Enum, auto


# ──────────────────────────────────────────────
# State machine states
# ──────────────────────────────────────────────
class State(Enum):
    SEARCH        = auto()
    ALIGN_HEADING = auto()
    APPROACH      = auto()
    FINAL_ALIGN   = auto()
    DONE          = auto()


# ──────────────────────────────────────────────
# Low-pass filter for 3-vectors
# ──────────────────────────────────────────────
class LowPassFilter:
    """Exponential moving average for numpy arrays."""

    def __init__(self, alpha: float):
        """
        alpha: smoothing factor in (0, 1].
               Lower  = more smoothing (slower response).
               Higher = less smoothing (faster response).
        """
        self.alpha = alpha
        self._value = None

    def update(self, new_value: np.ndarray) -> np.ndarray:
        if self._value is None:
            self._value = new_value.copy()
        else:
            self._value = self.alpha * new_value + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self):
        self._value = None

    @property
    def value(self):
        return self._value


# ──────────────────────────────────────────────
# Main node
# ──────────────────────────────────────────────
class ArucoDockNode(Node):

    def __init__(self):
        super().__init__('aruco_dock_node')

        # ── Parameters ────────────────────────────────────────────
        self.declare_parameter('marker_id',         0)
        self.declare_parameter('target_distance',   0.5)   # metres in front of marker
        self.declare_parameter('base_frame',        'base_link')
        self.declare_parameter('camera_frame',      'camera_link')

        # Controller gains
        self.declare_parameter('kp_linear',         0.4)
        self.declare_parameter('kp_angular',        1.2)

        # Speed limits
        self.declare_parameter('max_linear_vel',    0.3)   # m/s
        self.declare_parameter('max_angular_vel',   0.8)   # rad/s
        self.declare_parameter('min_linear_vel',    0.05)  # deadband threshold
        self.declare_parameter('min_angular_vel',   0.05)

        # Tolerances to declare success / advance state
        self.declare_parameter('distance_tol',      0.05)  # metres
        self.declare_parameter('heading_tol',       0.05)  # radians

        # Low-pass filter alpha (position and normal separately)
        self.declare_parameter('lpf_alpha',         0.35)

        # How long (s) to wait for a TF before giving up in SEARCH
        self.declare_parameter('tf_timeout',        0.1)

        self._load_params()

        # ── TF ────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Publisher ─────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # ── State ─────────────────────────────────────────────────
        self.state = State.SEARCH
        self.pos_filter    = LowPassFilter(self.lpf_alpha)
        self.normal_filter = LowPassFilter(self.lpf_alpha)

        # ── Control loop ──────────────────────────────────────────
        self.timer = self.create_timer(0.05, self._control_loop)   # 20 Hz

        self.get_logger().info(
            f'ArucoDockNode started — marker=aruco_marker_{self.marker_id}, '
            f'target_distance={self.target_distance:.2f} m'
        )

    # ──────────────────────────────────────────────────────────────
    # Parameter helpers
    # ──────────────────────────────────────────────────────────────
    def _load_params(self):
        g = self.get_parameter
        self.marker_id       = g('marker_id').value
        self.target_distance = g('target_distance').value
        self.base_frame      = g('base_frame').value
        self.camera_frame    = g('camera_frame').value
        self.kp_linear       = g('kp_linear').value
        self.kp_angular      = g('kp_angular').value
        self.max_lin         = g('max_linear_vel').value
        self.max_ang         = g('max_angular_vel').value
        self.min_lin         = g('min_linear_vel').value
        self.min_ang         = g('min_angular_vel').value
        self.dist_tol        = g('distance_tol').value
        self.head_tol        = g('heading_tol').value
        self.lpf_alpha       = g('lpf_alpha').value
        self.tf_timeout      = g('tf_timeout').value

    # ──────────────────────────────────────────────────────────────
    # TF lookup
    # ──────────────────────────────────────────────────────────────
    def _lookup_marker(self):
        """
        Look up the marker pose in base_frame.
        Returns (translation: np.array[3], rotation_matrix: np.array[3,3])
        or None if the TF is unavailable.
        """
        marker_frame = f'aruco_marker_{self.marker_id}'
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                marker_frame,
                rclpy.time.Time(),
                Duration(seconds=self.tf_timeout)
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None, None

        t = tf.transform.translation
        translation = np.array([t.x, t.y, t.z])

        q = tf.transform.rotation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
        rotation_matrix = rot.as_matrix()

        return translation, rotation_matrix

    # ──────────────────────────────────────────────────────────────
    # Marker normal (with ambiguity fix + low-pass)
    # ──────────────────────────────────────────────────────────────
    def _get_filtered_marker_data(self):
        """
        Returns (smoothed_translation, smoothed_normal) both as np.array[3],
        or (None, None) if the marker is not visible.
        """
        translation, R = self._lookup_marker()
        if translation is None:
            return None, None

        # The marker's Z-axis (column 2) is its outward normal in base_frame.
        raw_normal = R[:, 2].copy()

        # ── Ambiguity / flip fix ──────────────────────────────────
        # The marker normal should point *towards* the robot, which means
        # its x-component (forward axis of base_link) should be negative
        # (marker is in front of us, facing us).
        # If x > 0 the normal is pointing away — flip it.
        if raw_normal[0] > 0:
            raw_normal = -raw_normal

        # Additional flip-stability: if we already have a smoothed normal,
        # check dot product. A sudden sign flip appears as dot < 0.
        if self.normal_filter.value is not None:
            if np.dot(raw_normal, self.normal_filter.value) < 0:
                raw_normal = -raw_normal  # undo the flip for this frame

        smoothed_pos    = self.pos_filter.update(translation)
        smoothed_normal = self.normal_filter.update(raw_normal)
        # Re-normalise after smoothing
        norm = np.linalg.norm(smoothed_normal)
        if norm > 1e-6:
            smoothed_normal = smoothed_normal / norm

        return smoothed_pos, smoothed_normal

    # ──────────────────────────────────────────────────────────────
    # Velocity helpers
    # ──────────────────────────────────────────────────────────────
    def _clamp(self, value, min_abs, max_abs):
        """Clamp to [-max_abs, -min_abs] ∪ [min_abs, max_abs], or 0."""
        if abs(value) < min_abs:
            return 0.0
        return float(np.clip(value, -max_abs, max_abs))

    def _publish_twist(self, linear_x=0.0, angular_z=0.0):
        msg = Twist()
        msg.linear.x  = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(msg)

    def _stop(self):
        self._publish_twist()

    # ──────────────────────────────────────────────────────────────
    # Goal point computation
    # ──────────────────────────────────────────────────────────────
    def _compute_goal(self, marker_pos, marker_normal):
        """
        Goal point = marker_pos + target_distance * marker_normal.
        Because the normal points TOWARD the robot, walking along it
        from the marker lands us directly in front of it.
        """
        return marker_pos + self.target_distance * marker_normal

    # ──────────────────────────────────────────────────────────────
    # Control loop
    # ──────────────────────────────────────────────────────────────
    def _control_loop(self):

        # ── DONE ─────────────────────────────────────────────────
        if self.state == State.DONE:
            self._stop()
            return

        pos, normal = self._get_filtered_marker_data()

        # ── SEARCH ───────────────────────────────────────────────
        if self.state == State.SEARCH:
            if pos is None:
                # Slowly rotate to find the marker
                self._publish_twist(angular_z=0.3)
                self.get_logger().info('SEARCH: rotating to find marker...', throttle_duration_sec=2.0)
            else:
                self.get_logger().info('SEARCH: marker found, transitioning to ALIGN_HEADING')
                self.state = State.ALIGN_HEADING
            return

        # From here we always need a valid marker pose
        if pos is None:
            self.get_logger().warn(
                f'Lost marker in state {self.state.name}, stopping.', throttle_duration_sec=1.0
            )
            self._stop()
            return

        goal = self._compute_goal(pos, normal)

        # ── ALIGN_HEADING ─────────────────────────────────────────
        # Rotate until we are roughly pointing at the goal point.
        # goal[1] is the lateral (y) offset to the goal; goal[0] is forward.
        if self.state == State.ALIGN_HEADING:
            heading_to_goal = np.arctan2(goal[1], goal[0])
            ang_cmd = self._clamp(
                self.kp_angular * heading_to_goal,
                self.min_ang, self.max_ang
            )
            self._publish_twist(angular_z=ang_cmd)
            self.get_logger().debug(
                f'ALIGN_HEADING: heading_error={np.degrees(heading_to_goal):.1f}°'
            )
            if abs(heading_to_goal) < self.head_tol:
                self.get_logger().info('ALIGN_HEADING: done, transitioning to APPROACH')
                self.state = State.APPROACH
            return

        # ── APPROACH ─────────────────────────────────────────────
        # Drive forward while keeping the goal centred laterally.
        if self.state == State.APPROACH:
            forward_dist = goal[0]                        # x = forward in base_link
            lateral_err  = goal[1]                        # y = left  in base_link

            lin_cmd = self._clamp(
                self.kp_linear * forward_dist,
                self.min_lin, self.max_lin
            )
            ang_cmd = self._clamp(
                self.kp_angular * lateral_err,
                self.min_ang, self.max_ang
            )

            # Slow down angular correction when very close to avoid spinning in place
            if abs(forward_dist) < 0.15:
                ang_cmd *= 0.3

            self._publish_twist(linear_x=lin_cmd, angular_z=ang_cmd)
            self.get_logger().debug(
                f'APPROACH: fwd={forward_dist:.3f} m, lat={lateral_err:.3f} m'
            )

            if abs(forward_dist) < self.dist_tol and abs(lateral_err) < self.dist_tol:
                self._stop()
                self.get_logger().info('APPROACH: at goal, transitioning to FINAL_ALIGN')
                self.state = State.FINAL_ALIGN
            return

        # ── FINAL_ALIGN ───────────────────────────────────────────
        # The robot is at the goal point. Now rotate to face opposite
        # the marker's outward normal (i.e. face the marker squarely).
        # Desired robot heading = -normal projected onto the xy plane.
        if self.state == State.FINAL_ALIGN:
            # The robot should face the direction -normal (into the marker).
            # heading_error = angle between robot's forward (x-axis, = [1,0])
            # and the desired direction.
            desired = -normal                              # direction robot should face
            heading_error = np.arctan2(desired[1], desired[0])

            ang_cmd = self._clamp(
                self.kp_angular * heading_error,
                self.min_ang, self.max_ang
            )
            self._publish_twist(angular_z=ang_cmd)
            self.get_logger().debug(
                f'FINAL_ALIGN: heading_error={np.degrees(heading_error):.1f}°'
            )

            if abs(heading_error) < self.head_tol:
                self._stop()
                self.get_logger().info('FINAL_ALIGN: docking complete!')
                self.state = State.DONE
            return


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = ArucoDockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()