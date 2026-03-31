import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException, ConnectivityException
from scipy.spatial.transform import Rotation
import numpy as np
import math

"""
docking_node.py

ROS2 node for autonomous docking of a TurtleBot3 to an ArUco marker using TF2 transforms.

The node subscribes to /states for docking commands in the format 'DOCK_<marker_id>',
then performs closed-loop alignment by querying the TF transform of the target ArUco
marker relative to base_link and publishing velocity commands to /cmd_vel.

Alignment is done in three phases:
  Phase 1: Rotate to face the marker (angular correction only)
  Phase 2: Drive toward the marker with small angular corrections
  Phase 3: Final rotational alignment to face the marker head-on (uses stored
           heading from when the marker was still visible, since the camera loses
           the marker at close range ~5cm)

The marker's orientation is extracted from the TF quaternion to determine the marker's
facing direction. When the robot is close, LIDAR readings in the marker's direction
are used for more accurate distance measurement (the LIDAR section can be commented
out to revert to ArUco-only distance).

Docking is considered complete when the robot holds rotational alignment within the
angular threshold for 0.5 seconds. On completion, 'DOCK_DONE' is published to
/operation_status. On failure, 'DOCK_FAIL', 'MARKER_LOST', or 'TIMEOUT' is published.

Parameters:
    alignment_threshold (float): Positional tolerance in metres to consider aligned (default: 0.05)
    docking_distance (float):    Distance in metres to stop from the marker (default: 0.2)
    k_linear (float):            Proportional gain for linear velocity control (default: 0.5)
    k_angular (float):           Proportional gain for angular velocity control (default: 1.0)
    angular_threshold (float):   Angular error in radians below which linear motion begins (default: 0.05)
    lidar_switch_dist (float):   Distance in metres at which to switch from ArUco to LIDAR distance (default: 0.5)
    lidar_arc_deg (float):       Half-width in degrees of the LIDAR arc to sample around the marker direction (default: 5.0)
    verbose (bool):              Enable debug logging (default: False)
"""

class DockingNode(Node):
    def __init__(self):
        super().__init__('docking_node')

        # ================= SUBSCRIBERS & PUBLISHERS =================
        self.create_subscription(String, '/states', self.docking_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.dock_complete_pub = self.create_publisher(String, '/operation_status', 10)

        # Subscribe to LIDAR for distance verification when close to marker
        self.latest_scan = None
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)

        # ================= TF2 =================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ================= STATE VARIABLES =================
        self.marker_id = None           # ArUco marker ID currently being docked to
        self.aligned_iterations = 0     # consecutive iterations within distance tolerance
        self.consecutive_misses = 0     # consecutive TF lookup failures (marker lost detection)
        self.marker_yaw = 0.0           # orientation of the marker relative to base_link (radians)

        # Phase 3 (final rotation) state — used after distance alignment is achieved
        self.final_rotation = False             # True when robot has reached target distance, now rotating
        self.target_robot_yaw = None            # desired robot heading in odom frame (stored while marker was visible)
        self.rotation_aligned_iterations = 0    # consecutive iterations within rotation tolerance

        # ================= DECLARE PARAMETERS WITH DEFAULTS =================
        self.declare_parameter("alignment_threshold", 0.05)  # metres, positional tolerance for docking
        self.declare_parameter("docking_distance", 0.2)      # metres, target stop distance from marker
        self.declare_parameter("k_linear", 0.5)              # proportional gain for linear velocity
        self.declare_parameter("k_angular", 1.0)             # proportional gain for angular velocity
        self.declare_parameter("verbose", False)              # enable debug logging
        self.declare_parameter("angular_threshold", 0.05)    # rad, threshold to begin linear motion
        self.declare_parameter("lidar_switch_dist", 0.5)     # metres, distance at which to switch to LIDAR
        self.declare_parameter("lidar_arc_deg", 5.0)         # degrees, half-width of LIDAR sampling arc

        # ================= RETRIEVE PARAMETERS =================
        self.alignment_threshold = self.get_parameter("alignment_threshold").value
        self.docking_distance = self.get_parameter("docking_distance").value
        self.k_linear = self.get_parameter("k_linear").value
        self.k_angular = self.get_parameter("k_angular").value
        self.verbose = self.get_parameter("verbose").value
        self.angular_threshold = self.get_parameter("angular_threshold").value
        self.lidar_switch_dist = self.get_parameter("lidar_switch_dist").value
        self.lidar_arc_deg = self.get_parameter("lidar_arc_deg").value

        if self.verbose:
            self.get_logger().info(f"Loaded parameters: alignment_threshold={self.alignment_threshold}, "
                                f"docking_distance={self.docking_distance}, k_linear={self.k_linear}, "
                                f"k_angular={self.k_angular}, verbose={self.verbose}")

        # ================= DOCKING FLOW TRACKING =================
        self.docking_timer = None
        self.alignment_iterations = 0   # total iterations in current docking attempt (for timeout)

    # ================= LIDAR CALLBACK =================
    def scan_callback(self, msg):
        """Store the latest LIDAR scan for distance verification."""
        self.latest_scan = msg

    def get_lidar_distance(self, angle_rad):
        """
        Get the LIDAR distance reading in the direction of angle_rad (relative to base_link).
        Samples a small arc around the target angle and returns the minimum distance.
        Returns None if no valid readings are available.
        """
        # Return None if no scan data is available yet
        if self.latest_scan is None:
            return None

        scan = self.latest_scan
        # Convert the target angle to a LIDAR index
        # LIDAR index 0 is straight ahead, indices increase counter-clockwise
        angle_min = scan.angle_min
        angle_inc = scan.angle_increment
        num_readings = len(scan.ranges)

        # Normalize angle_rad into [angle_min, angle_min + 2*pi) to match LIDAR convention
        # atan2 returns [-pi, pi] but TurtleBot3 LIDAR uses [0, 2*pi]
        angle_normalized = angle_rad
        while angle_normalized < angle_min:
            angle_normalized += 2 * math.pi
        while angle_normalized >= angle_min + 2 * math.pi:
            angle_normalized -= 2 * math.pi

        # Calculate the center index corresponding to the normalized angle
        center_idx = int((angle_normalized - angle_min) / angle_inc)

        # Calculate how many indices correspond to the arc half-width
        arc_half = int(math.radians(self.lidar_arc_deg) / angle_inc)

        # Build the list of indices to sample, wrapping around the scan array
        indices = [(center_idx + i) % num_readings for i in range(-arc_half, arc_half + 1)]

        # Collect valid (finite, positive) range readings from the arc
        valid_ranges = []
        for idx in indices:
            r = scan.ranges[idx]
            if scan.range_min < r < scan.range_max and math.isfinite(r):
                valid_ranges.append(r)

        # Return the closest reading in the arc, or None if no valid readings
        if valid_ranges:
            return min(valid_ranges)
        return None

    # ================= HELPER: GET ROBOT YAW FROM ODOM =================
    def _get_robot_yaw(self):
        """
        Look up the robot's current heading (yaw) in the odom frame via TF2.
        Returns the yaw angle in radians, or None if the transform is unavailable.

        This is used during Phase 3 (final rotation) when the ArUco marker is no
        longer visible. The odom frame provides a stable reference for tracking
        how much the robot has rotated.
        """
        try:
            # Look up the transform from odom to base_link (robot's pose in the odom frame)
            t = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            # Extract the quaternion from the transform
            quat = t.transform.rotation
            # Convert quaternion to euler angles — we only need yaw (rotation about Z)
            rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
            return rot.as_euler('xyz')[2]
        except (LookupException, ExtrapolationException, ConnectivityException):
            return None

    # ================= PHASE 3: FINAL ROTATIONAL ALIGNMENT =================
    def _final_rotation_step(self):
        """
        Phase 3 of docking: rotate in place to face the marker head-on.

        At this point the robot has already reached the target distance (Phase 2 complete),
        but the camera can no longer see the marker at close range. So we use a heading
        that was computed and stored in the odom frame WHILE the marker was still visible.

        The math:
        - self.target_robot_yaw is the heading (in odom frame) that makes the robot
          face the marker perpendicularly. It was computed as:
              target = current_yaw + heading_error
          where heading_error = how far off the robot was from facing the marker head-on.
        - Each iteration, we look up the robot's current yaw from odom → base_link TF,
          compute the remaining angle, and apply proportional angular velocity.
        - atan2(sin(diff), cos(diff)) normalizes the angle difference to [-π, π],
          preventing wraparound issues (e.g., going from 179° to -179° via 358° of rotation).
        """
        # Look up where the robot is currently pointing in the odom frame
        current_yaw = self._get_robot_yaw()
        if current_yaw is None:
            # Can't determine heading — stop and wait for TF to become available
            self.cmd_pub.publish(Twist())
            if self.verbose:
                self.get_logger().warn("Cannot get robot yaw from odom TF, waiting...")
            return

        # Compute the remaining rotation needed to reach the target heading
        # atan2(sin, cos) normalizes the difference to [-π, π] to avoid wraparound
        raw_diff = self.target_robot_yaw - current_yaw
        remaining = math.atan2(math.sin(raw_diff), math.cos(raw_diff))

        cmd = Twist()
        max_angular = 0.5  # rad/s cap, same as approach phase

        # Apply proportional angular velocity to close the remaining angle
        cmd.angular.z = self.k_angular * remaining
        # No linear motion during final rotation — robot stays in place
        cmd.linear.x = 0.0

        # Clamp angular velocity to safe limits
        cmd.angular.z = max(-max_angular, min(max_angular, cmd.angular.z))
        self.cmd_pub.publish(cmd)

        if self.verbose:
            self.get_logger().info(
                f"Phase 3 rotation: remaining={math.degrees(remaining):.1f}deg, "
                f"current_yaw={math.degrees(current_yaw):.1f}deg, "
                f"target_yaw={math.degrees(self.target_robot_yaw):.1f}deg"
            )

        # Check if the robot is within the angular threshold
        if abs(remaining) < self.angular_threshold:
            self.rotation_aligned_iterations += 1
            # Must hold alignment for 5 consecutive iterations (0.5s at 10Hz)
            if self.rotation_aligned_iterations >= 5:
                # Final rotation complete — stop and report success
                self.cmd_pub.publish(Twist())
                self.get_logger().info("Phase 3 rotation complete, docking done")
                self._finish_docking("DOCK_DONE")
                return
        else:
            # Reset if the robot drifts out of angular tolerance
            self.rotation_aligned_iterations = 0

    # ================= DOCK COMMAND CALLBACK =================
    def docking_callback(self, msg):
        try:
            message, number = msg.data.split("_", 1)
        except ValueError:
            return

        if message == "DOCK":
            self.get_logger().info("Docking command received")
            self.marker_id = int(number)
            # Reset all state for the new docking attempt
            self.alignment_iterations = 0
            self.aligned_iterations = 0
            self.consecutive_misses = 0
            self.marker_yaw = 0.0
            self.final_rotation = False
            self.target_robot_yaw = None
            self.rotation_aligned_iterations = 0
            # Start main docking loop timer (only if not already running)
            if self.docking_timer is None:
                self.docking_timer = self.create_timer(0.1, self.docking_step)
            else:
                self.get_logger().warn("Docking already in progress, ignoring duplicate command")
        else:
            return

    def docking_step(self):
        """Main docking loop — runs every 0.1 seconds (10 Hz)."""
        self.alignment_iterations += 1

        # Safety timeout: abort after 30 seconds (300 iterations * 0.1s)
        if self.alignment_iterations > 300:
            self.get_logger().warn("Docking timeout after 30s")
            self.cmd_pub.publish(Twist())
            self._finish_docking("TIMEOUT")
            return

        # If we've completed distance alignment, delegate to the final rotation phase
        if self.final_rotation:
            self._final_rotation_step()
            return

        # ================= TF LOOKUP =================
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link', f'aruco_marker_{self.marker_id}', rclpy.time.Time()
            )

            # Extract translation: dx = forward, dy = left(+)/right(-)
            dx = transform.transform.translation.x
            dy = transform.transform.translation.y
            aruco_distance = math.sqrt(dx**2 + dy**2)

            # Extract marker orientation from the TF quaternion
            quat = transform.transform.rotation
            rot = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
            # Yaw = rotation about the Z axis, which gives the marker's facing direction
            self.marker_yaw = rot.as_euler('xyz')[2]

            # Reject stale transforms older than 0.5 seconds
            transform_time = rclpy.time.Time.from_msg(transform.header.stamp)
            age = (self.get_clock().now() - transform_time).nanoseconds / 1e9

            if age > 0.5:
                # Stale transform — if already close enough, accept it as success
                if abs(aruco_distance - self.docking_distance) < self.alignment_threshold:
                    if self.verbose:
                        self.get_logger().info("Marker lost but close enough, docking complete")
                    self.cmd_pub.publish(Twist())
                    self._finish_docking("DOCK_DONE")
                else:
                    raise LookupException("Stale transform")
                return

            # Valid fresh transform — reset the miss counter
            self.consecutive_misses = 0

        except (LookupException, ExtrapolationException, ConnectivityException):
            self.consecutive_misses += 1
            # Stop the robot while marker is not visible
            self.cmd_pub.publish(Twist())
            if self.consecutive_misses > 5:
                self.get_logger().warn("Marker lost for 5 consecutive frames, aborting")
                self._finish_docking("MARKER_LOST")
            return

        # ================= DISTANCE MEASUREMENT =================
        # Angle from the robot's forward axis to the marker
        angle_to_marker = math.atan2(dy, dx)

        # Default: use ArUco TF distance for control
        distance = aruco_distance

        # --- LIDAR DISTANCE OVERRIDE (comment out this block to revert to ArUco-only) ---
        # When the robot is close to the marker, LIDAR gives a more accurate distance
        # reading than the ArUco pose estimate. We sample a small arc of LIDAR rays
        # in the direction of the marker and use the minimum reading.
        if aruco_distance < self.lidar_switch_dist:
            lidar_dist = self.get_lidar_distance(angle_to_marker)
            if lidar_dist is not None:
                distance = lidar_dist
                if self.verbose:
                    self.get_logger().info(
                        f"LIDAR override: lidar={lidar_dist:.3f}m, aruco={aruco_distance:.3f}m, "
                        f"marker_yaw={math.degrees(self.marker_yaw):.1f}deg"
                    )
        # --- END LIDAR DISTANCE OVERRIDE ---

        # Distance remaining to the target stop point
        distance_error = distance - self.docking_distance

        # ================= VELOCITY COMMAND =================
        cmd = Twist()
        max_linear = 0.15   # m/s cap to keep approach slow and safe
        max_angular = 0.5   # rad/s cap

        # Two-phase control: rotate first, then drive forward
        if abs(angle_to_marker) > self.angular_threshold:
            # Phase 1: pure rotation to face the marker
            cmd.linear.x = 0.0
            cmd.angular.z = self.k_angular * angle_to_marker
        else:
            # Phase 2: drive toward marker with small angular corrections
            cmd.linear.x = self.k_linear * distance_error
            cmd.angular.z = self.k_angular * angle_to_marker

        # Clamp velocities to safe limits
        cmd.linear.x = max(-max_linear, min(max_linear, cmd.linear.x))
        cmd.angular.z = max(-max_angular, min(max_angular, cmd.angular.z))
        self.cmd_pub.publish(cmd)

        # ================= DOCKING DISTANCE CHECK =================
        # Robot must hold within alignment threshold for 5 consecutive iterations (0.5s)
        if abs(distance_error) < self.alignment_threshold:
            self.aligned_iterations += 1
            if self.aligned_iterations >= 5:
                # Distance phase complete — stop linear motion
                self.cmd_pub.publish(Twist())
                self.get_logger().info("Phase 2 distance aligned, entering Phase 3 (final rotation)")

                # Compute the target heading BEFORE we lose sight of the marker.
                #
                # marker_yaw is the marker's Z-axis yaw relative to base_link.
                # When the robot is perfectly head-on, the marker's Z-axis points
                # directly back at the robot (into -X of base_link), so marker_yaw = ±π.
                #
                # heading_error = how far the robot's heading is from facing the marker
                # head-on. We compute this as the angular difference between marker_yaw
                # and π, normalized to [-π, π] using atan2(sin, cos).
                heading_error = math.atan2(
                    math.sin(self.marker_yaw - math.pi),
                    math.cos(self.marker_yaw - math.pi)
                )

                # Get the robot's current absolute heading in the odom frame
                current_yaw = self._get_robot_yaw()
                if current_yaw is None:
                    # Cannot determine heading — fall back to finishing without rotation
                    self.get_logger().warn("Cannot get robot yaw, skipping Phase 3")
                    self._finish_docking("DOCK_DONE")
                    return

                # target_robot_yaw = the odom-frame heading the robot should rotate to.
                # Since heading_error is how much more the robot needs to rotate (in base_link
                # frame), adding it to the current odom yaw gives the absolute target heading.
                # Normalize to [-π, π] so we always take the shortest rotation path.
                raw_target = current_yaw + heading_error
                self.target_robot_yaw = math.atan2(math.sin(raw_target), math.cos(raw_target))

                if self.verbose:
                    self.get_logger().info(
                        f"Stored target heading: marker_yaw={math.degrees(self.marker_yaw):.1f}deg, "
                        f"heading_error={math.degrees(heading_error):.1f}deg, "
                        f"current_yaw={math.degrees(current_yaw):.1f}deg, "
                        f"target_yaw={math.degrees(self.target_robot_yaw):.1f}deg"
                    )

                # Activate Phase 3 — next iteration will call _final_rotation_step
                self.final_rotation = True
                return
        else:
            # Reset if the robot drifts out of distance tolerance
            self.aligned_iterations = 0
    
    def _finish_docking(self, status):
        """
        Cleanup and finish docking with the given status.
        status should be one of: "DOCK_DONE", "DOCK_FAIL", "MARKER_LOST", "TIMEOUT"
        """
        self.get_logger().info(f"Docking finished with status: {status}")

        # Cancel the docking loop timer
        if self.docking_timer is not None:
            self.destroy_timer(self.docking_timer)
            self.docking_timer = None

        # Reset all state variables for the next docking attempt
        self.marker_id = None
        self.alignment_iterations = 0
        self.aligned_iterations = 0
        self.consecutive_misses = 0
        self.marker_yaw = 0.0
        self.final_rotation = False
        self.target_robot_yaw = None
        self.rotation_aligned_iterations = 0

        # Publish the result status so the FSM can react accordingly
        status_msg = String()
        status_msg.data = status
        self.dock_complete_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DockingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()