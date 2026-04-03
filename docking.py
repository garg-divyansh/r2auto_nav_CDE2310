import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from tf2_ros import (
    Buffer, TransformListener,
    LookupException, ExtrapolationException, ConnectivityException
)
from scipy.spatial.transform import Rotation
import numpy as np
import math

"""
docking.py — Three-phase autonomous ArUco marker docking.

=== OVERVIEW ===
Docks a TurtleBot3 in front of an ArUco marker using TF2 transforms,
odometry navigation, and LIDAR for final positioning. Receives commands
on /states (format: 'DOCK_<marker_id>'), reports on /operation_status.

=== DOCKING PHASES ===

  Phase 1 — NAV_TO_STANDOFF (odom-based)
      Queries marker via TF twice, rejects if distance differs by
      >10%, then uses the second (fresher) reading to compute a
      goal point along the marker's normal at nav_standoff distance
      (20cm). Navigates there using odom frame dead reckoning.
      On arrival, rotates to face the marker.

  Phase 2 — FINE_APPROACH (TF-based with EMA filtering)
      Re-acquires marker via TF with exponential moving average
      smoothing on position and normal. Drives from 20cm to 15cm
      while correcting lateral offset and heading. Angular control
      blends bearing correction (far) with heading alignment (near).
      Transitions when lateral alignment < lateral_tol (0.5cm).
      If marker is lost for >2s, performs a 360° recovery spin.
      If re-acquired during spin, resumes approach. If not found
      after full rotation, reports DOCK_FAIL.

  Phase 3 — LIDAR_FINAL
      Switches to LIDAR for distance measurement. Drives straight to
      final standoff_distance (8cm) — heading is already aligned from
      Phase 2's blended angular control. Stops and reports DOCK_DONE.

=== TOPICS ===
  Subscribes:  /states (String), /scan (LaserScan)
  Publishes:   /cmd_vel (Twist), /operation_status (String)
  TF frames:   odom, base_link, aruco_marker_<id>

=== FUNCTIONS ===
  scan_callback()           — stores latest LIDAR scan
  get_lidar_distance()      — min distance in a LIDAR arc
  docking_callback()        — parses DOCK command, starts docking
  docking_step()            — 10Hz loop routing to current phase
  _lookup_marker_bl()       — single TF lookup → (pos, normal) in base_link
  _compute_odom_goal()      — two-sample TF → goal in odom frame
  _get_robot_odom_pose()    — robot (x, y, yaw) in odom frame
  _extract_normal()         — marker normal with flip detection
  _get_marker_data()        — TF lookup with EMA filtering
  _phase_nav_to_standoff()  — Phase 1: odom nav to standoff point
  _phase_fine_approach()    — Phase 2: TF fine approach to 15cm
  _recovery_spin_tick()     — Phase 2: 360° spin searching for marker
  _phase_lidar_final()      — Phase 3: LIDAR approach to 8cm
  _finish_docking()         — cleanup + publish status

=== PARAMETERS (all configurable via ros2 run --ros-args -p name:=val) ===
  nav_standoff       — Phase 1 goal distance from marker (0.20 m)
  fine_approach_dist — Phase 2 stopping distance (0.15 m)
  standoff_distance  — Phase 3 final distance (0.08 m)
  odom_position_tol  — Phase 1 arrival tolerance (0.02 m)
  lateral_tol        — Phase 2 lateral alignment (0.005 m)
  distance_tol       — Phase 2 distance tolerance (0.01 m)
  final_tol          — Phase 3 LIDAR tolerance (0.005 m)
  heading_tol        — angular tolerance throughout (0.05 rad)
  angular_threshold  — min angle error to trigger pure rotation (0.05 rad)
  k_linear / k_angular — proportional gains (0.5 / 1.0)
  lidar_arc_deg      — LIDAR sampling half-arc (3.0 deg)
  lpf_alpha          — EMA smoothing factor (0.35)
  max_linear / max_angular — velocity limits (0.15 m/s / 0.5 rad/s)
  min_angular        — deadband floor for rotation (0.05 rad/s)
  recovery_spin_speed — angular velocity for recovery spin (0.3 rad/s)
  timeout_sec        — safety abort timer (45.0 s)
  verbose            — debug logging (False)
"""


class LowPassFilter:
    """Exponential moving average for numpy arrays.
    Lower alpha = more smoothing, higher = faster response."""

    def __init__(self, alpha):
        self.alpha = alpha
        self._value = None

    def update(self, new_value):
        if self._value is None:
            self._value = new_value.copy()
        else:
            self._value = (self.alpha * new_value
                           + (1.0 - self.alpha) * self._value)
        return self._value

    def reset(self):
        self._value = None

    @property
    def value(self):
        return self._value


class DockingNode(Node):
    """Three-phase docking node: odom nav → TF fine approach → LIDAR final."""

    def __init__(self):
        super().__init__('docking_node')

        # --- Subscribers & publishers ---
        self.create_subscription(String, '/states', self.docking_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.dock_complete_pub = self.create_publisher(
            String, '/operation_status', 10)
        self.latest_scan = None
        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)

        # --- TF2 ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- State ---
        self.marker_id = None
        self.docking_phase = None
        self.docking_timer = None
        self.iteration_count = 0
        self.consecutive_misses = 0
        self.aligned_iterations = 0

        # Phase 1: odom-frame goal
        self.goal_odom_x = None
        self.goal_odom_y = None
        self.goal_odom_yaw = None
        self.goal_computed = False

        # Phase 2: EMA filters and flip detection state
        self.pos_filter = None
        self.normal_filter = None
        self.prev_normal = None

        # Phase 2: recovery spin state
        self.recovery_spin_active = False
        self.recovery_spin_prev_yaw = None
        self.recovery_spin_cumulative = 0.0
        self.recovery_spin_attempted = False

        # --- ROS2 parameters ---
        self.declare_parameter("nav_standoff", 0.20)        # Phase 1 goal distance from marker along normal (m)
        self.declare_parameter("fine_approach_dist", 0.15)   # Phase 2 stopping distance from marker (m)
        self.declare_parameter("standoff_distance", 0.08)    # Phase 3 final distance from marker surface (m)
        self.declare_parameter("odom_position_tol", 0.02)    # Phase 1 odom arrival position tolerance (m)
        self.declare_parameter("lateral_tol", 0.005)         # Phase 2 lateral alignment tolerance (m)
        self.declare_parameter("distance_tol", 0.01)         # Phase 2 distance arrival tolerance (m)
        self.declare_parameter("final_tol", 0.005)           # Phase 3 LIDAR distance tolerance (m)
        self.declare_parameter("heading_tol", 0.05)          # Angular tolerance for heading alignment (rad)
        self.declare_parameter("angular_threshold", 0.05)    # Min angle error to trigger pure rotation (rad)
        self.declare_parameter("k_linear", 0.5)              # Proportional gain for forward speed
        self.declare_parameter("k_angular", 1.0)             # Proportional gain for rotation speed
        self.declare_parameter("lidar_arc_deg", 3.0)         # LIDAR sampling half-arc width (deg)
        self.declare_parameter("lpf_alpha", 0.35)            # EMA smoothing factor (0=smooth, 1=raw)
        self.declare_parameter("max_linear", 0.15)           # Max forward velocity (m/s)
        self.declare_parameter("max_angular", 0.5)           # Max rotation velocity (rad/s)
        self.declare_parameter("min_angular", 0.05)          # Deadband floor for rotation commands (rad/s)
        self.declare_parameter("recovery_spin_speed", 0.3)    # Recovery spin angular velocity (rad/s)
        self.declare_parameter("timeout_sec", 45.0)          # Safety abort timer (s)
        self.declare_parameter("verbose", False)             # Enable debug logging
        self._load_params()

    def _load_params(self):
        g = self.get_parameter
        self.nav_standoff = g("nav_standoff").value
        self.fine_approach_dist = g("fine_approach_dist").value
        self.standoff_distance = g("standoff_distance").value
        self.odom_position_tol = g("odom_position_tol").value
        self.lateral_tol = g("lateral_tol").value
        self.distance_tol = g("distance_tol").value
        self.final_tol = g("final_tol").value
        self.heading_tol = g("heading_tol").value
        self.angular_threshold = g("angular_threshold").value
        self.k_linear = g("k_linear").value
        self.k_angular = g("k_angular").value
        self.lidar_arc_deg = g("lidar_arc_deg").value
        self.lpf_alpha = g("lpf_alpha").value
        self.max_linear = g("max_linear").value
        self.max_angular = g("max_angular").value
        self.min_angular = g("min_angular").value
        self.recovery_spin_speed = g("recovery_spin_speed").value
        self.timeout_sec = g("timeout_sec").value
        self.verbose = g("verbose").value

    # ================================================================
    # LIDAR
    # ================================================================

    def scan_callback(self, msg):
        """Store latest LIDAR scan."""
        self.latest_scan = msg

    def get_lidar_distance(self, angle_rad=0.0):
        """Min LIDAR distance within a small arc around angle_rad.
        Returns None if no valid readings in the arc."""
        if self.latest_scan is None:
            return None

        scan = self.latest_scan
        # Normalize angle into LIDAR's [angle_min, angle_min+2pi) range
        angle = angle_rad
        while angle < scan.angle_min:
            angle += 2 * math.pi
        while angle >= scan.angle_min + 2 * math.pi:
            angle -= 2 * math.pi

        center_idx = int((angle - scan.angle_min) / scan.angle_increment)
        arc_half = int(math.radians(self.lidar_arc_deg) / scan.angle_increment)

        # Sample arc, wrapping around the array
        valid = []
        for i in range(-arc_half, arc_half + 1):
            idx = (center_idx + i) % len(scan.ranges)
            r = scan.ranges[idx]
            if scan.range_min < r < scan.range_max and math.isfinite(r):
                valid.append(r)

        return min(valid) if valid else None

    # ================================================================
    # DOCKING COMMAND HANDLER
    # ================================================================

    def docking_callback(self, msg):
        """Parse DOCK_<id> from /states, reset state, start 10Hz loop."""
        try:
            message, number = msg.data.split("_", 1)
        except ValueError:
            return
        if message != "DOCK":
            return

        if self.docking_timer is not None:
            self.get_logger().warn("Docking already in progress, ignoring")
            return

        self.get_logger().info(f"Docking command: marker {number}")
        self.marker_id = int(number)

        # Reset all state for fresh attempt
        self.docking_phase = "nav_to_standoff"
        self.iteration_count = 0
        self.consecutive_misses = 0
        self.aligned_iterations = 0
        self.goal_computed = False
        self.pos_filter = LowPassFilter(self.lpf_alpha)
        self.normal_filter = LowPassFilter(self.lpf_alpha)
        self.prev_normal = None
        self.recovery_spin_active = False
        self.recovery_spin_prev_yaw = None
        self.recovery_spin_cumulative = 0.0
        self.recovery_spin_attempted = False

        self.docking_timer = self.create_timer(0.1, self.docking_step)

    # ================================================================
    # MAIN CONTROL LOOP (10Hz)
    # ================================================================

    def docking_step(self):
        """Route to current docking phase. Aborts on timeout."""
        self.iteration_count += 1
        if self.iteration_count > self.timeout_sec * 10:
            self.get_logger().warn("Docking timeout")
            self.cmd_pub.publish(Twist())
            self._finish_docking("TIMEOUT")
            return

        if self.docking_phase == "nav_to_standoff":
            self._phase_nav_to_standoff()
        elif self.docking_phase == "fine_approach":
            self._phase_fine_approach()
        elif self.docking_phase == "lidar_final":
            self._phase_lidar_final()

    # ================================================================
    # TF HELPERS
    # ================================================================

    def _get_robot_odom_pose(self):
        """Returns (x, y, yaw) in odom frame, or None on failure."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'odom', 'base_link', rclpy.time.Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            yaw = Rotation.from_quat(
                [q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
            return t.x, t.y, yaw
        except (LookupException, ExtrapolationException,
                ConnectivityException):
            return None

    def _extract_normal(self, rotation_matrix):
        """Marker normal from rotation matrix Z-column, projected to XY.
        Flips to ensure it points toward robot, with dot-product
        stability check against previous reading."""
        normal = rotation_matrix[:, 2].copy()
        normal[2] = 0.0

        # Normal should point toward robot (negative x in base_link)
        if normal[0] > 0:
            normal = -normal

        # Reject sudden 180-deg flips
        if self.prev_normal is not None:
            if np.dot(normal, self.prev_normal) < 0:
                normal = -normal

        n = np.linalg.norm(normal)
        if n > 1e-6:
            normal /= n
        self.prev_normal = normal.copy()
        return normal

    def _lookup_marker_bl(self):
        """Single TF lookup returning (position, normal) in base_link.
        Raises LookupException/ExtrapolationException on failure."""
        tf = self.tf_buffer.lookup_transform(
            'base_link', f'aruco_marker_{self.marker_id}',
            rclpy.time.Time())
        t = tf.transform.translation
        q = tf.transform.rotation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        pos = np.array([t.x, t.y, 0.0])
        normal = self._extract_normal(R)
        return pos, normal

    def _compute_odom_goal(self):
        """Two-sample TF lookup to place a goal point nav_standoff metres
        in front of the marker (along its normal) in odom frame.
        Retries until two consecutive readings agree within 15%.
        Uses the second (fresher) reading for the goal."""

        max_retries = 10
        for attempt in range(max_retries):
            pos1, _ = self._lookup_marker_bl()
            dist1 = float(np.linalg.norm(pos1[:2]))

            pos2, normal = self._lookup_marker_bl()
            dist2 = float(np.linalg.norm(pos2[:2]))

            if dist1 > 1e-6 and abs(dist2 - dist1) / dist1 > 0.15:
                self.get_logger().info(
                    f"TF mismatch ({dist1:.3f}m vs {dist2:.3f}m), "
                    f"retry {attempt + 1}/{max_retries}")
                continue

            # Two frames are consistent — use the second
            break
        else:
            raise LookupException(
                f"TF readings inconsistent after {max_retries} retries")

        marker_pos = pos2

        # Goal in base_link: standoff point along marker normal
        goal_bl = marker_pos + self.nav_standoff * normal
        # Target heading: face toward marker (opposite the normal)
        heading_bl = math.atan2(-normal[1], -normal[0])

        # Rotate and translate goal into odom frame
        pose = self._get_robot_odom_pose()
        if pose is None:
            raise LookupException("Cannot get odom pose")
        rx, ry, ryaw = pose

        cos_y, sin_y = math.cos(ryaw), math.sin(ryaw)
        self.goal_odom_x = rx + cos_y * goal_bl[0] - sin_y * goal_bl[1]
        self.goal_odom_y = ry + sin_y * goal_bl[0] + cos_y * goal_bl[1]
        self.goal_odom_yaw = math.atan2(
            math.sin(ryaw + heading_bl), math.cos(ryaw + heading_bl))
        self.goal_computed = True

        self.get_logger().info(
            f"Odom goal: ({self.goal_odom_x:.3f}, {self.goal_odom_y:.3f}), "
            f"heading={math.degrees(self.goal_odom_yaw):.1f} deg")

    def _get_marker_data(self):
        """TF lookup with EMA filtering on position and normal.
        Returns (smoothed_pos, smoothed_normal) or (None, None).
        Rejects transforms older than 0.5s."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link', f'aruco_marker_{self.marker_id}',
                rclpy.time.Time())

            age = (self.get_clock().now() - rclpy.time.Time.from_msg(
                tf.header.stamp)).nanoseconds / 1e9
            if age > 0.5:
                return None, None

            t = tf.transform.translation
            q = tf.transform.rotation
            R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

            pos = np.array([t.x, t.y, 0.0])
            normal = self._extract_normal(R)

            # EMA smoothing on both position and normal
            s_pos = self.pos_filter.update(pos)
            s_normal = self.normal_filter.update(normal)
            n = np.linalg.norm(s_normal)
            if n > 1e-6:
                s_normal = s_normal / n

            return s_pos, s_normal

        except (LookupException, ExtrapolationException,
                ConnectivityException):
            return None, None

    # ================================================================
    # PHASE 1: ODOM NAVIGATION TO STANDOFF POINT
    # ================================================================

    def _phase_nav_to_standoff(self):
        """Navigate to 20cm standoff via odom, then rotate to face marker.
        First call computes goal from a single TF lookup. Subsequent
        calls drive toward the odom-frame goal with P-control."""

        # Compute odom goal on first iteration
        if not self.goal_computed:
            try:
                self._compute_odom_goal()
            except (LookupException, ExtrapolationException,
                    ConnectivityException):
                self.get_logger().info(
                    "Waiting for marker TF...",
                    throttle_duration_sec=1.0)
                return

        pose = self._get_robot_odom_pose()
        if pose is None:
            self.cmd_pub.publish(Twist())
            return
        rx, ry, ryaw = pose

        dx = self.goal_odom_x - rx
        dy = self.goal_odom_y - ry
        dist = math.sqrt(dx**2 + dy**2)
        cmd = Twist()

        if dist > self.odom_position_tol:
            # Drive toward goal position
            angle_to_goal = math.atan2(dy, dx)
            angle_err = math.atan2(
                math.sin(angle_to_goal - ryaw),
                math.cos(angle_to_goal - ryaw))

            if abs(angle_err) > self.angular_threshold:
                # Pure rotation to face goal
                cmd.angular.z = self._clamp_angular(
                    self.k_angular * angle_err)
            else:
                # Drive forward with angular correction
                cmd.linear.x = self._clamp_linear(
                    self.k_linear * dist)
                cmd.angular.z = self._clamp_angular(
                    self.k_angular * angle_err)
        else:
            # At goal position — rotate to face marker
            h_err = math.atan2(
                math.sin(self.goal_odom_yaw - ryaw),
                math.cos(self.goal_odom_yaw - ryaw))

            if abs(h_err) > self.heading_tol:
                ang = self.k_angular * h_err
                # Deadband floor to overcome motor stiction
                if 0 < abs(ang) < self.min_angular:
                    ang = math.copysign(self.min_angular, ang)
                cmd.angular.z = self._clamp_angular(ang)
            else:
                # Phase 1 complete — at standoff, facing marker
                self.cmd_pub.publish(Twist())
                self.get_logger().info(
                    "Phase 1 complete — at standoff, facing marker")
                self.docking_phase = "fine_approach"
                self.aligned_iterations = 0
                self.consecutive_misses = 0
                self.pos_filter.reset()
                self.normal_filter.reset()
                self.prev_normal = None
                return

        self.cmd_pub.publish(cmd) #publishes stop if already reached goal position

    # ================================================================
    # PHASE 2: TF-BASED FINE APPROACH TO 15CM
    # ================================================================

    def _phase_fine_approach(self):
        """Drive from standoff to fine_approach_dist with lateral and
        heading corrections. Angular control blends bearing (keeps
        marker centered) at distance with heading alignment (faces
        marker normal) up close.

        If the marker is lost for >2s, initiates a 360-degree recovery
        spin. If the marker is re-acquired during the spin, resumes
        fine approach. If the full rotation completes without finding
        the marker, reports DOCK_FAIL."""

        # --- Recovery spin mode ---
        if self.recovery_spin_active:
            self._recovery_spin_tick()
            return

        pos, normal = self._get_marker_data()
        if pos is None:
            self.consecutive_misses += 1
            if self.consecutive_misses > 20:
                if not self.recovery_spin_attempted:
                    self.get_logger().warn(
                        "Marker lost in fine approach — "
                        "starting 360° recovery spin")
                    self.cmd_pub.publish(Twist())
                    self.recovery_spin_active = True
                    self.recovery_spin_prev_yaw = None
                    self.recovery_spin_cumulative = 0.0
                    self.recovery_spin_attempted = True
                    return
                else:
                    self.get_logger().warn(
                        "Marker lost after recovery spin — docking failed")
                    self.cmd_pub.publish(Twist())
                    self._finish_docking("DOCK_FAIL")
                    return
            self.cmd_pub.publish(Twist())
            return
        self.consecutive_misses = 0

        distance = float(np.linalg.norm(pos[:2]))
        distance_err = distance - self.fine_approach_dist
        angle_to_marker = math.atan2(pos[1], pos[0])
        lateral_err = abs(pos[1])

        # Desired heading: face opposite marker normal (into the marker)
        desired_heading = math.atan2(-normal[1], -normal[0])

        cmd = Twist()

        if abs(angle_to_marker) > self.angular_threshold:
            # Marker off-center — pure rotation to center it
            cmd.angular.z = self._clamp_angular(
                self.k_angular * angle_to_marker)
        else:
            # Blend: bearing correction (far) → heading alignment (near)
            approach_range = self.nav_standoff - self.fine_approach_dist
            blend = 0.0
            if approach_range > 0:
                blend = max(0.0, min(
                    1.0, 1.0 - distance_err / approach_range))

            ang = ((1.0 - blend) * angle_to_marker
                   + blend * desired_heading)
            cmd.linear.x = self._clamp_linear(
                self.k_linear * distance_err)
            cmd.angular.z = self._clamp_angular(self.k_angular * ang)

            # Damp angular near target to prevent oscillation
            if distance_err < 0.05:
                cmd.angular.z *= 0.3

        self.cmd_pub.publish(cmd)

        if self.verbose:
            self.get_logger().info(
                f"FINE: d={distance:.3f}, lat={pos[1]:.4f}, "
                f"head={math.degrees(desired_heading):.1f} deg")

        # Transition when at distance AND laterally aligned for 0.5s
        if (abs(distance_err) < self.distance_tol
                and lateral_err < self.lateral_tol):
            self.aligned_iterations += 1
            if self.aligned_iterations >= 5:
                self.cmd_pub.publish(Twist())
                self.get_logger().info(
                    f"Phase 2 complete — {distance*100:.1f}cm, "
                    f"lateral {lateral_err*100:.2f}cm")
                self.docking_phase = "lidar_final"
                self.aligned_iterations = 0
                return
        else:
            self.aligned_iterations = 0

    def _recovery_spin_tick(self):
        """Spin the robot 360° while checking for the ArUco marker.
        If the marker is found, cancel the spin and resume fine approach.
        If a full rotation completes without detection, report DOCK_FAIL."""

        # Check for marker during spin
        pos, _ = self._get_marker_data()
        if pos is not None:
            self.get_logger().info(
                "Marker re-acquired during recovery spin — "
                "resuming fine approach")
            self.cmd_pub.publish(Twist())
            self.recovery_spin_active = False
            self.consecutive_misses = 0
            self.pos_filter.reset()
            self.normal_filter.reset()
            self.prev_normal = None
            return

        # Get current yaw to track cumulative rotation
        pose = self._get_robot_odom_pose()
        if pose is None:
            self.cmd_pub.publish(Twist())
            return
        _, _, yaw = pose

        if self.recovery_spin_prev_yaw is not None:
            delta = yaw - self.recovery_spin_prev_yaw
            # Normalize delta to [-pi, pi]
            delta = math.atan2(math.sin(delta), math.cos(delta))
            self.recovery_spin_cumulative += delta
        self.recovery_spin_prev_yaw = yaw

        # Check if full 360° completed
        if self.recovery_spin_cumulative >= 2.0 * math.pi:
            self.get_logger().warn(
                "360° recovery spin complete — marker not found, "
                "docking failed")
            self.cmd_pub.publish(Twist())
            self.recovery_spin_active = False
            self._finish_docking("DOCK_FAIL")
            return

        # Spin in place
        cmd = Twist()
        cmd.angular.z = self.recovery_spin_speed
        self.cmd_pub.publish(cmd)

    # ================================================================
    # PHASE 3: LIDAR FINAL APPROACH TO STANDOFF DISTANCE
    # ================================================================

    def _phase_lidar_final(self):
        """Drive to standoff_distance using forward LIDAR readings.
        Heading is already aligned from Phase 2's blended angular
        control, so no TF angular corrections are applied here."""

        lidar_dist = self.get_lidar_distance(0.0)
        if lidar_dist is None:
            self.cmd_pub.publish(Twist())
            self.get_logger().warn(
                "No LIDAR data", throttle_duration_sec=1.0)
            return

        distance_err = lidar_dist - self.standoff_distance
        cmd = Twist()

        # Check if at final standoff distance
        if abs(distance_err) < self.final_tol:
            self.cmd_pub.publish(Twist())
            self.get_logger().info(
                f"Phase 3 complete — {lidar_dist*100:.1f}cm from surface")
            self._finish_docking("DOCK_DONE")
            return

        # Slow final approach (capped at 8cm/s)
        cmd.linear.x = max(-0.08, min(0.08,
                                       self.k_linear * distance_err))
        self.cmd_pub.publish(cmd)

    # ================================================================
    # VELOCITY HELPERS
    # ================================================================

    def _clamp_linear(self, v):
        return max(-self.max_linear, min(self.max_linear, v))

    def _clamp_angular(self, v):
        return max(-self.max_angular, min(self.max_angular, v))

    # ================================================================
    # CLEANUP
    # ================================================================

    def _finish_docking(self, status):
        """Stop robot, cancel timer, reset state, publish result."""
        self.get_logger().info(f"Docking finished: {status}")

        if self.docking_timer is not None:
            self.destroy_timer(self.docking_timer)
            self.docking_timer = None

        self.marker_id = None
        self.docking_phase = None
        self.iteration_count = 0
        self.consecutive_misses = 0
        self.aligned_iterations = 0
        self.goal_computed = False
        self.recovery_spin_active = False
        self.recovery_spin_prev_yaw = None
        self.recovery_spin_cumulative = 0.0
        self.recovery_spin_attempted = False
        if self.pos_filter:
            self.pos_filter.reset()
        if self.normal_filter:
            self.normal_filter.reset()
        self.prev_normal = None

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
