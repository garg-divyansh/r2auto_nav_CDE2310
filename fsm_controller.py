import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from geometry_msgs.msg import PoseStamped

class FSMNode(Node):
    def __init__(self):
        super().__init__('fsm_controller')

        # ================= INTERNAL VARIABLES =================
        self.state = "IDLE"
        self.prev_state = None

        self.marker_detected = False
        self.marker_count = 0
        self.required_markers = 2
        self.map_explored = False

        self.current_marker = None
        self.marker_id = None

        # Error handling
        self.error_detected = False
        self.error_type = None

        # 🔴 Dock retry logic
        self.dock_attempts = 0
        self.max_dock_attempts = 2

        # ================= PUBLISHERS =================
        self.state_pub = self.create_publisher(String, '/states', 10)
        self.current_marker_pub = self.create_publisher(Int32,'/current_marker', 10)

        # ================= SUBSCRIBERS =================
        self.create_subscription(PoseStamped, '/aruco_pose', self.aruco_callback, 10)
        self.create_subscription(String, '/operation_status', self.status_callback, 10)

        # ================= TIMER =================
        self.timer = self.create_timer(0.1, self.state_machine_loop)

        self.get_logger().info("FSM Controller Started")
        self.change_state("EXPLORE")

    # ================= STATE TRANSITION =================
    def change_state(self, new_state, marker_id=None):
        if self.state != new_state or marker_id is not None:
            self.prev_state = self.state
            self.state = new_state

            msg = String()
            if marker_id is not None and new_state in ["DOCK", "LAUNCH"]:
                msg.data = f"{new_state}_{marker_id}"
            else:
                msg.data = new_state

            self.state_pub.publish(msg)
            self.get_logger().info(f"Transitioned to {msg.data} state")

    # ================= FSM LOOP =================
    def state_machine_loop(self):

        if self.error_detected:
            self.handle_error()
            return

        if self.state == "EXPLORE":
            if self.marker_detected:
                self.marker_detected = False
                if self.current_marker:
                    self.marker_id = int(self.current_marker.header.frame_id.split("_")[-1])
                    self.change_state("DOCK", self.marker_id)

            elif self.map_explored and self.marker_count >= self.required_markers:
                self.change_state("END")

        elif self.state == "DOCK":
            pass

        elif self.state == "LAUNCH":
            pass

        elif self.state == "END":
            self.get_logger().info("Mission Complete! Goodbye!")

    # ================= ERROR HANDLER =================
    def handle_error(self):
        self.get_logger().error(f"Handling error: {self.error_type}")

        if self.error_type in ["DOCK_FAIL", "TIMEOUT"]:
            self.dock_attempts += 1

            if self.dock_attempts < self.max_dock_attempts:
                self.get_logger().warn(f"Dock failed (attempt {self.dock_attempts}) → retrying")

                if self.marker_id is not None:
                    self.change_state("DOCK", self.marker_id)
                else:
                    self.change_state("DOCK")

            else:
                self.get_logger().error("Dock failed twice → giving up")
                self.change_state("END")

        elif self.error_type == "NAV_FAIL":
            self.get_logger().warn("Navigation failed → return to explore")
            self.change_state("EXPLORE")

        elif self.error_type == "LAUNCH_FAIL":
            self.get_logger().warn("Launch failed → retry launch")

            if self.marker_id is not None:
                self.change_state("LAUNCH", self.marker_id)
            else:
                self.change