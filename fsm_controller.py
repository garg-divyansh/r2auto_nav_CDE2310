import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped

class FSMNode(Node):
    def __init__(self):
        super().__init__('fsm_controller')

        # Internal Variables
        self.state = "IDLE"
        self.prev_state = None

        self.marker_detected = False
        self.marker_count = 0
        self.required_markers = 2
        self.map_explored = False

        self.current_marker = None

        # Error Handling
        self.error_detected = False
        self.error_type = None

        # Publishers
        self.state_pub = self.create_publisher(String, '/states', 10)
        self.current_marker_pub = self.create_publisher(PoseStamped, '/current_marker', 10)

        # Subscribers
        self.create_subscription(PoseStamped, '/aruco_pose', self.aruco_callback, 10)
        self.create_subscription(Bool, '/dock_done', self.dock_done_callback, 10)
        self.create_subscription(Bool, '/launch_done', self.launch_done_callback, 10)
        self.create_subscription(Bool, '/map_explored', self.map_explored_callback, 10)
        self.create_subscription(String, '/operation_status', self.status_callback, 10)

        # Timer
        self.timer = self.create_timer(0.1, self.state_machine_loop)

        self.get_logger().info("FSM Controller Started")
        self.change_state("EXPLORE")

    def change_state(self, new_state):
        if self.state != new_state:
            self.prev_state = self.state
            self.state = new_state

            msg = String()
            msg.data = new_state
            self.state_pub.publish(msg)

            self.get_logger().info(f"Transitioned to {new_state} state")

    # ===================== FSM LOOP =====================
    def state_machine_loop(self):
        if self.error_detected:
            self.handle_error()
            return

        if self.state == "EXPLORE":
            if self.marker_detected:
                self.marker_detected = False
                self.change_state("DOCK")

            elif self.map_explored and self.marker_count >= self.required_markers:
                self.change_state("END")

        elif self.state == "DOCK":
            pass

        elif self.state == "LAUNCH":
            pass

        elif self.state == "END":
            self.get_logger().info("Mission Complete! Goodbye!")

    # ===================== ERROR HANDLER =====================
    def handle_error(self):
        self.get_logger().error(f"Handling error: {self.error_type}")

        if self.error_type == "DOCK_FAIL":
            self.get_logger().warn("Docking failed → retry docking")
            self.change_state("DOCK")

        elif self.error_type == "NAV_FAIL":
            self.get_logger().warn("Navigation failed → return to explore")
            self.change_state("EXPLORE")

        elif self.error_type == "LAUNCH_FAIL":
            self.get_logger().warn("Launch failed → retry launch")
            self.change_state("LAUNCH")

        elif self.error_type == "MARKER_LOST":
            self.get_logger().warn("Marker lost → re-exploring")
            self.change_state("EXPLORE")

        elif self.error_type == "TIMEOUT":
            self.get_logger().warn("Timeout → resetting to explore")
            self.change_state("EXPLORE")

        else:
            self.get_logger().fatal("Unknown error → stopping mission")
            self.change_state("END")

        # Reset error after handling
        self.error_detected = False
        self.error_type = None

    # ===================== CALLBACKS =====================
    def aruco_callback(self, msg):
        if self.state == "EXPLORE":
            self.get_logger().info("Marker Detected")
            self.marker_detected = True

            self.current_marker = msg
            self.current_marker_pub.publish(self.current_marker)

    def dock_done_callback(self, msg):
        if msg.data and self.state == "DOCK":
            self.get_logger().info("Docking done")
            self.current_marker = None
            self.change_state("LAUNCH")

    def launch_done_callback(self, msg):
        if msg.data and self.state == "LAUNCH":
            self.get_logger().info("Launch done")
            self.marker_count += 1
            self.change_state("EXPLORE")

    def map_explored_callback(self, msg):
        self.map_explored = msg.data

    def status_callback(self, msg):
        self.get_logger().info(f"Status received: {msg.data}")

        if msg.data != "OK":
            self.error_detected = True
            self.error_type = msg.data


# ===================== MAIN =====================
def main(args=None):
    rclpy.init(args=args)
    node = FSMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()