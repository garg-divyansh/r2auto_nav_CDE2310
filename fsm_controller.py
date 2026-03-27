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
        self.marker_id = None  # store current marker id

        # Error handling
        self.error_detected = False
        self.error_type = None

        # ================= PUBLISHERS =================
        self.state_pub = self.create_publisher(String, '/states', 10)
        self.current_marker_pub = self.create_publisher(Int32,'/current_marker', 10)

        # ================= SUBSCRIBERS =================
        self.create_subscription(PoseStamped, '/aruco_pose', self.aruco_callback, 10)
<<<<<<< HEAD
        self.create_subscription(String, '/operation_status', self.status_callback, 10)
=======
        self.create_subscription(Bool, '/dock_done', self.dock_done_callback, 10)
        self.create_subscription(Bool, '/launch_done', self.launch_done_callback, 10)
        self.create_subscription(Bool, '/map_explored', self.map_explored_callback, 10)
>>>>>>> nikidudu_branch

        # ================= TIMER =================
        self.timer = self.create_timer(0.1, self.state_machine_loop)

        self.get_logger().info("FSM Controller Started")
        self.change_state("EXPLORE")

    # ================= STATE TRANSITION =================
    def change_state(self, new_state, marker_id=None):
        """Change FSM state. Optionally include marker_id in the state string."""
        if self.state != new_state or marker_id is not None:
            self.prev_state = self.state
            self.state = new_state

            msg = String()
            # Only append marker ID for DOCK or LAUNCH states
            if marker_id is not None and new_state in ["DOCK", "LAUNCH"]:
                msg.data = f"{new_state}_{marker_id}"
            else:
                msg.data = new_state

            self.state_pub.publish(msg)
            self.get_logger().info(f"Transitioned to {msg.data} state")

    # ================= FSM LOOP =================
    def state_machine_loop(self):

        # Priority: handle errors first
        if self.error_detected:
            self.handle_error()
            return

        if self.state == "EXPLORE":
            if self.marker_detected:
                self.marker_detected = False
                if self.current_marker:
                    marker_id = int(self.current_marker.header.frame_id.split("_")[-1])
                    self.marker_id = marker_id
                    self.change_state("DOCK", marker_id)

            elif self.map_explored and self.marker_count >= self.required_markers:
                self.change_state("END")
<<<<<<< HEAD

        elif self.state == "DOCK":
=======
            # self.change_state("EXPLORE")
        else:
            if self.state == "END":
                self.get_logger().info("Mission Complete! Goodbye!")
>>>>>>> nikidudu_branch
            pass

        elif self.state == "LAUNCH":
            pass

        elif self.state == "END":
            self.get_logger().info("Mission Complete! Goodbye!")

    # ================= ERROR HANDLER =================
    def handle_error(self):
        self.get_logger().error(f"Handling error: {self.error_type}")

        if self.error_type == "DOCK_FAIL":
            self.get_logger().warn("Docking failed → retry docking")
            if self.marker_id is not None:
                self.change_state("DOCK", self.marker_id)
            else:
                self.change_state("DOCK")

        elif self.error_type == "NAV_FAIL":
            self.get_logger().warn("Navigation failed → return to explore")
            self.change_state("EXPLORE")

        elif self.error_type == "LAUNCH_FAIL":
            self.get_logger().warn("Launch failed → retry launch")
            if self.marker_id is not None:
                self.change_state("LAUNCH", self.marker_id)
            else:
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

    # ================= CALLBACKS =================
    def aruco_callback(self, msg):
        if self.state == "EXPLORE":
            self.get_logger().info("Marker Detected")
            self.marker_detected = True

            self.current_marker = msg
            self.marker_id = int(self.current_marker.header.frame_id.split("_")[-1])

            # Publish marker ID separately as Int32 (optional for other nodes)
            marker_msg = Int32()
            marker_msg.data = self.marker_id
            self.current_marker_pub.publish(marker_msg)

    def status_callback(self, msg):
        status = msg.data
        self.get_logger().info(f"Status received: {status}")

        # ================= SUCCESS CASES =================
        if status == "DOCK_DONE" and self.state == "DOCK":
            self.get_logger().info("Docking completed")
            self.current_marker = None
            self.change_state("LAUNCH", self.marker_id)

        elif status == "LAUNCH_DONE" and self.state == "LAUNCH":
            self.get_logger().info("Launch completed")
            self.marker_count += 1
            self.change_state("EXPLORE")

        elif status == "MAP_DONE" and self.state == "EXPLORE":
            self.get_logger().info("Map exploration completed")
            self.map_explored = True

<<<<<<< HEAD
        # ================= ERROR CASES =================
        elif status in ["DOCK_FAIL", "LAUNCH_FAIL", "NAV_FAIL", "MARKER_LOST", "TIMEOUT"]:
            self.error_detected = True
            self.error_type = status

# ================= MAIN =================
=======
>>>>>>> nikidudu_branch
def main(args=None):
    rclpy.init(args=args)
    node = FSMNode()
    rclpy.spin(node)
    node.destroy_node()
<<<<<<< HEAD
    rclpy.shutdown()

=======
    rclpy.shutdown()        
    
>>>>>>> nikidudu_branch
if __name__ == "__main__":
    main()