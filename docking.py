import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
from tf2_ros import Buffer, TransformListener, LookupException
import math
from geometry_msgs.msg import Quaternion

class DockingNode(Node):
    def __init__(self):
        super().__init__('docking_node')

        #Subscribers & Publishers
        self.create_subscription(String, '/states', self.docking_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.dock_complete_pub = self.create_publisher(String, '/operation_status', 10)

        #TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        #Parameters
        self.marker_id = None  # ArUco marker ID for docking
        self.list_of_markers = [2,4,6] #list of markers for positioning, will be removed when done

        self.alignment_threshold = 0.01  # meters, margin of error of robot from location of aruco marker in nav2 before considered aligned
        self.docking_distance = 0.2 #distance to stop away from the marker for docking, in meters
        
        '''
        self.declare_parameter("alignment_threshold",0.05) #metres, margin of error of robot from location of aruco marker 
        self.declare_parameter("docking_distance",0.2) #distance to stop away from the marker for docking, in meters
        self.declare_parameter("verbouse", False)
        '''

        # Control gains for fine alignment
        self.k_linear = 0.5
        self.k_angular = 1.0

        # Docking flow tracking (using variable states, not string states)
        self.docking_timer = None
        self.alignment_iterations = 0

    #Callback for dock command
    def docking_callback(self, msg):
        if msg.data == "DOCK":
            self.get_logger().info("Docking command received")
            self.marker_id = None
            self.alignment_iterations = 0
            # Start main docking loop timer
            if self.docking_timer is None:
                self.docking_timer = self.create_timer(0.1, self.docking_step)
        else:
            return 

    def docking_step(self):
        """Main docking flow - runs every 0.1 seconds"""
        
        # STEP 1: Find marker (if not found yet)
        if self.marker_id is None:
            for i in self.list_of_markers:
                try:
                    self.tf_buffer.lookup_transform('base_link', f'aruco_marker_{i}', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))
                    self.marker_id = i
                    self.get_logger().info(f"Found marker ID: {self.marker_id}")
                    break
                except LookupException:
                    continue
            return  # Still searching
        
        # Fine alignment with TF and cmd_vel
        self.alignment_iterations += 1
        
        # Safety timeout
        if self.alignment_iterations > 100:
            self.get_logger().warn("Fine alignment timeout")
            self.cmd_pub.publish(Twist())
            self._finish_docking()
            return
        
        try:
            transform = self.tf_buffer.lookup_transform('base_link', f'aruco_marker_{self.marker_id}', rclpy.time.Time(seconds=0))
        except LookupException:
            self.get_logger().warn("Lost marker during fine alignment")
            self.cmd_pub.publish(Twist())
            self._finish_docking()
            return
        
        dx = transform.transform.translation.x
        dy = transform.transform.translation.y
        angle_to_marker = math.atan2(dy, dx)
        distance_error = dx - self.docking_distance
        
        # Send cmd_vel for fine alignment
        cmd = Twist()
        cmd.linear.x = self.k_linear * distance_error
        cmd.angular.z = self.k_angular * angle_to_marker
        self.cmd_pub.publish(cmd)
        
        # Check if aligned (STEP 5)
        if abs(dy) < self.alignment_threshold and abs(distance_error) < self.alignment_threshold:
            self.get_logger().info("Fine alignment complete")
            self.cmd_pub.publish(Twist())
            self._finish_docking()
            return
    
    def _finish_docking(self):
        """Cleanup and finish docking"""
        if self.docking_timer is not None:
            self.destroy_timer(self.docking_timer)
            self.docking_timer = None
        
        # Remove completed marker and publish message
        if self.marker_id in self.list_of_markers:
            self.list_of_markers.remove(self.marker_id)
        self.marker_id = None
        self.alignment_iterations = 0
        
        done_msg = String()
        done_msg.data = "DOCK_DONE"
        self.dock_complete_pub.publish(done_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DockingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()