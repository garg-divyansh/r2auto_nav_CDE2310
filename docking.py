import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException, ConnectivityException
import math

"""
docking_node.py

ROS2 node for autonomous docking of a TurtleBot3 to an ArUco marker using TF2 transforms.

The node subscribes to /states for docking commands in the format 'DOCK_<marker_id>',
then performs closed-loop alignment by querying the TF transform of the target ArUco
marker relative to base_link and publishing velocity commands to /cmd_vel.

Alignment is done in two stages: angular correction first, then linear approach.
Docking is considered complete when the robot holds position within the alignment
threshold for 0.5 seconds. On completion, 'DOCK_DONE' is published to /operation_status.

Parameters:
    alignment_threshold (float): Positional tolerance in metres to consider aligned (default: 0.05)
    docking_distance (float):    Distance in metres to stop from the marker (default: 0.2)
    k_linear (float):            Proportional gain for linear velocity control (default: 0.5)
    k_angular (float):           Proportional gain for angular velocity control (default: 1.0)
    angular_threshold (float):   Angular error in radians below which linear motion begins (default: 0.05)
    verbose (bool):              Enable debug logging (default: False)
"""

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
        self.aligned_iterations = 0 #check how many iterations robot is within alignment tolerance for aruco_marker
        self.consecutive_misses = 0 #check how many iterations in a row the robot has lost track of the marker (maybe too close or just lost it)
        
        # Declare parameters with default values
        self.declare_parameter("alignment_threshold", 0.05)  # metres, margin of error of robot from location of aruco marker
        self.declare_parameter("docking_distance", 0.2)  # distance to stop away from the marker for docking, in meters
        self.declare_parameter("k_linear", 0.5)  # linear control gain
        self.declare_parameter("k_angular", 1.0)  # angular control gain
        self.declare_parameter("verbose", False)  # enable verbose logging
        self.declare_parameter("angular_threshold", 0.05)  # rad, threshold to switch from angular correction to linear movement
        
        # Retrieve parameters
        self.alignment_threshold = self.get_parameter("alignment_threshold").value
        self.docking_distance = self.get_parameter("docking_distance").value
        self.k_linear = self.get_parameter("k_linear").value
        self.k_angular = self.get_parameter("k_angular").value
        self.verbose = self.get_parameter("verbose").value
        self.angular_threshold = self.get_parameter("angular_threshold").value
        
        if self.verbose:
            self.get_logger().info(f"Loaded parameters: alignment_threshold={self.alignment_threshold}, "
                                f"docking_distance={self.docking_distance}, k_linear={self.k_linear}, "
                                f"k_angular={self.k_angular}, verbose={self.verbose}")
        
        
        # Docking flow tracking 
        self.docking_timer = None
        self.alignment_iterations = 0

    #Callback for dock command
    def docking_callback(self, msg):
        try:
            message, number = msg.data.split("_",1)
        except ValueError:
            return
        
        if message == "DOCK":
            if self.verbose:
                self.get_logger().info("Docking command received")
            self.marker_id = int(number)
            self.alignment_iterations = 0
            # Start main docking loop timer
            if self.docking_timer is None:
                self.docking_timer = self.create_timer(0.1, self.docking_step)
        else:
            return 

    def docking_step(self):
        #Main docking flow runs every 0.1 seconds
        # Fine alignment with TF and cmd_vel
        self.alignment_iterations += 1
        
        # Safety timeout
        if self.alignment_iterations > 300:
            if self.verbose:
                self.get_logger().warn("Fine alignment timeout")
            self.cmd_pub.publish(Twist())
            self._finish_docking()
            return
        
        try:
            transform = self.tf_buffer.lookup_transform('base_link', f'aruco_marker_{self.marker_id}', rclpy.time.Time())
            self.consecutive_misses = 0
        except (LookupException, ExtrapolationException, ConnectivityException):
            self.consecutive_misses += 1
            self.cmd_pub.publish(Twist())  # stop every miss
            if self.consecutive_misses > 5:
                self.get_logger().warn("Marker lost, aborting")
                self._finish_docking()
            return
        
        dx = transform.transform.translation.x #distance in forward direction
        dy = transform.transform.translation.y #distance sideways direction, positive is left, negative is right
        angle_to_marker = math.atan2(dy, dx)
        distance = math.sqrt(dx**2 + dy**2) #displacement to aruco
        distance_error = distance - self.docking_distance

        #send out cmd_vel for fine alignment
        cmd = Twist()
        max_linear = 0.15
        max_angular = 0.5

        if abs(angle_to_marker) > self.angular_threshold:
            cmd.linear.x = 0.0 # don't move forward yet
            cmd.angular.z = self.k_angular * angle_to_marker #only adjust angle
        else:
            cmd.linear.x = self.k_linear * distance_error   # only drive once pointing at it
            cmd.angular.z = self.k_angular * angle_to_marker # small corrections still ok
        
        cmd.linear.x = max(-max_linear, min(max_linear, cmd.linear.x))
        cmd.angular.z = max(-max_angular, min(max_angular, cmd.angular.z))
        self.cmd_pub.publish(cmd)

        # Check if aligned 
        if abs(dy) < self.alignment_threshold and abs(distance_error) < self.alignment_threshold:
            self.aligned_iterations += 1
            if self.aligned_iterations >= 5:  # must hold alignment for 0.5s
                self.cmd_pub.publish(Twist())
                self._finish_docking()
                return
        else:
            self.aligned_iterations = 0  # reset if it drifts out
    
    def _finish_docking(self):
        """Cleanup and finish docking"""
        if self.docking_timer is not None:
            self.destroy_timer(self.docking_timer)
            self.docking_timer = None
        
        # Remove completed marker and publish message
        self.marker_id = None
        self.alignment_iterations = 0
        self.aligned_iterations = 0
        self.consecutive_misses = 0
        
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