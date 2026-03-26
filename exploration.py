import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist
# from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import numpy as np
import time
from std_msgs.msg import String

#imports for transform
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians

class ExplorerNode(Node):
    def __init__(self):
        super().__init__('explorer')
        self.get_logger().info("Explorer Node Started")

        # Subscriber to the status
        self.state_sub = self.create_subscription(String, '/states', self.status_callback, 10)

        # Subscriber to the map topic
        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10)
        
        # Subscriber to the odometry topic
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(0.5, self.get_robot_pose)

        # Subscriber to the ros2 camera topic
        

        # Publisher for rotation
        self.publisher_ = self.create_publisher(Twist,'cmd_vel',10)



        # Action client for navigation
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Visited frontiers set
        self.visited_frontiers = set()

        # Map and position data
        self.map_data = None
        self.robot_position = (0, 0)  # Placeholder, update from localization

        
        # Timer for periodic exploration
        self.timer = self.create_timer(0.5, self.explore)

        self.flag = True  # Flag to indicate navigation completion
        self.time = 0
        self.explorationtime = time.time()
        self.tvec = None
        self.status_flag = False

    # for odometry
    # def odom_callback(self, msg):
    #     # self.get_logger().info('In odom_callback')
    #     orientation_quat =  msg.pose.pose.orientation
    #     self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w)
    #     position = msg.pose.pose.position
    #     self.robot_position = (position.x, position.y)  # Update robot position (row)


    def status_callback(self, msg):
        # self.get_logger().info(f"Received status: {msg.data}")
        if msg.data == "EXPLORE": 
            self.status_flag = True
            # self.get_logger().info("Status set to EXPLORE")
        else:
            self.status_flag = False
            # self.get_logger().info("Status set to non-EXPLORE")
    
    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            roll, pitch, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
            self.robot_position = (x, y)  # Update robot position (row)        
            return None    
        except Exception:
            return None, None, None

    def get_tvec(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'camera_link', rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z
            self.tvec = (x, y, z)
            return None
        except Exception:
            return None

    def map_callback(self, msg):
        self.map_data = msg
        # self.get_logger().info("Map received")

    def navigate_to(self, x, y):
        """
        Send navigation goal to Nav2.
        """

        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.orientation.w = 1.0  # Facing forward

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = goal_msg

        robot_x, robot_y = self.robot_position
        distance_to_goal = np.sqrt((robot_x - x)**2 + (robot_y - y)**2)
        self.get_logger().info(f"Robot at: ({robot_x:.2f}, {robot_y:.2f}), Navigating to goal: x={x:.2f}, y={y:.2f}, distance={distance_to_goal:.2f}m")
        if distance_to_goal < 0.5:
            self.flag = True  # Set flag to indicate navigation is complete if already close to the goal
        # Wait for the action server
        self.nav_to_pose_client.wait_for_server()

        # Send the goal and register a callback for the result
        send_goal_future = self.nav_to_pose_client.send_goal_async(nav_goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Handle the goal response and attach a callback to the result.
        """
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warning("Goal rejected!")
            return

        self.get_logger().info("Goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_complete_callback)

    def navigation_complete_callback(self, future):
        """
        Callback to handle the result of the navigation action.
        """
        try:
            result = future.result().result
            status = future.result().status
            self.get_logger().info(f"Navigation status: {status}, result: {result}")
            # self.flag = True  # Set flag to indicate navigation is complete
        except Exception as e:
            self.get_logger().error(f"Navigation failed: {e}")


    def find_frontiers(self, map_array):
        """
        Detect frontiers in the occupancy grid map.
        """
        np.savetxt("map_array.txt", map_array)  # Save the map array for debugging
        frontiers = []
        rows, cols = map_array.shape

        # if -1 in map_array:
        #     self.get_logger().info("Frontiers detected in the map")
        min_val = 100
        # Iterate through each cell in the map
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                # self.get_logger().info(f"Checking cell ({r}, {c}): value={map_array[r, c]}")
                if map_array[r, c] != -1 and  map_array[r, c] < min_val:
                        min_val = map_array[r, c]

                if 0 <= map_array[r, c] <= 49:  # Free cell IMPORTANT TO ADJUST
                    # Check if any neighbors are unknown
                    neighbors = map_array[r-1:r+2, c-1:c+2].flatten()
                    if -1 in neighbors:
                        frontiers.append((r, c))


        # self.get_logger().info(f"Minimum cell value in the map: {min_val}")
        


        # self.get_logger().info(f"Found {len(frontiers)} frontiers")
        return frontiers

    def choose_frontier(self, frontiers):
        """
        Choose the frontier with the most neighboring frontiers (cluster preference),
        with distance as a tiebreaker.
        """
        robot_x, robot_y = self.robot_position
        self.get_logger().info(f"Robot position: {self.robot_position}")
        
        neighbor_threshold = 0.5  # Distance in meters to consider frontiers as neighbors
        
        # Calculate neighbor count and distance for each frontier
        frontier_scores = []
        
        for frontier in frontiers:
            if frontier in self.visited_frontiers:
                continue
            
            # Calculate distance to robot
            position_x = frontier[1] * self.map_data.info.resolution + self.map_data.info.origin.position.x
            position_y = frontier[0] * self.map_data.info.resolution + self.map_data.info.origin.position.y
            distance_to_robot = np.sqrt((robot_x - position_x)**2 + (robot_y - position_y)**2)
            
            if distance_to_robot < 0.5:  # Skip frontiers too close to robot
                continue
            
            # Count neighboring frontiers (clustering)
            neighbor_count = 0
            for other_frontier in frontiers:
                if other_frontier == frontier or other_frontier in self.visited_frontiers:
                    continue
                
                other_x = other_frontier[1] * self.map_data.info.resolution + self.map_data.info.origin.position.x
                other_y = other_frontier[0] * self.map_data.info.resolution + self.map_data.info.origin.position.y
                
                distance_to_neighbor = np.sqrt((position_x - other_x)**2 + (position_y - other_y)**2)
                if distance_to_neighbor <= neighbor_threshold:
                    neighbor_count += 1
            
            # Store: (neighbor_count, distance_to_robot, frontier)
            frontier_scores.append((neighbor_count, distance_to_robot, frontier))
        
        if not frontier_scores:
            self.get_logger().warning("No valid frontier found")
            return None
        
        # Sort by: most neighbors first (descending), then closest distance (ascending)
        frontier_scores.sort(key=lambda x: (-x[0], x[1]))
        
        chosen_frontier = frontier_scores[0][2]
        neighbor_count, distance = frontier_scores[0][0], frontier_scores[0][1]
        
        self.visited_frontiers.add(chosen_frontier)
        self.get_logger().info(f"Chosen frontier: {chosen_frontier} with {neighbor_count} neighbors, distance: {distance:.2f}m")

        return chosen_frontier

    def explore(self):
        #if self.status_flag:
            if self.map_data is None:
                self.get_logger().warning("No map data available")
                return

            # Convert map to numpy array
            map_array = np.array(self.map_data.data).reshape(
                (self.map_data.info.height, self.map_data.info.width))

            # Detect frontiers
            frontiers = self.find_frontiers(map_array)

            # if time.time() - self.explorationtime < 120:
            #     # self.get_logger().info("Exploration in progress...")
            # else:
            #     if len(frontiers) == 0:
            #         self.get_logger().info("No frontiers found. Exploration complete!")
            #         self.timer.cancel()
            #         self.stopbot()
            #         rclpy.shutdown()

            #     # self.shutdown_robot()
            #         return

            # Choose the closest frontier
            chosen_frontier = self.choose_frontier(frontiers)

            if not chosen_frontier:
                self.get_logger().warning("No frontiers to explore")
                return

            # Convert the chosen frontier to world coordinates
            goal_x = chosen_frontier[1] * self.map_data.info.resolution + self.map_data.info.origin.position.x
            goal_y = chosen_frontier[0] * self.map_data.info.resolution + self.map_data.info.origin.position.y

            # Navigate to the chosen frontier once goal is reached or after 10 seconds
            if self.flag or time.time() - self.time > 10:
                self.navigate_to(goal_x, goal_y)
                self.flag = False
                self.time = time.time()

            return

    # def shudown_robot(self):
    #     
    #
    #
    #     self.get_logger().info("Shutting down robot exploration")

    def stopbot(self):
        self.get_logger().info('In stopbot')
        # publish to cmd_vel to move TurtleBot
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        # time.sleep(1)
        self.publisher_.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    explorer_node = ExplorerNode()

    try:
        explorer_node.get_logger().info("Starting exploration...")
        rclpy.spin(explorer_node)
    except KeyboardInterrupt:
        explorer_node.get_logger().info("Exploration stopped by user")
    finally:
        explorer_node.destroy_node()
        rclpy.shutdown()