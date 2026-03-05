import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

class FSMNode(Node):
    def __init__(self):
        super().__init__('fsm_controller')

        #Internal Variables
        self.state = "IDLE"
        self.marker_detected = False
        self.marker_count = 0
        self.required_markers = 2
        self.map_explored = False

        #Publishers
        self.state_pub = self.create_publisher(String, '/states', 10)

        #Subscibers


        #Timer
        self.timer = self.create_timer(0.1, self.state_machine_loop)
        self.get_logger().info("FSM Controller Started")
        self.change_state("EXPLORE")

    def change_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
        msg = String()
        msg.data = new_state
        self.state_pub.publish(msg)
        self.get_logger().info(f"Transitioned to {new_state} state")

    def state_machine_loop(self):
        if self.state == "EXPLORE":
            if self.marker_detected:
                self.marker_detected = False
                self.change_state("DOCK")
            
            elif self.map_explored and self.marker_count >= self.required_markers:
                self.change_state("END")
        else:
            if self.state == "END":
                self.get_logger().info("Mission Complete! Goodbye!")
            pass
        
    def main():
        print("Hello World!")
    
    if __name__ == "__main__":
        main()