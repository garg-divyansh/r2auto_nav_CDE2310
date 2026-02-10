# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
import numpy as np
import RPi.GPIO as GPIO
import time

class Scanner(Node):

    def __init__(self):
        super().__init__('scanner')

        # Command line configurable pinmap, solenoid pulse duration & distance threshold
        self.declare_parameter('servo_pin', 18)
        self.declare_parameter('solenoid_pin', 23)
        self.declare_parameter('solenoid_pulse_s', 0.5)
        self.declare_parameter('distance_threshold_m', 1)
        self.servo_pin = self.get_parameter('servo_pin').get_parameter_value().integer_value
        self.solenoid_pin = self.get_parameter('solenoid_pin').get_parameter_value().integer_value
        self.solenoid_pulse_s = self.get_parameter('solenoid_pulse_s').get_parameter_value().double_value
        self.distance_threshold_m = self.get_parameter('distance_threshold_m').get_parameter_value().double_value
        
        # Subscription to LIDAR
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.listener_callback,
            qos_profile_sensor_data)
        self.subscription  # prevent unused variable warning

        # Set pin by BCM (Broadcom SOC channel)
        GPIO.setmode(GPIO.BCM)
        
        # Set the servo, solenoid pin as output
        GPIO.setup(self.solenoid_pin, GPIO.OUT)
        GPIO.setup(self.servo_pin, GPIO.OUT)

        # Initialise the servo to be controlled by pwm with 50 Hz frequency
        self.servo_pwm = GPIO.PWM(self.servo_pin, 50)

        # Start pwm
        self.servo_pwm.start(0)

        # State flag
        self.completed_flag = False

    def listener_callback(self, msg):
        # create numpy array
        laser_range = np.array(msg.ranges)
        # replace 0's with nan
        laser_range[laser_range==0] = np.nan
        # find index with minimum value
        lr2i = np.nanargmin(laser_range)

        shortest_distance = laser_range[lr2i]

        self.get_logger().info(
            f'Shortest distance is {shortest_distance}'
        )
        
        # Check if object is closer than threshold distance, and actuate once
        if (shortest_distance is not None) and (shortest_distance < self.distance_threshold_m):
            if not self.completed_flag:
                self.completed_flag = True
                self.get_logger().info(f'Object too close! {shortest_distance}m')
                self.servo_moveTo(45)
                self.pulse_solenoid(self.solenoid_pulse_s)
    
        elif (shortest_distance is not None) and self.completed_flag:
            self.get_logger().info('No objects nearby.')
            self.completed_flag = False
            self.servo_moveTo(0)

    # @brief Rotate servo to angle (degrees)
    # @note Blocks for 0.5s for servo to rotate
    def servo_moveTo(self, angle_deg):
        # Constrain angle to [0, 180]
        angle_deg = max(0, min(180, angle_deg))
        self.servo_pwm.ChangeDutyCycle(2.5 + angle_deg/180 * 10)

    # @brief Actuate solenoid for specified duration (seconds)
    # @note Blocks until actuation complete, then blocks another 0.5s for recovery
    def pulse_solenoid(self, pulse_s):
        GPIO.output(self.solenoid_pin, GPIO.HIGH)
        time.sleep(pulse_s)
        GPIO.output(self.solenoid_pin, GPIO.LOW)
        time.sleep(0.5)

def main(args=None):
    rclpy.init(args=args)

    scanner = Scanner()

    rclpy.spin(scanner)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    scanner.destroy_node()
    rclpy.shutdown() 


if __name__ == '__main__':
    main()