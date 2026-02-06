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

    def init(self):
        super().init('scanner')
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.listener_callback,
            qos_profile_sensor_data)
        self.subscription  # prevent unused variable warning
        self.shortest_distance = None
        # Set pin numbering convention
        GPIO.setmode(GPIO.BCM)
        # Choose an appropriate pwm channel to be used to control the servo
        self.servo_pin = 18
        # Set the pin as an output
        GPIO.setup(self.servo_pin, GPIO.OUT)
        # Initialise the servo to be controlled by pwm with 50 Hz frequency
        self.p = GPIO.PWM(self.servo_pin, 50)
        # Set servo to 90 degrees as it's starting position
        self.p.start(2.5)

    def listener_callback(self, msg):
        # create numpy array
        laser_range = np.array(msg.ranges)
        # replace 0's with nan
        laser_range[laser_range==0] = np.nan
        # find index with minimum value
        lr2i = np.nanargmin(laser_range)

        # log the info
        angle_rad = msg.angle_min + lr2i * msg.angle_increment
        angle_deg = np.degrees(angle_rad)

        self.shortest_distance = laser_range[lr2i]

        self.get_logger().info(
            f'Shortest distance at {laser_range[lr2i]} degrees'
        )
        
        self.servospin()

        # self.get_logger().info('Shortest distance at %i degrees' % lr2i)

    def servospin(self):
        short = self.shortest_distance
        if short is not None:
            if short < 1:
                self.p.ChangeDutyCycle(5)  # turn servo to 90 degrees
                self.get_logger().info('Object too close! Stopping robot.')
            else:
                self.p.ChangeDutyCycle(2.5)  # turn servo to 0 degrees
                self.get_logger().info('Path is clear. Continuing.')

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