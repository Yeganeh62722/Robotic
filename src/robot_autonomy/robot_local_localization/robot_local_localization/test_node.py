import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math

class RectangleFollowerNode(Node):
    """
    Subscribes to EKF fused odometry, calculates the required velocity commands 
    to follow a rectangular path, and publishes them to /cmd_vel.
    """
    def __init__(self):
        super().__init__('rectangle_follower_node')
        self.get_logger().info('Rectangle Follower Node started.')

        # ----------------------------------------------------
        # 1. Path Definition and Parameters
        # ----------------------------------------------------
        
        # Target rectangle dimensions (side length in meters)
        self.SIDE_LENGTH = 3.0 

        # Control Constants
        self.LINEAR_VELOCITY = 0.5  # m/s
        self.ANGULAR_VELOCITY = 0.5 # rad/s
        self.TOLERANCE_DISTANCE = 0.1 # Distance tolerance for reaching a point (m)
        self.TOLERANCE_ANGLE = 0.05 # Angle tolerance for turning (rad)

        # Path waypoints (Defining a 3x3 square starting at (0, 0))
        # The goal state is [x, y, yaw]
        self.waypoints = [
            {'x': self.SIDE_LENGTH, 'y': 0.0, 'yaw': 0.0},
            {'x': self.SIDE_LENGTH, 'y': self.SIDE_LENGTH, 'yaw': math.pi / 2.0},
            {'x': 0.0, 'y': self.SIDE_LENGTH, 'yaw': math.pi},
            {'x': 0.0, 'y': 0.0, 'yaw': -math.pi / 2.0} # Last segment turns back to start
        ]
        
        self.current_waypoint_index = 0
        self.state = 'TURNING_TO_START' # Initial state for the finite state machine

        # ----------------------------------------------------
        # 2. Node State (Current Robot Pose)
        # ----------------------------------------------------
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # ----------------------------------------------------
        # 3. Subscribers and Publishers
        # ----------------------------------------------------
        
        # Subscriber for Fused Odometry (from EKF)
        self.create_subscription(
            Odometry,
            '/odom/fused', # Subscribing to the EKF output
            self.odometry_callback,
            10)

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Control loop timer (Runs faster than EKF/Sensors for responsiveness)
        self.timer = self.create_timer(0.01, self.control_loop) # 100 Hz control loop

    def odometry_callback(self, msg: Odometry):
        """
        Updates the robot's current pose based on EKF feedback.
        """
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Convert Quaternion to Yaw (Euler z-axis angle)
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(
            2.0 * (q.z * q.w + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def control_loop(self):
        """
        Implements the finite state machine to navigate between waypoints.
        """
        # Stop if no odometry data received yet
        if self.current_x is None:
            return

        cmd_vel_msg = Twist()
        target = self.waypoints[self.current_waypoint_index]

        # ----------------------------------------------------
        # State Machine Implementation
        # ----------------------------------------------------
        if self.state == 'TURNING_TO_START':
            # Calculate the required angle to face the first waypoint
            angle_to_target = math.atan2(target['y'] - self.current_y, target['x'] - self.current_x)
            angle_diff = angle_to_target - self.current_yaw
            
            # Normalize angle difference to [-pi, pi]
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

            if abs(angle_diff) > self.TOLERANCE_ANGLE:
                # Still turning: publish angular velocity
                cmd_vel_msg.angular.z = self.ANGULAR_VELOCITY if angle_diff > 0 else -self.ANGULAR_VELOCITY
                self.get_logger().debug(f"State: Turning | Diff: {angle_diff:.2f}")
            else:
                # Finished turning, switch to moving state
                self.state = 'MOVING_TO_POINT'
                self.get_logger().info(f"Target Yaw reached. Moving to point {self.current_waypoint_index}")

        elif self.state == 'MOVING_TO_POINT':
            # Calculate distance to the target point
            distance = math.sqrt((target['x'] - self.current_x)**2 + (target['y'] - self.current_y)**2)
            
            if distance > self.TOLERANCE_DISTANCE:
                # Still moving: publish linear velocity
                cmd_vel_msg.linear.x = self.LINEAR_VELOCITY
                self.get_logger().debug(f"State: Moving | Distance: {distance:.2f}")

                # Minor correction: constantly adjust yaw while moving
                angle_to_target = math.atan2(target['y'] - self.current_y, target['x'] - self.current_x)
                angle_diff = angle_to_target - self.current_yaw
                angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
                cmd_vel_msg.angular.z = 0.5 * angle_diff # Simple P-controller for angular correction

            else:
                # Finished moving, switch to next waypoint
                self.get_logger().info(f"Waypoint {self.current_waypoint_index} reached.")
                self.current_waypoint_index += 1

                if self.current_waypoint_index >= len(self.waypoints):
                    self.state = 'FINISHED'
                    self.get_logger().info("Rectangle path finished. Stopping.")
                else:
                    self.state = 'TURNING_TO_START' # Start turning towards the new target
                    
        elif self.state == 'FINISHED':
            # Stop the robot
            cmd_vel_msg.linear.x = 0.0
            cmd_vel_msg.angular.z = 0.0
            
        # Publish the command
        self.cmd_vel_pub.publish(cmd_vel_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RectangleFollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()