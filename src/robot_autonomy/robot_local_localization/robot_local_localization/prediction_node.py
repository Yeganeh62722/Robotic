import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import numpy as np

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class PredictionNode(Node):
    """
    A node that implements the differential drive kinematics 
    to predict the robot's odometry (pose and velocity) 
    based on received /cmd_vel messages.
    """
    def __init__(self):
        super().__init__('prediction_node')
        self.get_logger().info('Prediction Node started.')

        #Path Publisher for Visualization
        self.motion_model_path_pub = self.create_publisher(Path, '/path/motion_model', 1)

        # History list to store poses
        self.motion_model_path = Path()
        self.motion_model_path.header.frame_id = self.odom_frame_id # 'odom'

        # ----------------------------------------------------
        # 1. Parameter Declaration and Retrieval
        # ----------------------------------------------------
        # The node declares the parameters and sets default values.
        self.declare_parameter('wheel_separation', 0.45)
        self.declare_parameter('wheel_radius', 0.1)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('odom_topic', '/odom/prediction')
        self.declare_parameter('publish_tf', True) # Toggle publishing the TF transform

        # Retrieve parameters
        self.wheel_separation = self.get_parameter('wheel_separation').get_parameter_value().double_value
        self.wheel_radius = self.get_parameter('wheel_radius').get_parameter_value().double_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.odom_frame_id = self.get_parameter('odom_frame_id').get_parameter_value().string_value
        self.base_frame_id = self.get_parameter('base_frame_id').get_parameter_value().string_value
        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.publish_tf = self.get_parameter('publish_tf').get_parameter_value().bool_value
        
        # ----------------------------------------------------
        # 2. Node State Variables
        # ----------------------------------------------------
        # Current estimated pose (x, y, theta)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Current received velocities (linear, angular)
        self.linear_vel = 0.0
        self.angular_vel = 0.0

        # Time variables for integration
        self.last_time = self.get_clock().now()
        
        # TF Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # ----------------------------------------------------
        # 3. Subscribers and Publishers
        # ----------------------------------------------------
        # Subscriber for velocity commands
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_vel_callback,
            10)

        # Publisher for Odometry messages
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)

        # Odometry update timer (running at 50 Hz)
        self.timer = self.create_timer(0.02, self.update_odometry) # 50 Hz

    def cmd_vel_callback(self, msg):
        """
        Stores the latest commanded velocities.
        """
        self.linear_vel = msg.linear.x
        self.angular_vel = msg.angular.z

    def update_odometry(self):
        """
        Calculates the change in pose using differential drive kinematics 
        and publishes the result as an Odometry message and TF transform.
        """
        # Calculate time elapsed since the last update
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # If time step is zero or no velocity command, skip update
        if dt == 0.0 or (self.linear_vel == 0.0 and self.angular_vel == 0.0):
            return

        # ----------------------------------------------------
        # Differential Drive Kinematics (Dead Reckoning)
        # ----------------------------------------------------
        v_lin = self.linear_vel
        v_ang = self.angular_vel

        # Calculate change in pose
        delta_x = v_lin * math.cos(self.theta + v_ang * dt / 2.0) * dt
        delta_y = v_lin * math.sin(self.theta + v_ang * dt / 2.0) * dt
        delta_theta = v_ang * dt
        
        # Update current pose
        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta
        
        # Normalize theta to be between -pi and pi
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        
        # ----------------------------------------------------
        # Odometry Message Creation and Publishing
        # ----------------------------------------------------
        
        # Convert Euler yaw (theta) to Quaternion
        odom_quat = self.euler_to_quaternion(0, 0, self.theta)

        # 1. Publish the TF transform (if enabled)
        if self.publish_tf:
            self.publish_transform(current_time, odom_quat)

        # 2. Publish the Odometry message
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = self.odom_frame_id
        odom.child_frame_id = self.base_frame_id
        
        # Set pose
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = odom_quat[0]
        odom.pose.pose.orientation.y = odom_quat[1]
        odom.pose.pose.orientation.z = odom_quat[2]
        odom.pose.pose.orientation.w = odom_quat[3]
        
        # Set velocity
        odom.twist.twist.linear.x = v_lin
        odom.twist.twist.angular.z = v_ang
        
        # Set covariance (This is an open-loop prediction, so covariance should be high/increasing)
        # For simplicity, we are setting a nominal covariance here.
        odom.pose.covariance = list(np.diag([0.001, 0.001, 1000.0, 1000.0, 1000.0, 1000.0]).flatten())
        odom.twist.covariance = list(np.diag([0.001, 0.001, 1000.0, 1000.0, 1000.0, 1000.0]).flatten())

        self.odom_pub.publish(odom)

        # --- Path Message Publishing ---
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose

        # Add the current pose to the path history
        self.motion_model_path.poses.append(pose)

        # Optional: Limit the path history size to prevent excessive memory use
        MAX_PATH_LENGTH = 1000 
        if len(self.motion_model_path.poses) > MAX_PATH_LENGTH:
            self.motion_model_path.poses.pop(0)

        # Publish the entire path history
        self.motion_model_path_pub.publish(self.motion_model_path)

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Converts Euler roll, pitch, yaw to Quaternion (x, y, z, w)"""
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]
    
    def publish_transform(self, current_time, odom_quat):
        """Broadcasts the transform from odom_frame_id to base_frame_id."""
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = self.odom_frame_id
        t.child_frame_id = self.base_frame_id
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = odom_quat[0]
        t.transform.rotation.y = odom_quat[1]
        t.transform.rotation.z = odom_quat[2]
        t.transform.rotation.w = odom_quat[3]
        
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = PredictionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()