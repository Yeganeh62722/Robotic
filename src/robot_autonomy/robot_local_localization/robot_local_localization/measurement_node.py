import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistStamped
import math

class MeasurementNode(Node):
    """
    Combines linear velocity from Visual Odometry (VO) and angular velocity 
    from the IMU to create a refined TwistStamped measurement input for EKF.
    """
    def __init__(self):
        super().__init__('measurement_node')
        self.get_logger().info('Measurement Combination Node started.')

        # ----------------------------------------------------
        # 1. Node State Variables
        # ----------------------------------------------------
        self.vo_linear_x = 0.0
        self.imu_angular_z = 0.0
        
        # Last received timestamp for both messages (used for synchronization checks)
        self.last_imu_stamp = self.get_clock().now()
        self.last_vo_stamp = self.get_clock().now()

        # ----------------------------------------------------
        # 2. Subscribers and Publishers
        # ----------------------------------------------------
        
        # Subscriber for Visual Odometry (from rtabmap_vo_node)
        # We extract linear velocity (x) from this
        self.vo_sub = self.create_subscription(
            Odometry,
            '/vo/odom', # Matches the remapping in your launch file
            self.vo_callback,
            10)

        # Subscriber for IMU data
        # We extract angular velocity (z) from this
        self.imu_sub = self.create_subscription(
            Imu,
            '/zed/zed_node/imu/data_raw', # IMU topic from your URDF/GZ
            self.imu_callback,
            10)

        # Publisher for the Combined Measurement (TwistStamped)
        # This will be the observation input for your EKF
        self.combined_pub = self.create_publisher(TwistStamped, '/odom/combined_measurement', 10)

        # Timer to publish the combined measurement at a regular rate (e.g., 50 Hz)
        self.timer = self.create_timer(0.02, self.publish_combined_measurement) # 50 Hz

    def vo_callback(self, msg: Odometry):
        """
        Processes Visual Odometry message and extracts linear velocity (x).
        """
        # The velocity is stored in the twist section of the Odometry message
        self.vo_linear_x = msg.twist.twist.linear.x
        self.last_vo_stamp = self.get_clock().now()
        self.get_logger().debug(f"VO Linear X: {self.vo_linear_x}")


    def imu_callback(self, msg: Imu):
        """
        Processes IMU message and extracts angular velocity (z).
        """
        # The angular velocity is stored in the angular_velocity section
        self.imu_angular_z = msg.angular_velocity.z
        self.last_imu_stamp = self.get_clock().now()
        self.get_logger().debug(f"IMU Angular Z: {self.imu_angular_z}")

    def publish_combined_measurement(self):
        """
        Combines the latest VO linear velocity and IMU angular velocity
        into a single TwistStamped message and publishes it.
        """
        
        # Check if data is recent (optional but recommended sync check)
        time_since_vo = (self.get_clock().now() - self.last_vo_stamp).nanoseconds / 1e9
        time_since_imu = (self.get_clock().now() - self.last_imu_stamp).nanoseconds / 1e9

        # If data is older than 0.5 seconds, log a warning and don't publish
        if time_since_vo > 0.5 or time_since_imu > 0.5:
             self.get_logger().warn("Stale sensor data detected. Not publishing combined measurement.")
             return

        # ----------------------------------------------------
        # Combined Measurement Model (The fusion step)
        # ----------------------------------------------------
        
        # 1. Linear velocity is taken from the Visual Odometry (VO)
        combined_linear_x = self.vo_linear_x
        
        # 2. Angular velocity is taken from the IMU
        combined_angular_z = self.imu_angular_z
        
        # ----------------------------------------------------
        # Publishing
        # ----------------------------------------------------
        combined_msg = TwistStamped()
        combined_msg.header.stamp = self.get_clock().now().to_msg()
        combined_msg.header.frame_id = "base_link" # The velocity is measured relative to the robot's frame

        combined_msg.twist.linear.x = combined_linear_x
        combined_msg.twist.angular.z = combined_angular_z

        self.combined_pub.publish(combined_msg)
        self.get_logger().debug(f"Published Combined: L={combined_linear_x:.3f}, A={combined_angular_z:.3f}")


def main(args=None):
    rclpy.init(args=args)
    node = MeasurementNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()