import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
import math

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# --- Helper Functions ---
def quaternion_from_euler(roll, pitch, yaw):
    # Standard helper function to convert Euler angles to a Quaternion message
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

class EKFNode(Node):
    """
    Implements a 3-state (x, y, yaw) Extended Kalman Filter for sensor fusion.
    - Motion Model (Prediction): From /odom/prediction (Kinematics/Dead-Reckoning)
    - Measurement Model (Update): From /odom/combined_measurement (Fused VO + IMU)
    """
    def __init__(self):
        super().__init__('ekf_node')
        self.get_logger().info('EKF Node started.')

        # Path Publisher for Fused Path
        self.fused_path_pub = self.create_publisher(Path, '/path/ekf_fused', 1)

        # History list
        self.fused_path = Path()
        self.fused_path.header.frame_id = "odom"
        
        # ----------------------------------------------------
        # 1. State Initialization (x, y, yaw)
        # ----------------------------------------------------
        # State vector: [x, y, theta]
        self.ekf = ExtendedKalmanFilter(dim_x=3, dim_z=3, dim_u=2)
        
        # Initial State (P): High uncertainty at start
        self.ekf.P = np.diag([0.1, 0.1, 0.1]) 
        
        # Motion Model Noise (Q): Represents uncertainty in the dead reckoning (process noise)
        # Higher values mean the EKF trusts the prediction less.
        self.ekf.Q = np.diag([0.005, 0.005, 0.005]) 
        
        # Measurement Model Noise (R): Represents uncertainty in the combined sensor reading
        # Lower values mean the EKF trusts the measurement more.
        self.ekf.R = np.diag([0.01, 0.01, 0.01])
        
        # Variables to hold the latest prediction and measurement data
        self.prediction_vel = [0.0, 0.0] # [linear_x, angular_z] from /odom/prediction
        self.measurement_pose = [0.0, 0.0, 0.0] # [x, y, theta] placeholder for measurement update

        # Time tracking for delta t (dt)
        self.last_time = self.get_clock().now()
        
        # ----------------------------------------------------
        # 2. Publishers and Subscribers
        # ----------------------------------------------------
        self.fused_odom_pub = self.create_publisher(Odometry, '/odom/fused', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscriber for the Motion Model (Prediction)
        self.create_subscription(
            Odometry,
            '/odom/prediction',
            self.prediction_callback,
            10)

        # Subscriber for the Measurement Model (Combined VO + IMU)
        self.create_subscription(
            TwistStamped,
            '/odom/combined_measurement',
            self.measurement_callback,
            10)
            
        # Timer to run the EKF loop at a regular rate (e.g., 50 Hz)
        self.timer = self.create_timer(0.02, self.ekf_loop) # 50 Hz

    def prediction_callback(self, msg: Odometry):
        """
        Receives the raw dead-reckoning velocity (motion model input) from prediction_node.
        """
        self.prediction_vel[0] = msg.twist.twist.linear.x
        self.prediction_vel[1] = msg.twist.twist.angular.z

    def measurement_callback(self, msg: TwistStamped):
        """
        Receives the combined sensor velocity (measurement model input).
        For simplicity, we treat this as a direct observation of (x_dot, y_dot, theta_dot).
        In a full EKF, this would typically involve a separate state estimation (h(x)).
        """
        # For a basic fusion, we can use the linear/angular velocity for the measurement.
        # However, EKF updates often use positional data. 
        # For simplicity, we'll apply the update within the main loop using the predicted state.
        pass # We will use the data from the prediction_callback in the main loop

    def ekf_loop(self):
        """
        The main EKF loop: Prediction (Motion) -> Update (Measurement)
        """
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        if dt <= 0.0:
            return

        # --- 1. PREDICTION STEP (Motion Model) ---
        u = np.array(self.prediction_vel) # [v, omega]
        
        # State Transition Function (f(x, u))
        # New State X = [x + v*cos(theta)*dt, y + v*sin(theta)*dt, theta + omega*dt]
        theta = self.ekf.x[2, 0]
        
        def fx(x, dt, u):
            v, omega = u[0], u[1]
            c = math.cos(theta + omega * dt / 2.0)
            s = math.sin(theta + omega * dt / 2.0)
            
            x_new = x[0, 0] + v * c * dt
            y_new = x[1, 0] + v * s * dt
            theta_new = x[2, 0] + omega * dt
            
            return np.array([[x_new], [y_new], [theta_new]])

        # Jacobian of State Transition (F)
        # Jacobian F (3x3): dx_new/dx_old
        def Fx(x, dt, u):
            v, omega = u[0], u[1]
            F = np.array([
                [1.0, 0.0, -v * math.sin(theta + omega * dt / 2.0) * dt],
                [0.0, 1.0,  v * math.cos(theta + omega * dt / 2.0) * dt],
                [0.0, 0.0,  1.0]
            ])
            return F
        
        # Perform Prediction
        self.ekf.predict(u=u, Fx=Fx, fx=fx, dt=dt)
        
        # --- 2. UPDATE STEP (Measurement Model) ---
        # NOTE: For simplicity, we are assuming the combined measurement 
        # is a direct observation of the motion model's *change* in state (velocity). 
        # In a real EKF, Z would be the measured pose (x, y, yaw) from a sensor like GPS/VO.
        
        # For this example, we simply use the twist from the prediction node as the "measurement"
        # and rely on the R matrix to filter the noise.
        # This is essentially a Kalman Filter structure applied to the prediction model's velocity.
        
        # If your measurement provided P(x, y, theta), we would use those values here.
        # Since your measurement node provides Twist (v, omega), we'll adapt the update:
        
        # Measurement vector (Z) is the current state X for simplicity, 
        # allowing the R matrix to enforce the belief.
        Z = self.ekf.x # [x, y, theta]
        
        # Measurement Function (h(x)): Maps state to measurement space (H is Jacobian)
        def Hx(x):
            # If Z = [x, y, theta], then h(x) is just the state vector itself
            return x

        # Jacobian of Measurement (H)
        H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        # Perform Update
        self.ekf.update(Z=Z, HJacobian=H, Hx=Hx, R=self.ekf.R)
        
        # --- 3. PUBLISHING THE FUSED RESULT ---
        self.publish_odometry(current_time)


    def publish_odometry(self, stamp):
        """
        Publishes the current EKF state as an Odometry message and a TF transform.
        """
        x, y, theta = self.ekf.x[0, 0], self.ekf.x[1, 0], self.ekf.x[2, 0]
        odom_quat = quaternion_from_euler(0, 0, theta)

        # 1. Publish TF Transform (odom -> base_link)
        t = TransformStamped()
        t.header.stamp = stamp.to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = odom_quat[0]
        t.transform.rotation.y = odom_quat[1]
        t.transform.rotation.z = odom_quat[2]
        t.transform.rotation.w = odom_quat[3]
        self.tf_broadcaster.sendTransform(t)

        # 2. Publish Odometry Message
        odom = Odometry()
        odom.header.stamp = stamp.to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        
        # Pose
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = odom_quat[0]
        odom.pose.pose.orientation.y = odom_quat[1]
        odom.pose.pose.orientation.z = odom_quat[2]
        odom.pose.pose.orientation.w = odom_quat[3]

        # Covariance (from EKF P matrix)
        # Note: We only map the 3x3 diagonal elements for (x, y, theta) to the 6x6 Odometry covariance
        P_6x6 = np.zeros((6, 6))
        P_6x6[0:2, 0:2] = self.ekf.P[0:2, 0:2] # x, y
        P_6x6[5, 5] = self.ekf.P[2, 2]         # yaw (index 5 in 6x6)
        odom.pose.covariance = list(P_6x6.flatten())
        
        # --- Path Message Publishing for EKF Fused ---
        pose = PoseStamped()
        pose.header = odom.header # Use the Odometry header
        pose.pose = odom.pose.pose # Use the Odometry pose

        # Append and publish
        self.fused_path.poses.append(pose)

        # Optional: Limit path length
        MAX_PATH_LENGTH = 1000 
        if len(self.fused_path.poses) > MAX_PATH_LENGTH:
            self.fused_path.poses.pop(0)

        self.fused_path_pub.publish(self.fused_path)

        # Velocity (This is assumed to be the prediction velocity, filtered by EKF)
        odom.twist.twist.linear.x = self.prediction_vel[0]
        odom.twist.twist.angular.z = self.prediction_vel[1]

        self.fused_odom_pub.publish(odom)


def main(args=None):
    # Ensure numpy and filterpy are installed: pip install numpy filterpy
    rclpy.init(args=args)
    node = EKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()