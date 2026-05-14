import rclpy
from rclpy.node import Node
import mujoco
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger

class ObjectPosePublisher(Node):
    def __init__(self, mj_model, mj_data, object_name="object", topic_name="/simulation/object_pose", enable_reset_service=True):
        # Safely initialize rclpy if it hasn't been started in the main script
        if not rclpy.ok():
            rclpy.init()

        node_name = f'mujoco_{object_name}_publisher'
        super().__init__(node_name)

        self.mj_model = mj_model
        self.mj_data = mj_data
        self.object_name = object_name
        self._reset_requested = False

        # ROS 2 Publisher creation uses the Node method
        self.pub = self.create_publisher(PoseStamped, topic_name, 10)

        # Service for reset commands from external processes (only on primary publisher)
        if enable_reset_service:
            self.create_service(Trigger, '/simulation/reset', self._reset_callback)

        # Find the object's body ID in MuJoCo
        try:
            self.body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.object_name)
        except Exception:
            self.get_logger().warn(f"'{self.object_name}' not found in MuJoCo model.")
            self.body_id = -1

    def _reset_callback(self, request, response):
        self._reset_requested = True
        response.success = True
        response.message = "Reset requested"
        return response

    def publish(self):
        # rclpy.ok() replaces rospy.is_shutdown()
        if self.body_id == -1 or not rclpy.ok():
            return
            
        msg = PoseStamped()
        
        # ROS 2 Time API
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        
        # Extract pose arrays
        pos = self.mj_data.xpos[self.body_id]
        quat = self.mj_data.xquat[self.body_id] # MuJoCo: [w, x, y, z]
        
        # Cast to float to avoid numpy float64 serialization issues in ROS 2
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        
        # ROS expects [x, y, z, w]
        msg.pose.orientation.x = float(quat[1])
        msg.pose.orientation.y = float(quat[2])
        msg.pose.orientation.z = float(quat[3])
        msg.pose.orientation.w = float(quat[0])
        
        self.pub.publish(msg)