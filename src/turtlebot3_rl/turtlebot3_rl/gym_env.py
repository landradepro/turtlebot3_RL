import math
import time

from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

import numpy as np
import rclpy
from rclpy.node import Node

import gymnasium as gym
from gymnasium import spaces

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from rclpy.duration import Duration
from rclpy.parameter import Parameter

LASER_BEAMS = 24
LASER_MAX_RANGE = 3.5
COLLISION_DISTANCE = 0.13
GOAL_DISTANCE = 0.2
MAX_STEPS = 500
CONTROL_PERIOD = 0.2  # seconds of sim time to let each action play out
GOAL_MIN_RADIUS = 0.6
GOAL_MAX_RADIUS = 2.0
DISTANCE_REWARD_SCALE = 20.0
STEP_PENALTY = 0.01
GOAL_REWARD = 100.0
COLLISION_PENALTY = -50.0

LINEAR_MIN = 0.0
LINEAR_MAX = 0.2
ANGULAR_MAX = 1.5

GOAL_MARKER_NAME = "rl_goal_marker"
GOAL_MARKER_SDF = """<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='rl_goal_marker'>
    <static>true</static>
    <link name='link'>
      <visual name='visual'>
        <geometry>
          <cylinder><radius>0.1</radius><length>0.1</length></cylinder>
        </geometry>
        <material>
          <ambient>0.1 0.9 0.1 1</ambient>
          <diffuse>0.1 0.9 0.1 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

class TurtleBot3GazeboEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        if not rclpy.ok():
            rclpy.init()
        self.node = Node( "turtlebot3_rl_env", parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],)

        self.action_space = spaces.Box(low=np.array([LINEAR_MIN, -ANGULAR_MAX], dtype=np.float32), high=np.array([LINEAR_MAX, ANGULAR_MAX], dtype=np.float32), dtype=np.float32,)
        obs_low = np.concatenate([np.zeros(LASER_BEAMS), [0.0, -math.pi]]).astype(np.float32)
        obs_high = np.concatenate(
            [np.full(LASER_BEAMS, LASER_MAX_RANGE), [10.0, math.pi]]
        ).astype(np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._latest_scan = None
        self._latest_odom = None

        self.cmd_vel_pub = self.node.create_publisher(Twist, "/cmd_vel", 10)
        self.node.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self.node.create_subscription(Odometry, "/odom", self._odom_cb, 10)

        self.reset_world_client = self.node.create_client(Empty, "/reset_world")

        self.spawn_client = self.node.create_client(SpawnEntity, "/spawn_entity")
        self.delete_client = self.node.create_client(DeleteEntity, "/delete_entity")
        self._goal_marker_spawned = False
    
        self._goal = np.zeros(2, dtype=np.float32)
        self._prev_distance = None
        self._step_count = 0

        self._wait_for_first_messages()

    def _scan_cb(self, msg):
        self._latest_scan = msg

    def _odom_cb(self, msg):
        self._latest_odom = msg

    def _spin(self, duration):
        target = self.node.get_clock().now() + Duration(seconds=duration)
        while self.node.get_clock().now() < target:
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def _wait_for_first_messages(self, timeout_sec=10.0):
        deadline = time.time() + timeout_sec
        while (self._latest_scan is None or self._latest_odom is None) and time.time() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if self._latest_scan is None or self._latest_odom is None:
            raise TimeoutError(
                "Did not receive /scan and /odom messages. Is Gazebo + TurtleBot3 running?"
            )

    @staticmethod
    def _yaw_from_quaternion(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _downsample_scan(self, scan_msg):
        ranges = np.array(scan_msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=LASER_MAX_RANGE, posinf=LASER_MAX_RANGE, neginf=0.0)
        ranges = np.clip(ranges, 0.0, LASER_MAX_RANGE)
        bins = np.array_split(ranges, LASER_BEAMS)
        return np.array([b.min() if b.size else LASER_MAX_RANGE for b in bins], dtype=np.float32)

    def _get_obs(self):
        scan = self._downsample_scan(self._latest_scan)

        pos = self._latest_odom.pose.pose.position
        yaw = self._yaw_from_quaternion(self._latest_odom.pose.pose.orientation)

        dx = self._goal[0] - pos.x
        dy = self._goal[1] - pos.y
        distance = math.hypot(dx, dy)
        heading_error = self._wrap_angle(math.atan2(dy, dx) - yaw)

        obs = np.concatenate([scan, [distance, heading_error]]).astype(np.float32)
        return obs, distance, scan.min()

    def _sample_goal(self):
        radius = self.np_random.uniform(GOAL_MIN_RADIUS, GOAL_MAX_RADIUS)
        angle = self.np_random.uniform(-math.pi, math.pi)
        return np.array([radius * math.cos(angle), radius * math.sin(angle)], dtype=np.float32)
    
    def _update_goal_marker(self):
     if self._goal_marker_spawned and self.delete_client.wait_for_service(timeout_sec=2.0):
         req = DeleteEntity.Request()
         req.name = GOAL_MARKER_NAME
         future = self.delete_client.call_async(req)
         rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)

     if self.spawn_client.wait_for_service(timeout_sec=2.0):
         req = SpawnEntity.Request()
         req.name = GOAL_MARKER_NAME
         req.xml = GOAL_MARKER_SDF
         req.initial_pose = Pose()
         req.initial_pose.position.x = float(self._goal[0])
         req.initial_pose.position.y = float(self._goal[1])
         req.initial_pose.position.z = 0.05
         future = self.spawn_client.call_async(req)
         rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
         self._goal_marker_spawned = True
     else:
         self.node.get_logger().warn("/spawn_entity service unavailable, skipping goal marker")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.cmd_vel_pub.publish(Twist())

        if self.reset_world_client.wait_for_service(timeout_sec=5.0):
            future = self.reset_world_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
        else:
            self.node.get_logger().warn("/reset_world service unavailable, skipping reset")

        self._goal = self._sample_goal()
        self._update_goal_marker()
        self._step_count = 0
        self._step_count = 0

        # let odometry/scan settle to the post-reset state before reading them
        self._spin(0.3)

        obs, distance, _ = self._get_obs()
        self._prev_distance = distance

        info = {"goal": self._goal.copy()}
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_vel_pub.publish(twist)

        self._spin(CONTROL_PERIOD)

        obs, distance, min_range = self._get_obs()
        self._step_count += 1

        collision = bool(min_range < COLLISION_DISTANCE)
        goal_reached = bool(distance < GOAL_DISTANCE)

        reward = (self._prev_distance - distance) * DISTANCE_REWARD_SCALE - STEP_PENALTY
        terminated = False

        if goal_reached:
            reward += GOAL_REWARD
            terminated = True
        elif collision:
            reward += COLLISION_PENALTY
            terminated = True

        truncated = self._step_count >= MAX_STEPS

        self._prev_distance = distance

        info = {"goal": self._goal.copy(), "collision": collision, "goal_reached": goal_reached}
        return obs, reward, terminated, truncated, info

    def close(self):
        self.cmd_vel_pub.publish(Twist())
        self.node.destroy_node()
