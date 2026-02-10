import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from MapGenerator import MapGenerator

class PeopleBotEnv(gym.Env):
    """
    PeopleBotEnv - Differential Drive Edition
    
    Observation:
        [0-15]: Lidar distances (Meters)
        [16]: Distance to Current Goal
        [17]: Heading Error to Goal
        
    Action:
        [0]: Left Wheel Velocity (rad/s)
        [1]: Right Wheel Velocity (rad/s)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        super(PeopleBotEnv, self).__init__()
        
        # -- Physical Constants (Pioneer 3-DX / PeopleBot) --
        self.wheel_radius = 0.0975  # 195mm diameter
        self.wheel_base = 0.33      # 330mm width
        self.robot_radius = 0.35
        
        # Motor Limits
        # Max speed 1.2 m/s -> 1.2 / 0.0975 ~= 12.3 rad/s
        self.max_wheel_vel = 12.3 
        self.min_wheel_vel = -12.3
        
        # Inertia (Wheel Acceleration Limits)
        # Rad/s^2. Let's assume it takes ~0.5s to reach max speed.
        self.max_wheel_accel = 20.0 
        
        # Sensor & Nav
        self.max_sensor_range = 5.0
        self.waypoint_radius = 1.0  # NEW: 1 Meter detection radius
        self.dt = 0.1
        
        # -- Define Spaces --
        # Observation: [16 Lidar, 1 Dist, 1 Heading]
        high_obs = np.array([self.max_sensor_range] * 16 + [20.0, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)
        
        # Action: [Left_Vel, Right_Vel]
        self.action_space = spaces.Box(
            low=self.min_wheel_vel, 
            high=self.max_wheel_vel, 
            shape=(2,), 
            dtype=np.float32
        )
        
        # -- Sensor Config --
        front_degs = [90, 50, 30, 10, -10, -30, -50, -90]
        rear_degs = [90, 130, 150, 170, -170, -150, -130, -90]
        self.sensor_angles = np.deg2rad(front_degs + rear_degs)
        
        # -- State Variables --
        self.map_grid = None
        self.resolution = 50
        self.waypoints = []
        self.current_pose = np.zeros(3) # x, y, theta
        
        # Wheel States (Actual current speed of wheels)
        self.left_wheel_vel = 0.0
        self.right_wheel_vel = 0.0
        
        self.current_goal_index = 0
        self.current_goal = np.zeros(2)
        self.previous_distance = 0.0
        self.episode_count = 0
        self.map_reload_interval = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        
        if self.map_grid is None or (self.episode_count % self.map_reload_interval == 0):
            self.map_grid, self.waypoints, self.resolution = MapGenerator.generate(20, 20)
            
        # Init State
        start_pt = self.waypoints[0]
        
        if len(self.waypoints) > 1:
            second_pt = self.waypoints[1]
            start_theta = math.atan2(second_pt[1] - start_pt[1], second_pt[0] - start_pt[0])
            self.current_goal_index = 1
            self.current_goal = self.waypoints[1]
        else:
            start_theta = 0.0
            self.current_goal_index = 0
            self.current_goal = start_pt
            
        self.current_pose = np.array([start_pt[0], start_pt[1], start_theta])
        
        # Reset Wheel Speeds
        self.left_wheel_vel = 0.0
        self.right_wheel_vel = 0.0
        
        self.previous_distance = np.linalg.norm(self.current_pose[:2] - self.current_goal)
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. Parse Action (Target Wheel Velocities)
        target_vl = np.clip(action[0], self.min_wheel_vel, self.max_wheel_vel)
        target_vr = np.clip(action[1], self.min_wheel_vel, self.max_wheel_vel)
        
        # 2. Apply Inertia (Wheel Acceleration)
        # Smoothly ramp current wheel speed to target
        max_delta = self.max_wheel_accel * self.dt
        
        self.left_wheel_vel += np.clip(target_vl - self.left_wheel_vel, -max_delta, max_delta)
        self.right_wheel_vel += np.clip(target_vr - self.right_wheel_vel, -max_delta, max_delta)
        
        # 3. Differential Drive Kinematics
        # Convert Wheel Speeds (rad/s) to Robot Body Velocity (m/s, rad/s)
        # v = r/2 * (vr + vl)
        # w = r/L * (vr - vl)
        
        v = (self.wheel_radius / 2.0) * (self.right_wheel_vel + self.left_wheel_vel)
        w = (self.wheel_radius / self.wheel_base) * (self.right_wheel_vel - self.left_wheel_vel)
        
        # 4. Integrate Position (Runge-Kutta 2nd order is better, but Euler is fine for 10Hz)
        theta = self.current_pose[2]
        self.current_pose[0] += v * np.cos(theta) * self.dt
        self.current_pose[1] += v * np.sin(theta) * self.dt
        self.current_pose[2] = self._angdiff(0, theta + w * self.dt)
        
        # 5. Goal Logic (Updated for 1.0m Radius)
        dist_to_goal = np.linalg.norm(self.current_pose[:2] - self.current_goal)
        hit_checkpoint = False
        hit_final_goal = False
        
        # NEW: 1.0 Meter Acceptance Radius
        if dist_to_goal < self.waypoint_radius:
            if self.current_goal_index < len(self.waypoints) - 1:
                self.current_goal_index += 1
                self.current_goal = self.waypoints[self.current_goal_index]
                hit_checkpoint = True
                
                # Reset distance metric
                self.previous_distance = np.linalg.norm(self.current_pose[:2] - self.current_goal)
                dist_to_goal = self.previous_distance
            else:
                # For the Final Goal, maybe require getting closer? (Optional)
                # Let's stick to 1.0m for consistency, or 0.5m for precision.
                if dist_to_goal < 0.5: 
                    hit_final_goal = True
        
        # 6. Observations & Collision
        obs = self._get_obs()
        scan_data = obs[:16]
        heading_error = obs[17]
        
        is_collided = self._check_collision()
        
        # 7. Reward Function
        reward = -0.01 # Slightly higher step penalty
        
        # Progress
        dist_improvement = self.previous_distance - dist_to_goal
        reward += (1.0 * dist_improvement) # Increased weight
        self.previous_distance = dist_to_goal
        
        # Orientation (Help it face the goal)
        reward += (0.1 * np.cos(heading_error))
        
        # Smoothness (Penalize jerky wheel changes)
        # Note: This might slow down learning, use sparingly
        # reward -= 0.01 * abs(target_vl - target_vr) 
        
        # Safety
        min_dist = np.min(scan_data)
        if min_dist < 0.6:
            reward -= (0.6 - min_dist) * 2.0
            
        terminated = False
        truncated = False
        
        if hit_checkpoint:
            reward += 5.0
            
        if is_collided:
            reward -= 20.0
            terminated = True
        elif hit_final_goal:
            reward += 20.0
            terminated = True
            
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        scan_data = self._get_sensor_readings()
        dist_to_goal = np.linalg.norm(self.current_pose[:2] - self.current_goal)
        desired_angle = math.atan2(self.current_goal[1] - self.current_pose[1],
                                   self.current_goal[0] - self.current_pose[0])
        heading_error = self._angdiff(self.current_pose[2], desired_angle)
        return np.concatenate([scan_data, [dist_to_goal], [heading_error]]).astype(np.float32)

    def _get_sensor_readings(self):
        scan_data = np.zeros(16, dtype=np.float32)
        start_x, start_y = int(self.current_pose[0] * self.resolution), int(self.current_pose[1] * self.resolution)
        h, w = self.map_grid.shape
        max_px = int(self.max_sensor_range * self.resolution)
        
        for i, angle in enumerate(self.sensor_angles):
            glob_angle = self.current_pose[2] + angle
            dx, dy = math.cos(glob_angle), math.sin(glob_angle)
            curr_x, curr_y = float(start_x), float(start_y)
            dist_px = 0
            
            while dist_px < max_px:
                dist_px += 2 # Optimization: Step by 2 pixels for speed
                curr_x += dx * 2
                curr_y += dy * 2
                ix, iy = int(curr_x), int(curr_y)
                
                if ix < 0 or ix >= w or iy < 0 or iy >= h: break
                if self.map_grid[iy, ix] == 1: break
            
            dist_m = dist_px / self.resolution
            noise = 0.05 * np.random.randn()
            scan_data[i] = max(0.0, min(self.max_sensor_range, dist_m + noise))
            
        return scan_data

    def _check_collision(self):
        cx, cy = self.current_pose[0], self.current_pose[1]
        # Quick center check
        if self._is_occupied(cx, cy): return True
        # 8-point perimeter check
        for ang in np.linspace(0, 2*np.pi, 8, endpoint=False):
            if self._is_occupied(cx + self.robot_radius*math.cos(ang), 
                                 cy + self.robot_radius*math.sin(ang)):
                return True
        return False

    def _is_occupied(self, x, y):
        ix, iy = int(x * self.resolution), int(y * self.resolution)
        if ix < 0 or ix >= self.map_grid.shape[1] or iy < 0 or iy >= self.map_grid.shape[0]: return True
        return (self.map_grid[iy, ix] == 1)

    def _angdiff(self, th1, th2):
        return (th2 - th1 + np.pi) % (2 * np.pi) - np.pi