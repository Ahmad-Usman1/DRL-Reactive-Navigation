import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from MapGenerator import MapGenerator

class PeopleBotEnv(gym.Env):
    """
    PeopleBotEnv - Phase 2 Physics (Differential Drive + Inertia)
    
    Physics Pipeline:
    1. Action -> Requested Velocity
    2. Differential Constraints -> Achievable Velocity (Motor Limits)
    3. Inertia/Mass -> Actual Velocity (Acceleration Limits)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        super(PeopleBotEnv, self).__init__()
        
        # -- Physical Constants (Pioneer 3-DX / PeopleBot) --
        self.wheel_radius = 0.0975  # Meters
        self.wheel_base = 0.33      # Meters
        self.robot_radius = 0.35    # Meters
        
        # -- Constraints --
        self.max_lin_vel = 0.8      # Forward Max
        self.min_lin_vel = -0.2     # Reverse Max
        self.max_ang_vel = 1.0      # Rad/s
        
        # Motor Limit: 12.3 rad/s ~= 1.2 m/s linear speed
        # This provides "headroom" for turning while moving fast.
        self.max_wheel_omega = 12.3 
        
        # -- Inertia (The "Brave" Settings) --
        self.dt = 0.1
        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0
        
        # Updated to match "Snappy" MATLAB Phase 2 settings
        self.max_lin_accel = 2.0    
        self.max_ang_accel = 3.0    
        
        # -- Sensor & Nav --
        self.max_sensor_range = 5.0
        self.waypoint_radius = 1.0  # Hit radius for intermediate waypoints
        self.goal_radius = 1.0      # Hit radius for final goal (Phase 2 Spec)
        
        # -- Define Spaces --
        # Observation: 16 Lidar + Dist + Heading
        high_obs = np.array([self.max_sensor_range] * 16 + [20.0, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)
        
        # Action: Normalized [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # -- Sensor Config --
        front_degs = [90, 50, 30, 10, -10, -30, -50, -90]
        rear_degs = [90, 130, 150, 170, -170, -150, -130, -90]
        self.sensor_angles = np.deg2rad(front_degs + rear_degs)
        
        # -- State Variables --
        self.map_grid = None
        self.resolution = 50
        self.waypoints = []
        self.current_pose = np.zeros(3) 
        
        self.current_goal_index = 0
        self.current_goal = np.zeros(2)
        self.previous_distance = 0.0
        self.episode_count = 0
        self.map_reload_interval = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        
        # Generate new map periodically
        if self.map_grid is None or (self.episode_count % self.map_reload_interval == 0):
            self.map_grid, self.waypoints, self.resolution = MapGenerator.generate(20, 20)
            
        start_pt = self.waypoints[0]
        
        # Select Goal
        if len(self.waypoints) > 1:
            self.current_goal_index = 1
            self.current_goal = np.array(self.waypoints[1])
            # Point towards first goal
            start_theta = math.atan2(self.current_goal[1] - start_pt[1], 
                                     self.current_goal[0] - start_pt[0])
        else:
            self.current_goal_index = 0
            self.current_goal = np.array(start_pt)
            start_theta = 0.0
            
        self.current_pose = np.array([start_pt[0], start_pt[1], start_theta])
        
        # Reset Physics
        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0
        self.previous_distance = np.linalg.norm(self.current_pose[:2] - self.current_goal)
        
        return self._get_obs(), {}

    def step(self, action):
        # --- 1. Interpret Action ---
        # Map [-1, 1] -> [-0.2, 0.8] Linear
        norm_lin = np.clip(action[0], -1.0, 1.0)
        norm_ang = np.clip(action[1], -1.0, 1.0)
        
        if norm_lin >= 0:
            req_v = norm_lin * self.max_lin_vel
        else:
            req_v = abs(norm_lin) * self.min_lin_vel # Reversing
            
        req_w = norm_ang * self.max_ang_vel
        
        # --- 2. Differential Drive Constraints (Motor Saturation) ---
        # Calculate required wheel angular velocities (rad/s)
        # w_l = (v - w * L/2) / R
        # w_r = (v + w * L/2) / R
        req_wl = (req_v - (req_w * self.wheel_base / 2.0)) / self.wheel_radius
        req_wr = (req_v + (req_w * self.wheel_base / 2.0)) / self.wheel_radius
        
        # Check against physical motor limits
        max_req_omega = max(abs(req_wl), abs(req_wr))
        
        if max_req_omega > self.max_wheel_omega:
            # Scale down both wheels equally to preserve curvature (trajectory)
            scale_factor = self.max_wheel_omega / max_req_omega
            req_wl *= scale_factor
            req_wr *= scale_factor
            
        # Convert back to Target Robot Velocities (Achievable Command)
        target_v = (self.wheel_radius / 2.0) * (req_wl + req_wr)
        target_w = (self.wheel_radius / self.wheel_base) * (req_wr - req_wl)
        
        # --- 3. Inertia (Acceleration Limits) ---
        # Apply F=ma approximation to reach the Target Velocity
        diff_v = target_v - self.current_lin_vel
        diff_w = target_w - self.current_ang_vel
        
        delta_v = np.clip(diff_v, -self.max_lin_accel*self.dt, self.max_lin_accel*self.dt)
        delta_w = np.clip(diff_w, -self.max_ang_accel*self.dt, self.max_ang_accel*self.dt)
        
        # Update Actual Velocity
        self.current_lin_vel += delta_v
        self.current_ang_vel += delta_w
        
        # --- 4. Kinematics (Pose Update) ---
        theta = self.current_pose[2]
        self.current_pose[0] += self.current_lin_vel * np.cos(theta) * self.dt
        self.current_pose[1] += self.current_lin_vel * np.sin(theta) * self.dt
        self.current_pose[2] = self._angdiff(0, theta + self.current_ang_vel * self.dt)
        
        # --- 5. Goal Logic ---
        dist_to_goal = np.linalg.norm(self.current_pose[:2] - self.current_goal)
        hit_checkpoint = False
        hit_final_goal = False
        
        # Checkpoint Logic
        if dist_to_goal < self.waypoint_radius:
            if self.current_goal_index < len(self.waypoints) - 1:
                self.current_goal_index += 1
                self.current_goal = np.array(self.waypoints[self.current_goal_index])
                hit_checkpoint = True
                
                # Reset distance metric for the new leg
                self.previous_distance = np.linalg.norm(self.current_pose[:2] - self.current_goal)
                dist_to_goal = self.previous_distance
            else:
                # Final Goal Logic (Phase 2 Spec: 1.0m radius)
                if dist_to_goal < self.goal_radius:
                    hit_final_goal = True

        # --- 6. Observation & Collision ---
        obs = self._get_obs()
        scan_data = obs[:16]
        heading_error = obs[17]
        is_collided = self._check_collision()

        # --- 7. Reward Calculation (Deflated Economy) ---
        reward = -0.001 # Time Penalty
        
        # Progress (Salary)
        dist_improvement = self.previous_distance - dist_to_goal
        reward += (0.5 * dist_improvement)
        self.previous_distance = dist_to_goal
        
        # Compass (Guidance)
        reward += (0.05 * np.cos(heading_error))
        
        # Anti-Oscillation (Tiny penalty to stop camera shake)
        reward -= (0.01 * abs(self.current_ang_vel))

        # Reverse Penalty (Stop the vibrating)
        if (self.current_lin_vel) < 0:
            reward -= 0.01  # Heavy tax for using reverse gear

        # Parking Fine (Don't spin in placear from goal)
        if dist_to_goal > 1.5:
            if abs(self.current_lin_vel) < 0.1 and abs(self.current_ang_vel) > 0.5:
                reward -= 0.1
                
        # Safety Bubble (Velocity Scaled Penalty)
        min_dist = np.min(scan_data)
        if min_dist < 0.75:
            danger_level = 0.75 - min_dist
            # Penalize speed when near walls
            vel_multiplier = 1.0 + (5.0 * abs(self.current_lin_vel))
            penalty = (vel_multiplier * danger_level) * 0.1
            reward -= penalty
            
        # Terminal Rewards
        terminated = False
        truncated = False
        
        if hit_checkpoint:
            reward += 2.0
            
        if is_collided:
            reward -= 20.0
            terminated = True
        elif hit_final_goal:
            reward += 10.0
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
        # Basic Raycasting (Assumes MapGenerator provides binary grid)
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
                dist_px += 2 # Skip steps for performance
                curr_x += dx * 2
                curr_y += dy * 2
                ix, iy = int(curr_x), int(curr_y)
                
                if ix < 0 or ix >= w or iy < 0 or iy >= h: break # Out of bounds
                if self.map_grid[iy, ix] == 1: break # Wall
            
            dist_m = dist_px / self.resolution
            noise = 0.05 * np.random.randn()
            scan_data[i] = max(0.0, min(self.max_sensor_range, dist_m + noise))
            
        return scan_data

    def _check_collision(self):
        cx, cy = self.current_pose[0], self.current_pose[1]
        # Check center
        if self._is_occupied(cx, cy): return True
        # Check periphery (simple circle collision)
        for ang in np.linspace(0, 2*np.pi, 8, endpoint=False):
            px = cx + self.robot_radius * math.cos(ang)
            py = cy + self.robot_radius * math.sin(ang)
            if self._is_occupied(px, py): return True
        return False

    def _is_occupied(self, x, y):
        ix, iy = int(x * self.resolution), int(y * self.resolution)
        if ix < 0 or ix >= self.map_grid.shape[1] or iy < 0 or iy >= self.map_grid.shape[0]: 
            return True # Bound check
        return (self.map_grid[iy, ix] == 1)

    def _angdiff(self, th1, th2):
        # Returns smallest angle difference
        return (th2 - th1 + np.pi) % (2 * np.pi) - np.pi