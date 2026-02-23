import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from numba import njit
from MapGenerator import MapGenerator
from MapBank import MapBank

# --- HIGH-PERFORMANCE C-COMPILED RAYCASTER ---
# This bypasses Python's slow loops entirely.
@njit(fastmath=True)
def fast_raycast(rx, ry, rth, angles, map_grid, resolution, max_range):
    scan = np.zeros(16, dtype=np.float32)
    h, w = map_grid.shape
    max_px = max_range * resolution
    
    for i in range(16):
        glob_angle = rth + angles[i]
        # Step size of 2 pixels for speed
        dx = math.cos(glob_angle) * 2.0
        dy = math.sin(glob_angle) * 2.0
        
        curr_x = rx * resolution
        curr_y = ry * resolution
        dist_px = 0.0
        
        while dist_px < max_px:
            dist_px += 2.0
            curr_x += dx
            curr_y += dy
            ix, iy = int(curr_x), int(curr_y)
            
            # Bound & Wall Check
            if ix < 0 or ix >= w or iy < 0 or iy >= h: 
                break
            if map_grid[iy, ix] == 1: 
                break
                
        dist_m = dist_px / resolution
        scan[i] = min(max_range, dist_m)
        
    return scan

class PeopleBotEnv(gym.Env):
    """
    PeopleBotEnv - Phase 2 Physics (Fixed Reward & Speed)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        super(PeopleBotEnv, self).__init__()
        
        # Physical Constants
        self.wheel_radius = 0.0975  
        self.wheel_base = 0.33      
        self.robot_radius = 0.35    
        
        # Constraints
        self.max_lin_vel = 0.8      
        self.min_lin_vel = -0.2     
        self.max_ang_vel = 1.0      
        self.max_wheel_omega = 12.3 
        
        # Inertia
        self.dt = 0.1
        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0
        self.max_lin_accel = 2.0    
        self.max_ang_accel = 3.0    
        
        # Sensor & Nav
        self.max_sensor_range = 5.0
        self.waypoint_radius = 1.0  
        self.goal_radius = 1.0      
        
        # Map Bank
        self.map_bank = MapBank(dataset_dir="training_maps")

        # Spaces
        high_obs = np.array([self.max_sensor_range] * 16 + [20.0, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        front_degs = [90, 50, 30, 10, -10, -30, -50, -90]
        rear_degs = [90, 130, 150, 170, -170, -150, -130, -90]
        self.sensor_angles = np.deg2rad(front_degs + rear_degs).astype(np.float32)
        
        # State Variables
        self.map_grid = None
        self.resolution = 50
        self.waypoints = []
        self.current_pose = np.zeros(3) 
        self.current_goal_index = 0
        self.current_goal = np.zeros(2)
        self.previous_distance = 0.0
        
        # --- CRITICAL FIX: MAP CACHING ---
        self.episode_count = 0
        self.map_reload_interval = 1 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Only generate a new map every 50 episodes
        # if self.map_grid is None or (self.episode_count % self.map_reload_interval == 0):
        #     self.map_grid, self.waypoints, self.resolution = MapGenerator.generate(20, 20)

        # Select a random map from mapbank
        self.grid_map, self.waypoints, self.resolution = self.map_bank.get_random_map()
            
        self.episode_count += 1
            
        start_pt = self.waypoints[0]
        if len(self.waypoints) > 1:
            self.current_goal_index = 1
            self.current_goal = np.array(self.waypoints[1])
            start_theta = math.atan2(self.current_goal[1] - start_pt[1], 
                                     self.current_goal[0] - start_pt[0])
        else:
            self.current_goal_index = 0
            self.current_goal = np.array(start_pt)
            start_theta = 0.0
            
        self.current_pose = np.array([start_pt[0], start_pt[1], start_theta])
        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0
        self.previous_distance = np.linalg.norm(self.current_pose[:2] - self.current_goal)
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. Action Scaling
        norm_lin = np.clip(action[0], -1.0, 1.0)
        norm_ang = np.clip(action[1], -1.0, 1.0)
        req_v = norm_lin * self.max_lin_vel if norm_lin >= 0 else abs(norm_lin) * self.min_lin_vel
        req_w = norm_ang * self.max_ang_vel
        
        # 2. Diff Drive Limits
        req_wl = (req_v - (req_w * self.wheel_base / 2.0)) / self.wheel_radius
        req_wr = (req_v + (req_w * self.wheel_base / 2.0)) / self.wheel_radius
        max_req_omega = max(abs(req_wl), abs(req_wr))
        if max_req_omega > self.max_wheel_omega:
            scale = self.max_wheel_omega / max_req_omega
            req_wl *= scale
            req_wr *= scale
            
        target_v = (self.wheel_radius / 2.0) * (req_wl + req_wr)
        target_w = (self.wheel_radius / self.wheel_base) * (req_wr - req_wl)
        
        # 3. Inertia
        delta_v = np.clip(target_v - self.current_lin_vel, -self.max_lin_accel*self.dt, self.max_lin_accel*self.dt)
        delta_w = np.clip(target_w - self.current_ang_vel, -self.max_ang_accel*self.dt, self.max_ang_accel*self.dt)
        self.current_lin_vel += delta_v
        self.current_ang_vel += delta_w
        
        # 4. Kinematics
        theta = self.current_pose[2]
        self.current_pose[0] += self.current_lin_vel * np.cos(theta) * self.dt
        self.current_pose[1] += self.current_lin_vel * np.sin(theta) * self.dt
        self.current_pose[2] = self._angdiff(0, theta + self.current_ang_vel * self.dt)
        
        # 5. Goal Logic
        dist_to_goal = np.linalg.norm(self.current_pose[:2] - self.current_goal)
        hit_checkpoint, hit_final_goal = False, False
        
        if dist_to_goal < self.waypoint_radius:
            if self.current_goal_index < len(self.waypoints) - 1:
                self.current_goal_index += 1
                self.current_goal = np.array(self.waypoints[self.current_goal_index])
                hit_checkpoint = True
                self.previous_distance = np.linalg.norm(self.current_pose[:2] - self.current_goal)
                dist_to_goal = self.previous_distance
            else:
                if dist_to_goal < self.goal_radius:
                    hit_final_goal = True

        # 6. Observation
        obs = self._get_obs()
        scan_data = obs[:16]
        heading_error = obs[17]
        is_collided = self._check_collision()

        # 7. REWARD ECONOMY (Fixed)
        reward = -0.01 
        
        # Progress
        dist_improvement = self.previous_distance - dist_to_goal
        reward += (0.5 * dist_improvement)
        self.previous_distance = dist_to_goal
        reward += (0.1 * math.cos(heading_error)) * self.current_lin_vel
        
        # --- ANTI-VIBRATION TAX ---
        reward -= (0.01 * abs(self.current_ang_vel)) # Spin penalty
        if self.current_lin_vel < -0.01:
            reward -= 0.05 # Reversing penalty
            
        # Safety Penalty
        min_dist = np.min(scan_data)
        if min_dist < 0.75:
            danger_level = 0.75 - min_dist
            vel_multiplier = 1.0 + (5.0 * abs(self.current_lin_vel))
            reward -= (vel_multiplier * danger_level) * 0.1
            
        terminated = False
        if hit_checkpoint:
            reward += 2.0
        if is_collided:
            reward -= 20.0
            terminated = True
        elif hit_final_goal:
            reward += 10.0
            terminated = True
            
        return obs, reward, terminated, False, {}

    def _get_obs(self):
        # Call the blazing fast Numba function
        scan_data = fast_raycast(self.current_pose[0], self.current_pose[1], self.current_pose[2],
                                 self.sensor_angles, self.map_grid, self.resolution, self.max_sensor_range)
        
        # Add slight realistic noise
        noise = np.random.randn(16) * 0.02
        scan_data = np.clip(scan_data + noise, 0, self.max_sensor_range)
        
        dist_to_goal = np.linalg.norm(self.current_pose[:2] - self.current_goal)
        desired_angle = math.atan2(self.current_goal[1] - self.current_pose[1],
                                   self.current_goal[0] - self.current_pose[0])
        heading_error = self._angdiff(self.current_pose[2], desired_angle)
        
        return np.concatenate([scan_data, [dist_to_goal], [heading_error]]).astype(np.float32)

    def _check_collision(self):
        cx, cy = self.current_pose[0], self.current_pose[1]
        if self._is_occupied(cx, cy): return True
        for ang in np.linspace(0, 2*np.pi, 8, endpoint=False):
            px = cx + self.robot_radius * math.cos(ang)
            py = cy + self.robot_radius * math.sin(ang)
            if self._is_occupied(px, py): return True
        return False

    def _is_occupied(self, x, y):
        ix, iy = int(x * self.resolution), int(y * self.resolution)
        if ix < 0 or ix >= self.map_grid.shape[1] or iy < 0 or iy >= self.map_grid.shape[0]: 
            return True 
        return (self.map_grid[iy, ix] == 1)

    def _angdiff(self, th1, th2):
        return (th2 - th1 + np.pi) % (2 * np.pi) - np.pi