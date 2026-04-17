import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from numba import njit
from MapBank import MapBank

# --- HIGH-PERFORMANCE C-COMPILED RAYCASTER ---
@njit(fastmath=True)
def fast_raycast(rx, ry, rth, angles, map_grid, resolution, max_range):
    num_rays = angles.shape[0] 
    scan = np.zeros(num_rays, dtype=np.float32)
    h, w = map_grid.shape
    max_px = max_range * resolution
    
    for i in range(num_rays):
        glob_angle = rth + angles[i]
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
            
            if ix < 0 or ix >= w or iy < 0 or iy >= h: 
                break
            if map_grid[iy, ix] == 1: 
                break
                
        dist_m = dist_px / resolution
        scan[i] = min(max_range, dist_m)
        
    return scan

class PeopleBotEnv(gym.Env):
    """
    PeopleBotEnv - Phase 6 (Dead-Time Simulation & Virtual Sonars)
    Simulates a 300ms hardware delay and integrates ESP32-Cam FOV density.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        super(PeopleBotEnv, self).__init__()
        
        # --- TRUE HARDWARE SPECS ---
        self.wheel_radius = 0.0955  
        self.wheel_base = 0.33      
        self.robot_radius = 0.31    
        
        # Kinematic Limits
        self.max_lin_vel = 0.4 
        # BRUTAL FIX: If rear sensors are removed, reverse MUST be disabled.
        self.min_lin_vel = 0.0     
        self.max_ang_vel = 2.0 

        # --- DELAY SIMULATION (ACTION STACKING) ---
        self.dt = 0.1
        self.lag_steps = 3 # 300ms delay at 10Hz
        # FIFO queue to hold the actions "in the pipe"
        self.action_history = np.zeros((self.lag_steps, 2), dtype=np.float32)
        
        self.map_bank = MapBank(dataset_dir="training_maps")
        self.difficulty = 0.0

        self.max_wheel_omega = (self.max_lin_vel + (self.max_ang_vel * self.wheel_base / 2.0)) / self.wheel_radius
        
        # Dynamics Parameters
        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0
        self.tau_motor = 0.30 
        
        # Safety Thresholds
        self.tipping_threshold = 2.0 
        
        # Sensor & Nav
        self.max_sensor_range = 5.0
        self.waypoint_radius = 2.5  
        self.goal_radius = 1.5      
        
        # --- THE ESP32-CAM DENSITY UPGRADE ---
        # Physical Sonars + ESP32 Virtual Bins tightly packed in the +/- 30 deg FOV
        front_degs = [90, 50, 30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30, -50, -90]
        self.sensor_angles = np.deg2rad(front_degs).astype(np.float32)
        self.num_rays = len(self.sensor_angles)

        # Dynamic Lidar Indexing for Anisotropic Safety (Camera FOV is the front zone)
        self.front_indices = np.where(np.abs(self.sensor_angles) <= np.deg2rad(30))[0]
        self.side_indices = np.where(np.abs(self.sensor_angles) > np.deg2rad(30))[0]

        # --- NORMALIZED SPACES ---
        # 17 Rays + 1 Dist + 1 Head + 1 Lin + 1 Ang + 6 History (3 steps * 2 actions) = 27 Inputs
        obs_size = self.num_rays + 4 + (self.lag_steps * 2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # State Variables
        self.map_grid = None
        self.resolution = 50
        self.waypoints = []
        self.current_pose = np.zeros(3) 
        self.current_goal_index = 0
        self.current_goal = np.zeros(2)
        self.previous_distance = 0.0
        self.current_step = 0
        self.max_steps = 3000  

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Flush the hardware pipe on reset
        self.action_history = np.zeros((self.lag_steps, 2), dtype=np.float32)
        
        # Telemetry
        self.ep_velocity_history = []
        self.ep_min_lidar_history = []
        self.ep_vibration_count = 0
        self.ep_checkpoints_hit = 0
        
        self.map_grid, self.waypoints, self.resolution = self.map_bank.get_random_map()
        self.total_checkpoints = max(1, len(self.waypoints) - 1)
            
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
        self.current_step = 0
        
        return self._get_obs(), {}

    def set_difficulty(self, difficulty_level):
        self.difficulty = np.clip(difficulty_level, 0.0, 1.0)
        self.map_bank.set_difficulty(self.difficulty)

    def step(self, action):
        self.current_step += 1
        
        # Track vibration using the requested action, not the delayed one
        action_delta = np.abs(action - self.action_history[-1])
        if action_delta[0] > 0.4 or action_delta[1] > 0.4:
            self.ep_vibration_count += 1

        # --- THE 300MS DEAD-TIME SIMULATION ---
        # 1. Pull the action from 300ms ago to execute NOW
        delayed_action = self.action_history[0].copy()
        
        # 2. Shift the pipe forward
        self.action_history[:-1] = self.action_history[1:]
        
        # 3. Insert the new requested action at the end of the pipe
        self.action_history[-1] = action.copy()

        # 4. Action Scaling (Using the DELAYED action for physics)
        norm_lin = np.clip(delayed_action[0], -1.0, 1.0)
        norm_ang = np.clip(delayed_action[1], -1.0, 1.0)
        
        # Min velocity is now 0.0, so bot only goes forward
        req_v = norm_lin * self.max_lin_vel if norm_lin >= 0 else 0.0 
        req_w = norm_ang * self.max_ang_vel
        
        # 5. Diff Drive Kinematic Limits
        req_wl = (req_v - (req_w * self.wheel_base / 2.0)) / self.wheel_radius
        req_wr = (req_v + (req_w * self.wheel_base / 2.0)) / self.wheel_radius
        max_req_omega = max(abs(req_wl), abs(req_wr))
        if max_req_omega > self.max_wheel_omega:
            scale = self.max_wheel_omega / max_req_omega
            req_wl *= scale
            req_wr *= scale
            
        target_v = (self.wheel_radius / 2.0) * (req_wl + req_wr)
        target_w = (self.wheel_radius / self.wheel_base) * (req_wr - req_wl)
        
        # 6. Motor Dynamics
        alpha = self.dt / (self.tau_motor + self.dt)
        self.current_lin_vel += alpha * (target_v - self.current_lin_vel)
        self.current_ang_vel += alpha * (target_w - self.current_ang_vel)
        
        # 7. Kinematics
        theta = self.current_pose[2]
        self.current_pose[0] += self.current_lin_vel * np.cos(theta) * self.dt
        self.current_pose[1] += self.current_lin_vel * np.sin(theta) * self.dt
        self.current_pose[2] = self._angdiff(0, theta + self.current_ang_vel * self.dt)
        
        # 8. Physics States
        centrifugal_accel = abs(self.current_lin_vel * self.current_ang_vel)
        is_tipped = centrifugal_accel > self.tipping_threshold
        is_collided = self._check_collision()
        
        # 9. Goal Logic
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

        # 10. Sensor Raycasting
        scan_data = fast_raycast(self.current_pose[0], self.current_pose[1], self.current_pose[2],
                                 self.sensor_angles, self.map_grid, self.resolution, self.max_sensor_range)
        noise = np.random.randn(self.num_rays) * 0.02 
        scan_data = np.clip(scan_data + noise, 0, self.max_sensor_range)
        
        desired_angle = math.atan2(self.current_goal[1] - self.current_pose[1],
                                   self.current_goal[0] - self.current_pose[0])
        heading_error = self._angdiff(self.current_pose[2], desired_angle)

        # 11. Reward Architecture
        terminated = False
        truncated = False
        reward = -0.1 
        
        # Danger Zones
        front_dist = np.min(scan_data[self.front_indices])
        side_dist = np.min(scan_data[self.side_indices])
        min_dist = np.min([front_dist,side_dist])

        # Progress & Heading
        dist_improvement = self.previous_distance - dist_to_goal
        reward += (5.0 * dist_improvement) 
        self.previous_distance = dist_to_goal
        if min_dist > 0.4:
            reward += (2.0 * math.cos(heading_error)) * max(0, self.current_lin_vel)
        if (min_dist < 0.4 and self.current_lin_vel > 0.2):
            reward -= 2 * (self.current_lin_vel/min_dist)
        
        
        if front_dist < 0.8:
            danger_front = (0.8 - front_dist) ** 2
            reward -= danger_front * (2.0 + 10.0 * max(0, self.current_lin_vel))
            
        if side_dist < 0.4:
            danger_side = (0.4 - side_dist) ** 2
            reward -= danger_side * 6 

        if hit_checkpoint:
            reward += 10.0
            self.ep_checkpoints_hit += 1

        # Terminal States
        if is_collided or is_tipped:
            reward = -150.0 
            terminated = True
        elif hit_final_goal:
            reward = 200.0
            terminated = True
            
        if self.current_step >= self.max_steps and not hit_final_goal:
            reward = -150.0 
            truncated = True

        # 12. Telemetry Update
        self.ep_velocity_history.append(self.current_lin_vel)
        self.ep_min_lidar_history.append(min(front_dist, side_dist))

        info = {}
        if terminated or truncated:
            avg_vel = float(np.mean(self.ep_velocity_history)) if self.ep_velocity_history else 0.0
            avg_lidar = float(np.mean(self.ep_min_lidar_history)) if self.ep_min_lidar_history else 0.0
            capture_rate = float(self.ep_checkpoints_hit / self.total_checkpoints)
            
            info["telemetry"] = {
                "avg_velocity": avg_vel,
                "avg_wall_clearance": avg_lidar,
                "vibration_events": float(self.ep_vibration_count),
                "checkpoint_capture_rate": capture_rate,
                "rate_success": 1.0 if hit_final_goal else 0.0,
                "rate_crash": 1.0 if (is_collided or is_tipped) else 0.0,
                "rate_timeout": 1.0 if truncated and not hit_final_goal else 0.0
            }
            
        return self._get_obs(scan_data, dist_to_goal, heading_error), reward, terminated, truncated, info

    def _get_obs(self, scan_data=None, dist_to_goal=None, heading_error=None):
        if scan_data is None:
            scan_data = fast_raycast(self.current_pose[0], self.current_pose[1], self.current_pose[2],
                                     self.sensor_angles, self.map_grid, self.resolution, self.max_sensor_range)
            noise = np.random.randn(self.num_rays) * 0.02 
            scan_data = np.clip(scan_data + noise, 0, self.max_sensor_range)
            
            dist_to_goal = np.linalg.norm(self.current_pose[:2] - self.current_goal)
            desired_angle = math.atan2(self.current_goal[1] - self.current_pose[1],
                                       self.current_goal[0] - self.current_pose[0])
            heading_error = self._angdiff(self.current_pose[2], desired_angle)
            
        norm_scan = scan_data / self.max_sensor_range
        norm_dist = min(dist_to_goal / 10.0, 1.0)
        norm_head = heading_error / np.pi
        norm_v = self.current_lin_vel / self.max_lin_vel
        norm_w = self.current_ang_vel / self.max_ang_vel
        
        # Flatten the action history array so the network sees [a_t-3, a_t-2, a_t-1]
        flat_history = self.action_history.flatten()
        
        return np.concatenate([
            norm_scan, 
            [norm_dist], 
            [norm_head], 
            [norm_v], 
            [norm_w],
            flat_history
        ]).astype(np.float32)

    def _check_collision(self):
        cx, cy = self.current_pose[0], self.current_pose[1]
        if self._is_occupied(cx, cy): return True
        for ang in np.linspace(0, 2*np.pi, 24, endpoint=False):
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