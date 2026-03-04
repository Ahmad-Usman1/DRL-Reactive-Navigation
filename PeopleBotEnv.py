import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from numba import njit
from MapBank import MapBank

# --- HIGH-PERFORMANCE C-COMPILED RAYCASTER ---
@njit(fastmath=True)
def fast_raycast(rx, ry, rth, angles, map_grid, resolution, max_range):
    scan = np.zeros(16, dtype=np.float32)
    h, w = map_grid.shape
    max_px = max_range * resolution
    
    for i in range(16):
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
    PeopleBotEnv - Phase 4 Physics (Hardware Ground-Truth)
    Accurate to ActivMedia Perf PB kinematics, dynamics, and safety thresholds.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        super(PeopleBotEnv, self).__init__()
        
        # --- TRUE HARDWARE SPECS (ActivMedia Perf PB) ---
        self.wheel_radius = 0.0955  
        self.wheel_base = 0.33      
        self.robot_radius = 0.31    
        
        # Kinematic Limits
        self.max_lin_vel = 0.8 #0.9 actual limit, reduced for safety margin
        self.min_lin_vel = -0.3     
        self.max_ang_vel = 2.5 #2.618 actual limit, reduced for safety margin

        self.prev_action = np.zeros(2, dtype=np.float32)
        
        # Max motor RPM based on wheel limits
        self.max_wheel_omega = (self.max_lin_vel + (self.max_ang_vel * self.wheel_base / 2.0)) / self.wheel_radius
        
        # Dynamics Parameters
        self.dt = 0.1
        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0
        self.tau_motor = 0.30 # 300ms mechanical time constant for a heavy chassis
        
        # Safety Thresholds
        self.tipping_threshold = 2.0 # m/s^2 (Centrifugal limit for 1.24m tall CoG)
        
        # Sensor & Nav
        self.max_sensor_range = 5.0
        self.waypoint_radius = 1.0  
        self.goal_radius = 1.0      
        
        # --- MAP BANK INTEGRATION ---
        self.map_bank = MapBank(dataset_dir="training_maps")

        # Spaces
        high_obs = np.array([self.max_sensor_range] * 16 + [60.0, np.pi], dtype=np.float32)
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
        self.current_step = 0
        self.max_steps = 2000  

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.prev_action = np.zeros(2, dtype=np.float32)
        
        # --- TELEMETRY TRACKERS ---
        self.ep_velocity_history = []
        self.ep_min_lidar_history = []
        self.ep_vibration_count = 0
        self.ep_checkpoints_hit = 0
        
        # Fetch fresh map
        self.map_grid, self.waypoints, self.resolution = self.map_bank.get_random_map()
        
        # Total checkpoints is waypoints minus the starting point
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

    def step(self, action):
        self.current_step += 1
        
        action_delta = np.abs(action - self.prev_action)
        self.prev_action = action.copy()

        # 1. Action Scaling (Mapped to true PB limits)
        norm_lin = np.clip(action[0], -1.0, 1.0)
        norm_ang = np.clip(action[1], -1.0, 1.0)
        req_v = norm_lin * self.max_lin_vel if norm_lin >= 0 else abs(norm_lin) * self.min_lin_vel
        req_w = norm_ang * self.max_ang_vel
        
        # 2. Diff Drive Kinematic Limits
        req_wl = (req_v - (req_w * self.wheel_base / 2.0)) / self.wheel_radius
        req_wr = (req_v + (req_w * self.wheel_base / 2.0)) / self.wheel_radius
        max_req_omega = max(abs(req_wl), abs(req_wr))
        if max_req_omega > self.max_wheel_omega:
            scale = self.max_wheel_omega / max_req_omega
            req_wl *= scale
            req_wr *= scale
            
        target_v = (self.wheel_radius / 2.0) * (req_wl + req_wr)
        target_w = (self.wheel_radius / self.wheel_base) * (req_wr - req_wl)
        
        # 3. First-Order Motor Inertia (Emulating mass)
        alpha = self.dt / (self.tau_motor + self.dt)
        self.current_lin_vel += alpha * (target_v - self.current_lin_vel)
        self.current_ang_vel += alpha * (target_w - self.current_ang_vel)
        
        # 4. Kinematics Update
        theta = self.current_pose[2]
        self.current_pose[0] += self.current_lin_vel * np.cos(theta) * self.dt
        self.current_pose[1] += self.current_lin_vel * np.sin(theta) * self.dt
        self.current_pose[2] = self._angdiff(0, theta + self.current_ang_vel * self.dt)
        
        # 5. DYNAMICS CHECK: Centrifugal Tipping
        centrifugal_accel = abs(self.current_lin_vel * self.current_ang_vel)
        is_tipped = centrifugal_accel > self.tipping_threshold
        
        # 6. Goal Logic
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

        # 7. Observation
        obs = self._get_obs()
        scan_data = obs[:16]
        heading_error = obs[17]
        is_collided = self._check_collision()
        min_dist = np.min(scan_data) # Extracted once to use in logging and rewards

        # 8. REWARD ECONOMY
        reward = -0.01 # Standard time tax

        # Deadzone: 0.4 is a massive swing (20% of the total -1 to 1 joystick range).
        # Smooth driving stays below this. Vibration instantly triggers it.
        if action_delta[0] > 0.4:
            reward -= 0.05 * action_delta[0] 
        
        if action_delta[1] > 0.4:
            reward -= 0.05 * action_delta[1]
        
        # Distance Progress
        dist_improvement = self.previous_distance - dist_to_goal
        reward += (0.5 * dist_improvement)
        self.previous_distance = dist_to_goal
        
        # Velocity & Alignment
        reward += (0.1 * math.cos(heading_error)) * self.current_lin_vel
            
        # The 0.40m Dynamic Safety Net
        if min_dist < 0.40:
            danger_level = 0.40 - min_dist
            vel_multiplier = 1.0 + (5.0 * abs(self.current_lin_vel))
            reward -= (vel_multiplier * danger_level) * 0.1
            
        terminated = False
        truncated = False
        
        # Sparse Rewards
        if hit_checkpoint:
            reward += 10.0
            
        if is_collided:
            reward -= 50.0 
            terminated = True
        elif is_tipped: 
            reward -= 50.0
            terminated = True
        elif hit_final_goal:
            reward += 50.0
            terminated = True
            
        if self.current_step >= self.max_steps:
            truncated = True
            
        # --- 9. TELEMETRY LOGGING (The Hard Data) ---
        self.ep_velocity_history.append(self.current_lin_vel)
        self.ep_min_lidar_history.append(min_dist)
        
        if action_delta[0] > 0.4 or action_delta[1] > 0.4:
            self.ep_vibration_count += 1
            
        if hit_checkpoint:
            self.ep_checkpoints_hit += 1

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
                "rate_timeout": 1.0 if truncated and not (hit_final_goal or is_collided or is_tipped) else 0.0
            }
            
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        scan_data = fast_raycast(self.current_pose[0], self.current_pose[1], self.current_pose[2],
                                 self.sensor_angles, self.map_grid, self.resolution, self.max_sensor_range)
        
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