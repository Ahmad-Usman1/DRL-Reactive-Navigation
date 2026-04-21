"""
PeopleBotEnv_V3.py
==================
BEANS V3 — parallel dual-sensor delay model.

CHANGES FROM ORIGINAL
─────────────────────
max_steps: 2500 → 3500
  CornerGauntlet maps (3 turns, tightened excursion) have worst-case path ~55m.
  At 0.028 m/step travel alone requires ~1960 steps. 3500 provides 1540 steps
  of overhead for corner braking, grace periods, and heading corrections.
  The timeout penalty (−150) retains its meaning — 3500 is not so loose that
  dawdling becomes free.

Heading penalty: grace period added (15 steps after each checkpoint)
  Without this, the ±π heading jump when current_goal advances fires a full
  misalignment penalty (up to −3.0) in the very next step. The policy learns
  to slow before every waypoint to minimise this spike — a slow-near-waypoints
  reflex that transfers into all finetuned models and compounds timeouts.
  The finetuning env already has this grace period; adding it here ensures
  future base-trained models do not develop the reflex at all.
  steps_since_checkpoint is tracked with two added lines in reset() and step().

All physics, FIFOs, obs space: UNCHANGED. Existing V3 weights remain compatible.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from numba import njit
from MapBank import MapBank


@njit(fastmath=True)
def fast_raycast(rx, ry, rth, angles, map_grid, resolution, max_range):
    num_rays = angles.shape[0]
    scan     = np.zeros(num_rays, dtype=np.float32)
    h, w     = map_grid.shape
    max_px   = max_range * resolution
    for i in range(num_rays):
        glob_angle = rth + angles[i]
        dx = math.cos(glob_angle) * 2.0
        dy = math.sin(glob_angle) * 2.0
        curr_x = rx * resolution;  curr_y = ry * resolution;  dist_px = 0.0
        while dist_px < max_px:
            dist_px += 2.0;  curr_x += dx;  curr_y += dy
            ix = int(curr_x);  iy = int(curr_y)
            if ix < 0 or ix >= w or iy < 0 or iy >= h: break
            if map_grid[iy, ix] == 1: break
        scan[i] = min(max_range, dist_px / resolution)
    return scan


class PeopleBotEnv(gym.Env):
    """PeopleBotEnv V3 — parallel dual-sensor delay + velocity derivative obs."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    SONAR_DELAY_STEPS  = 1
    CAMERA_DELAY_STEPS = 2
    ACTUATOR_LAG_STEPS = 1

    def __init__(self):
        super().__init__()

        self.wheel_radius  = 0.0955
        self.wheel_base    = 0.33
        self.robot_radius  = 0.31
        self.max_lin_vel   = 0.4
        self.min_lin_vel   = 0.0
        self.max_ang_vel   = 1.9
        self.dt            = 0.1
        self.lag_steps     = self.ACTUATOR_LAG_STEPS
        self.action_history = np.zeros((self.lag_steps, 2), dtype=np.float32)
        self.tau_motor     = 0.20
        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0
        self.max_wheel_omega = (
            self.max_lin_vel + self.max_ang_vel * self.wheel_base / 2.0
        ) / self.wheel_radius
        self.tipping_threshold = 2.0
        self.max_sensor_range  = 3.0

        front_degs = [90, 50, 30, 25, 20, 15, 10, 5, 0,
                      -5, -10, -15, -20, -25, -30, -50, -90]
        self.sensor_angles = np.deg2rad(front_degs).astype(np.float32)
        self.num_rays      = len(self.sensor_angles)

        self.sonar_indices  = np.array([0, 1, 2, 6, 10, 14, 15, 16], dtype=np.int64)
        self.camera_indices = np.array([3, 4, 5, 7, 8, 9, 11, 12, 13], dtype=np.int64)
        self.front_indices  = np.where(np.abs(self.sensor_angles) <= np.deg2rad(30))[0]
        self.side_indices   = np.where(np.abs(self.sensor_angles) >  np.deg2rad(30))[0]

        self._init_sensor_fifos()

        self.waypoint_radius = 1.5
        self.goal_radius     = 1.0
        self.max_steps       = 3500   # raised from 2500

        obs_size = self.num_rays + 4 + 2 + (self.lag_steps * 2)
        assert obs_size == 25
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(2,), dtype=np.float32)

        self.map_bank   = MapBank(dataset_dir="training_maps")
        self.difficulty = 0.0

        self.map_grid           = None
        self.resolution         = 50
        self.waypoints          = []
        self.current_pose       = np.zeros(3)
        self.current_goal_index = 0
        self.current_goal       = np.zeros(2)
        self.previous_distance  = 0.0
        self.current_step       = 0
        self.steps_since_checkpoint = 0   # NEW: grace period tracker

    # ── Sensor FIFOs ─────────────────────────────────────────────────────────

    def _init_sensor_fifos(self):
        sonar_row = np.concatenate([
            np.full(self.num_rays, self.max_sensor_range, dtype=np.float32),
            np.zeros(2, dtype=np.float32)
        ])
        self.sonar_fifo  = np.tile(sonar_row, (self.SONAR_DELAY_STEPS, 1))
        self.camera_fifo = np.full((self.CAMERA_DELAY_STEPS, self.num_rays),
                                   self.max_sensor_range, dtype=np.float32)

    def _push_sensor_data(self, fresh_scan, lin_vel, ang_vel):
        new_row = np.concatenate([fresh_scan, [lin_vel, ang_vel]])
        self.sonar_fifo[:-1]  = self.sonar_fifo[1:];   self.sonar_fifo[-1]  = new_row
        self.camera_fifo[:-1] = self.camera_fifo[1:];  self.camera_fifo[-1] = fresh_scan

    def _get_delayed_obs_components(self):
        sonar_row  = self.sonar_fifo[0];  camera_row = self.camera_fifo[0]
        merged = np.empty(self.num_rays, dtype=np.float32)
        merged[self.camera_indices] = camera_row[self.camera_indices]
        merged[self.sonar_indices]  = sonar_row[self.sonar_indices]
        return merged, float(sonar_row[-2]), float(sonar_row[-1])

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.action_history = np.zeros((self.lag_steps, 2), dtype=np.float32)
        self._init_sensor_fifos()

        self.ep_velocity_history  = []
        self.ep_min_lidar_history = []
        self.ep_vibration_count   = 0
        self.ep_checkpoints_hit   = 0

        self.map_grid, self.waypoints, self.resolution = self.map_bank.get_random_map()
        self.total_checkpoints = max(1, len(self.waypoints) - 1)

        start_pt = self.waypoints[0]
        if len(self.waypoints) > 1:
            self.current_goal_index = 1
            self.current_goal = np.array(self.waypoints[1], dtype=np.float32)
            start_theta = math.atan2(self.current_goal[1] - start_pt[1],
                                     self.current_goal[0] - start_pt[0])
        else:
            self.current_goal_index = 0
            self.current_goal = np.array(start_pt, dtype=np.float32)
            start_theta = 0.0

        self.current_pose      = np.array([start_pt[0], start_pt[1], start_theta])
        self.current_lin_vel   = 0.0
        self.current_ang_vel   = 0.0
        self.previous_distance = float(np.linalg.norm(
            self.current_pose[:2] - self.current_goal))
        self.current_step           = 0
        self.steps_since_checkpoint = 0   # NEW

        return self._get_obs(), {}

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        self.current_step += 1

        action_delta = np.abs(action - self.action_history[-1])
        if action_delta[0] > 0.4 or action_delta[1] > 0.4:
            self.ep_vibration_count += 1

        delayed_action           = self.action_history[0].copy()
        self.action_history[:-1] = self.action_history[1:]
        self.action_history[-1]  = action.copy()

        norm_lin = float(np.clip(delayed_action[0], -1.0, 1.0))
        norm_ang = float(np.clip(delayed_action[1], -1.0, 1.0))
        req_v    = norm_lin * self.max_lin_vel if norm_lin >= 0 else 0.0
        req_w    = norm_ang * self.max_ang_vel

        req_wl = (req_v - req_w * self.wheel_base / 2.0) / self.wheel_radius
        req_wr = (req_v + req_w * self.wheel_base / 2.0) / self.wheel_radius
        max_req = max(abs(req_wl), abs(req_wr))
        if max_req > self.max_wheel_omega:
            s = self.max_wheel_omega / max_req;  req_wl *= s;  req_wr *= s

        target_v = (self.wheel_radius / 2.0)            * (req_wl + req_wr)
        target_w = (self.wheel_radius / self.wheel_base) * (req_wr - req_wl)

        alpha = self.dt / (self.tau_motor + self.dt)
        self.current_lin_vel += alpha * (target_v - self.current_lin_vel)
        self.current_ang_vel += alpha * (target_w - self.current_ang_vel)

        theta = self.current_pose[2]
        self.current_pose[0] += self.current_lin_vel * math.cos(theta) * self.dt
        self.current_pose[1] += self.current_lin_vel * math.sin(theta) * self.dt
        self.current_pose[2]  = self._angdiff(0, theta + self.current_ang_vel * self.dt)

        is_tipped   = abs(self.current_lin_vel * self.current_ang_vel) > self.tipping_threshold
        is_collided = self._check_collision()

        dist_to_goal   = float(np.linalg.norm(self.current_pose[:2] - self.current_goal))
        hit_checkpoint = False;  hit_final_goal = False

        if dist_to_goal < self.waypoint_radius:
            if self.current_goal_index < len(self.waypoints) - 1:
                self.current_goal_index += 1
                self.current_goal = np.array(
                    self.waypoints[self.current_goal_index], dtype=np.float32)
                hit_checkpoint     = True
                self.previous_distance = float(
                    np.linalg.norm(self.current_pose[:2] - self.current_goal))
                dist_to_goal = self.previous_distance
            else:
                if dist_to_goal < self.goal_radius:
                    hit_final_goal = True

        # NEW: update grace period counter
        if hit_checkpoint:
            self.steps_since_checkpoint = 0
        else:
            self.steps_since_checkpoint += 1

        fresh_scan = fast_raycast(
            self.current_pose[0], self.current_pose[1], self.current_pose[2],
            self.sensor_angles, self.map_grid, self.resolution, self.max_sensor_range)
        fresh_scan = np.clip(
            fresh_scan + np.random.randn(self.num_rays).astype(np.float32) * 0.02,
            0.0, self.max_sensor_range)

        self._push_sensor_data(fresh_scan, self.current_lin_vel, self.current_ang_vel)
        delayed_scan, delayed_lin_v, delayed_ang_v = self._get_delayed_obs_components()

        desired_angle = math.atan2(self.current_goal[1] - self.current_pose[1],
                                   self.current_goal[0] - self.current_pose[0])
        heading_error = self._angdiff(self.current_pose[2], desired_angle)

        # ── Reward ────────────────────────────────────────────────────────────
        terminated = False;  truncated = False
        reward     = -0.05

        front_dist = float(np.min(delayed_scan[self.front_indices]))
        side_dist  = float(np.min(delayed_scan[self.side_indices]))

        dist_improvement    = self.previous_distance - dist_to_goal
        reward             += 10.0 * dist_improvement
        self.previous_distance = dist_to_goal

        SPEED_SCALE_FRONT = 3.0;  SPEED_SCALE_SIDE = 2.0;  SIDE_WEIGHT = 0.40
        front_ratio  = float(np.clip(front_dist / SPEED_SCALE_FRONT, 0.0, 1.0))
        side_ratio   = float(np.clip(side_dist  / SPEED_SCALE_SIDE,  0.0, 1.0))
        budget_ratio = front_ratio * (1.0 - SIDE_WEIGHT) + side_ratio * SIDE_WEIGHT

        velocity_excess = max(0.0, self.current_lin_vel - budget_ratio * self.max_lin_vel)
        if velocity_excess > 0.01:
            reward -= 20.0 * (velocity_excess ** 2)

        if self.current_lin_vel > 0.05:
            ttc = front_dist / self.current_lin_vel
            if ttc < 1.5:
                reward -= 4.0 * ((1.5 / ttc - 1.0) ** 2)

        # NEW: grace period on heading penalty — ramps 0→1 over 15 steps after
        # each checkpoint to suppress the ±π jump that fires on the very next step
        grace_factor = min(1.0, self.steps_since_checkpoint / 15.0)
        reward -= 1.5 * (1.0 - math.cos(heading_error)) * budget_ratio * grace_factor

        if hit_checkpoint:
            reward += 10.0 + 25.0 * max(0.0, math.cos(heading_error))
            self.ep_checkpoints_hit += 1

        if is_collided or is_tipped:
            reward -= 500.0;  terminated = True
        elif hit_final_goal:
            reward += 500.0;  terminated = True

        if self.current_step >= self.max_steps and not hit_final_goal:
            reward -= 150.0;  truncated = True

        min_clearance = min(front_dist, side_dist)
        self.ep_velocity_history.append(self.current_lin_vel)
        self.ep_min_lidar_history.append(min_clearance)

        info = {}
        if terminated or truncated:
            info["telemetry"] = {
                "avg_velocity":            float(np.mean(self.ep_velocity_history)),
                "avg_wall_clearance":      float(np.mean(self.ep_min_lidar_history)),
                "vibration_events":        float(self.ep_vibration_count),
                "checkpoint_capture_rate": float(self.ep_checkpoints_hit / self.total_checkpoints),
                "rate_success": 1.0 if hit_final_goal else 0.0,
                "rate_crash":   1.0 if (is_collided or is_tipped) else 0.0,
                "rate_timeout": 1.0 if (truncated and not hit_final_goal) else 0.0,
            }

        return (self._build_obs(delayed_scan, dist_to_goal, heading_error,
                                delayed_lin_v, delayed_ang_v),
                reward, terminated, truncated, info)

    # ── Obs builder ───────────────────────────────────────────────────────────

    def _build_obs(self, delayed_scan, dist_to_goal, heading_error,
                   delayed_lin_v, delayed_ang_v):
        obs = np.concatenate([
            delayed_scan / self.max_sensor_range,
            [min(dist_to_goal / 10.0, 1.0)],
            [heading_error / np.pi],
            [self.current_lin_vel / self.max_lin_vel],
            [self.current_ang_vel / self.max_ang_vel],
            [delayed_lin_v / self.max_lin_vel],
            [delayed_ang_v / self.max_ang_vel],
            self.action_history.flatten(),
        ]).astype(np.float32)
        assert obs.shape == (25,), f"Obs shape error: {obs.shape}"
        return obs

    def _get_obs(self):
        delayed_scan, delayed_lin_v, delayed_ang_v = self._get_delayed_obs_components()
        dist_to_goal  = float(np.linalg.norm(self.current_pose[:2] - self.current_goal))
        desired_angle = math.atan2(self.current_goal[1] - self.current_pose[1],
                                   self.current_goal[0] - self.current_pose[0])
        heading_error = self._angdiff(self.current_pose[2], desired_angle)
        return self._build_obs(delayed_scan, dist_to_goal, heading_error,
                               delayed_lin_v, delayed_ang_v)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def set_difficulty(self, difficulty_level):
        self.difficulty = float(np.clip(difficulty_level, 0.0, 1.0))
        self.map_bank.set_difficulty(self.difficulty)

    def _check_collision(self):
        cx, cy = self.current_pose[0], self.current_pose[1]
        if self._is_occupied(cx, cy): return True
        for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            if self._is_occupied(cx + self.robot_radius * math.cos(ang),
                                 cy + self.robot_radius * math.sin(ang)):
                return True
        return False

    def _is_occupied(self, x, y):
        ix = int(x * self.resolution);  iy = int(y * self.resolution)
        if ix < 0 or ix >= self.map_grid.shape[1]: return True
        if iy < 0 or iy >= self.map_grid.shape[0]: return True
        return self.map_grid[iy, ix] == 1

    def _angdiff(self, th1, th2):
        return (th2 - th1 + np.pi) % (2 * np.pi) - np.pi