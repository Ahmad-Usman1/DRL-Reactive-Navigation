"""
PeopleBotEnv_V5.py
==================
BEANS Navigation Stack — V5 Final Architecture

ARCHITECTURAL DECISIONS vs V3
──────────────────────────────────────────────────────────────────────────────
Simplified Delay Model:
  V3's dual sonar/camera FIFO added physical correctness but no measurable
  improvement on hard maps vs V1's unified FIFO. V5 uses a clean asymmetric model:
    Obs FIFO     : 1 step  = 100ms  (all sensor data unified, parallel pipelines merged)
    Actuator FIFO: 2 steps = 200ms  (command → ESP32 → P2OS → motor execution)

Domain Randomization per Episode (NEW):
  Linear  inertia τ_v ~ U[0.50, 1.15]s  robot mass × terrain drag variation
    alpha_v range: 0.167 (τ=0.5) to 0.080 (τ=1.15) — slower linear response under load
  Angular inertia τ_w ~ U[0.03, 0.10]s  wheel/motor angular response variation
    alpha_w range: 0.769 (τ=0.03) to 0.500 (τ=0.10) — fast angular dynamics, correct
  Positional slip: σ_pos = K_SLIP_POS × |v| × dt (Gaussian per step, decoupled)
    At max speed: σ_pos = 0.05 × 0.4 × 0.1 = 2mm/step → ~10cm drift over 2500 steps
  Heading   slip: σ_hdg = K_SLIP_HDG × |ω| × dt
    At max ω: σ_hdg = 0.01 × 1.9 × 0.1 = 1.9mrad/step → ~9.5deg over 2500 steps

Leaky Integrator Jerk Penalty (NEW):
  Exploit closed: flat jerk penalty allowed sustained oscillation at constant cost.
  The agent widened its 1D LiDAR sweep by oscillating, relaxing the velocity budget.
  Fix via leaky integrator with quadratic penalty:
    δω*_t = |ω_t - ω_{t-1}| / ω_max          (normalised angular velocity change)
    J_t   = λ·J_{t-1} + δω*_t,  λ = 0.85
    r_jerk = −k_j · J_t²,       k_j = 0.02
  Analysis:
    Single evasive manoeuvre: J→1.0, r_jerk=−0.02 (trivial, decays ~86% in 700ms)
    Sustained oscillation:    J→J∞=13.33, r_jerk=−3.55/step
    Forward progress reward: ~+0.4/step max → oscillation is 10× more costly. Closed.
  J_t / J_max included in obs so the network sees and regulates its own jerk debt.

Path Efficiency Reward (NEW):
  A* on a 2000×2000 pixel grid during training = O(4M·log4M) per reset → deadlock.
  Solution: waypoints[] in .npz ARE already the A* optimal path (MapGenerator writes them).
  Ideal path length L* = Σ||w_{i+1}−w_i||  cost O(n_waypoints) ≈ O(8) per reset.
  Terminal efficiency bonus (success only, avoids rewarding inefficient crashes):
    r_eff = K_EFF × (L* / max(L_actual, L*))   ∈ (0, 200]
  A direct bee-line success earns +700 total (+500 success + up to +200 efficiency).
  No per-step shaping — heading penalty already handles directional efficiency per step.

Reward Backbone Choice:
  V1's [128,128,128]+Tanh performed better on hard maps than V3.
  Key difference: V1 used SPEED_SCALE=1.5m (aggressive braking) vs V3's 3.0m (conservative).
  V5 uses SPEED_SCALE=2.0m (balanced, accounts for τ_v up to 1.15s inertia).
  Worst-case stopping distance: τ_v=1.15s at v=0.4m/s → ~0.44m + 0.08m act delay = 0.52m.
  SPEED_SCALE=2.0m provides 1.48m margin. Correct for hard-map safety.
  V3's quadratic velocity penalty and TTC backstop are KEPT (proven improvements).

OBSERVATION SPACE (28-dimensional)
────────────────────────────────────
  [0:17]  delayed scan     1-step FIFO, 100ms, /max_range         ∈ [0, 1]
  [17]    dist_to_goal     fresh from current pose, /10m           ∈ [0, 1]
  [18]    heading_error    fresh, /π                               ∈ [-1, 1]
  [19]    current_lin_vel  motor model output, /max_lin_vel        ∈ [0, 1]
  [20]    current_ang_vel  motor model output, /max_ang_vel        ∈ [-1, 1]
  [21]    jerk_integrator  J_t/J_max                              ∈ [0, 1]
    [22]    delayed_lin_vel  1-step delayed linear velocity          ∈ [0, 1]
    [23]    delayed_ang_vel  1-step delayed angular velocity         ∈ [-1, 1]
    [24]    act_fifo[0][0]   v commanded 200ms ago (executes now)    ∈ [-1, 1]
    [25]    act_fifo[0][1]   ω commanded 200ms ago                   ∈ [-1, 1]
    [26]    act_fifo[1][0]   v commanded 100ms ago (queued next)     ∈ [-1, 1]
    [27]    act_fifo[1][1]   ω commanded 100ms ago                   ∈ [-1, 1]

WHY INCLUDE FIFO CONTENTS IN OBS:
  With 200ms actuator lag, the network must anticipate what the robot WILL DO
  before issuing corrections. Without FIFO visibility, the agent cannot distinguish
  "I commanded stop 200ms ago and it hasn't executed yet" from "robot ignoring me."
  This ambiguity is the root cause of oscillatory braking in delay-unaware policies.

V1/V2/V3/V4 WEIGHTS ARE INCOMPATIBLE. Full retrain required (see TrainV5.py).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import os
import random
from numba import njit
from MapBank import MapBank


# ─────────────────────────────────────────────────────────────────────────────
# NUMBA RAYCASTER  (unchanged from V1/V3 — proven correct and JIT-cached)
# ─────────────────────────────────────────────────────────────────────────────

@njit(fastmath=True, cache=True)
def fast_raycast(rx, ry, rth, angles, map_grid, resolution, max_range):
    num_rays = angles.shape[0]
    scan     = np.zeros(num_rays, dtype=np.float32)
    h, w     = map_grid.shape
    max_px   = max_range * resolution

    for i in range(num_rays):
        glob_angle = rth + angles[i]
        dx = math.cos(glob_angle) * 2.0
        dy = math.sin(glob_angle) * 2.0
        curr_x  = rx * resolution
        curr_y  = ry * resolution
        dist_px = 0.0

        while dist_px < max_px:
            dist_px += 2.0
            curr_x  += dx
            curr_y  += dy
            ix = int(curr_x)
            iy = int(curr_y)
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                break
            if map_grid[iy, ix] == 1:
                break

        scan[i] = min(max_range, dist_px / resolution)

    return scan


# ─────────────────────────────────────────────────────────────────────────────
# MAP BANK V5  (extends MapBank with finetune pool support)
# ─────────────────────────────────────────────────────────────────────────────

class MapBankV5(MapBank):
    """
    Extends MapBank with a dedicated finetune map pool.

    Finetune pool activated when difficulty == FINETUNE_SENTINEL (-1.0).
    The sentinel value is set by the curriculum callback at the finetune tier.
    If finetune_dir does not exist or is empty, falls back to density 0.25
    (logged as warning — training continues uninterrupted).

    Directory structure expected:
      training_maps/
        diff_0.0/   *.npz
        diff_0.1/   *.npz
        diff_0.25/  *.npz
        diff_0.5/   *.npz
        diff_0.75/  *.npz
        diff_1.0/   *.npz
      finetune_maps/
        *.npz        (flat — no subdirectories)
    """
    FINETUNE_SENTINEL = -1.0

    def __init__(self, dataset_dir="training_maps", finetune_dir="finetune_maps"):
        self.finetune_dir   = finetune_dir
        self.finetune_paths = []
        super().__init__(dataset_dir)
        self._index_finetune()

    def _index_finetune(self):
        if not os.path.exists(self.finetune_dir):
            print(f"[MapBankV5] WARNING: finetune_dir='{self.finetune_dir}' not found. "
                  f"Finetune tier will use density=0.25 as fallback.")
            return
        self.finetune_paths = [
            os.path.join(self.finetune_dir, f)
            for f in os.listdir(self.finetune_dir) if f.endswith(".npz")
        ]
        n = len(self.finetune_paths)
        if n == 0:
            print(f"[MapBankV5] WARNING: finetune_dir='{self.finetune_dir}' is empty. "
                  f"Using density=0.25 fallback.")
        else:
            print(f"[MapBankV5] Indexed {n} finetune maps from '{self.finetune_dir}'.")

    def get_random_map(self):
        if self.current_difficulty == self.FINETUNE_SENTINEL:
            if self.finetune_paths:
                filepath = random.choice(self.finetune_paths)
                with np.load(filepath) as d:
                    return d["grid_map"], d["waypoints"], d["resolution"].item()
            else:
                # Graceful degradation
                self.current_difficulty = 0.25
        return super().get_random_map()


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class PeopleBotEnv(gym.Env):
    """
    PeopleBotEnv V5
    ── Domain Randomization (τ_v, τ_w, positional slip)
    ── Asymmetric Delay (100ms obs / 200ms actuator)
    ── Leaky Integrator Jerk Penalty
    ── Path Efficiency Terminal Bonus
    ── V1 Reward Backbone + V3 TTC + V3 Quadratic Velocity Penalty
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # ── Delay model constants ────────────────────────────────────────────────
    OBS_DELAY_STEPS    = 1   # 100ms — unified sensor pipeline delay
    ACTUATOR_LAG_STEPS = 2   # 200ms — command → ESP32 Rx → P2OS → motor

    # ── Leaky integrator (jerk penalty) ─────────────────────────────────────
    JERK_LAMBDA = 0.85                               # per-step decay factor
    JERK_K      = 0.02                               # quadratic coefficient
    J_MAX       = 2.0 / (1.0 - JERK_LAMBDA)         # = 13.33 — normalisation

    # ── Slip coefficients ────────────────────────────────────────────────────
    K_SLIP_POS = 0.05    # σ_pos = K_SLIP_POS × |v| × dt  (m)
    K_SLIP_HDG = 0.01    # σ_hdg = K_SLIP_HDG × |ω| × dt  (rad)

    # ── Reward constants ─────────────────────────────────────────────────────
    EXIST_TAX         = -0.05
    PROGRESS_SCALE    = 10.0
    VEL_EXCESS_SCALE  = 20.0   # quadratic multiplier
    SPEED_SCALE       = 2.0    # front clearance threshold for full speed (m)
    SPEED_SCALE_SIDE  = 1.5    # side clearance threshold (m)
    SIDE_WEIGHT       = 0.35   # lateral blend weight
    TTC_SAFE          = 1.5    # time-to-collision safety margin (s)
    TTC_COEF          = 4.0
    TTC_EPS           = 0.05   # minimum v for TTC computation (m/s)
    HDG_PENALTY_SCALE = 1.5
    WP_BASE_BONUS     = 10.0
    WP_QUALITY_BONUS  = 25.0
    CRASH_PENALTY     = -500.0
    SUCCESS_REWARD    = 500.0
    TIMEOUT_PENALTY   = -150.0
    EFF_SCALE         = 200.0  # max path efficiency bonus (on success)

    def __init__(self, dataset_dir="training_maps", finetune_dir="finetune_maps"):
        super().__init__()

        # ── Hardware specs ───────────────────────────────────────────────────
        self.wheel_radius = 0.0955   # m
        self.wheel_base   = 0.33     # m (axle separation)
        self.robot_radius = 0.31     # m (collision footprint)

        # ── Kinematic limits ─────────────────────────────────────────────────
        self.max_lin_vel = 0.4       # m/s  (forward-only; no rear sensors)
        self.min_lin_vel = 0.0
        self.max_ang_vel = 1.9       # rad/s

        # ── Timing ───────────────────────────────────────────────────────────
        self.dt = 0.1   # 10 Hz inference loop

        # ── Derived limits ───────────────────────────────────────────────────
        self.max_wheel_omega = (
            self.max_lin_vel + self.max_ang_vel * self.wheel_base / 2.0
        ) / self.wheel_radius

        # ── Safety ───────────────────────────────────────────────────────────
        self.tipping_threshold = 2.0  # |v × ω| (m·rad/s²) centrifugal proxy

        # ── Sensor configuration (17-ray front-arc, identical to V1/V3) ──────
        self.max_sensor_range = 3.0
        front_degs = [90, 50, 30, 25, 20, 15, 10, 5, 0,
                      -5, -10, -15, -20, -25, -30, -50, -90]
        self.sensor_angles = np.deg2rad(front_degs).astype(np.float32)
        self.num_rays      = len(self.sensor_angles)   # 17

        # Spatial indices for reward gating (V1 convention: spatial, not hardware)
        self.front_indices = np.where(
            np.abs(self.sensor_angles) <= np.deg2rad(30)
        )[0]
        self.side_indices = np.where(
            np.abs(self.sensor_angles) > np.deg2rad(30)
        )[0]

        # ── Delay FIFOs (reset in reset()) ───────────────────────────────────
        # obs_scan_fifo  : shape (1, 17)  — 100ms obs delay
        # action_fifo    : shape (2,  2)  — 200ms actuator delay
        self.obs_scan_fifo = np.full(
            (self.OBS_DELAY_STEPS, self.num_rays),
            self.max_sensor_range, dtype=np.float32
        )
        self.obs_vel_fifo = np.zeros(
            (self.OBS_DELAY_STEPS, 2), dtype=np.float32
        )
        self.action_fifo = np.zeros(
            (self.ACTUATOR_LAG_STEPS, 2), dtype=np.float32
        )

        # ── Navigation ───────────────────────────────────────────────────────
        self.waypoint_radius = 1.5
        self.goal_radius     = 1.0
        self.max_steps       = 2500

        # ── Observation space: 28-dimensional ────────────────────────────────
        # 17 scan + 5 state + 2 delayed velocity + 4 action history = 28
        obs_size = self.num_rays + 5 + 2 + (self.ACTUATOR_LAG_STEPS * 2)
        assert obs_size == 28, f"V5 obs size mismatch: {obs_size}"
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # ── Map bank ─────────────────────────────────────────────────────────
        self.map_bank   = MapBankV5(dataset_dir=dataset_dir, finetune_dir=finetune_dir)
        self.difficulty = 0.0

        # ── Episode state (initialised properly in reset()) ───────────────────
        self.map_grid           = None
        self.resolution         = 50
        self.waypoints          = []
        self.current_pose       = np.zeros(3, dtype=np.float64)
        self.current_goal_index = 0
        self.current_goal       = np.zeros(2, dtype=np.float32)
        self.previous_distance  = 0.0
        self.current_step       = 0

        # Domain randomisation (overwritten each reset)
        self.tau_v = 0.30   # linear  inertia (s)
        self.tau_w = 0.06   # angular inertia (s)

        # Dynamics
        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0
        self.prev_ang_vel    = 0.0

        # Leaky integrator
        self.jerk_integrator = 0.0

        # Path efficiency
        self.ideal_path_length     = 1.0
        self.traversed_path_length = 0.0

        # Telemetry
        self.ep_velocity_history  = []
        self.ep_min_lidar_history = []
        self.ep_vibration_count   = 0
        self.ep_checkpoints_hit   = 0
        self.total_checkpoints    = 1

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def set_difficulty(self, difficulty_level: float):
        """
        Called by curriculum callback on tier transitions.
        Pass -1.0 (MapBankV5.FINETUNE_SENTINEL) for finetune map pool.
        """
        self.difficulty = float(difficulty_level)
        self.map_bank.set_difficulty(self.difficulty)

    # ─────────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # ── Flush all delay pipelines ────────────────────────────────────────
        self.obs_scan_fifo[:] = self.max_sensor_range
        self.obs_vel_fifo[:]  = 0.0
        self.action_fifo[:]   = 0.0

        # ── Domain randomisation (new τ per episode) ──────────────────────────
        # Using numpy RNG seeded from Gym seed for reproducibility
        rng        = np.random.default_rng(seed)
        self.tau_v = float(rng.uniform(0.50, 1.15))
        self.tau_w = float(rng.uniform(0.03, 0.10))

        # ── Reset dynamics ────────────────────────────────────────────────────
        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0
        self.prev_ang_vel    = 0.0
        self.jerk_integrator = 0.0

        # ── Telemetry buffers ─────────────────────────────────────────────────
        self.ep_velocity_history  = []
        self.ep_min_lidar_history = []
        self.ep_vibration_count   = 0
        self.ep_checkpoints_hit   = 0

        # ── Load map ──────────────────────────────────────────────────────────
        self.map_grid, raw_wps, self.resolution = self.map_bank.get_random_map()
        self.waypoints         = [list(wp) for wp in raw_wps]
        self.total_checkpoints = max(1, len(self.waypoints) - 1)

        # ── Pre-compute ideal A* path length (O(n_waypoints) ≈ O(8)) ────────
        # The waypoints[] are the A* solution already embedded in the .npz file.
        # No pathfinding needed at runtime — this is essentially free.
        self.ideal_path_length = max(0.1, sum(
            math.hypot(self.waypoints[i + 1][0] - self.waypoints[i][0],
                       self.waypoints[i + 1][1] - self.waypoints[i][1])
            for i in range(len(self.waypoints) - 1)
        ))
        self.traversed_path_length = 0.0

        # ── Set start pose ────────────────────────────────────────────────────
        start_pt = self.waypoints[0]
        if len(self.waypoints) > 1:
            self.current_goal_index = 1
            self.current_goal       = np.array(self.waypoints[1], dtype=np.float32)
            start_theta = math.atan2(
                self.current_goal[1] - start_pt[1],
                self.current_goal[0] - start_pt[0]
            )
        else:
            self.current_goal_index = 0
            self.current_goal       = np.array(start_pt, dtype=np.float32)
            start_theta = 0.0

        self.current_pose = np.array(
            [start_pt[0], start_pt[1], start_theta], dtype=np.float64
        )
        self.previous_distance = float(
            np.linalg.norm(self.current_pose[:2] - self.current_goal)
        )
        self.current_step = 0

        # ── Warm obs FIFO with ground-truth initial scan ──────────────────────
        # Prevents the first observation from being an uninformative max-range field
        init_scan = fast_raycast(
            self.current_pose[0], self.current_pose[1], self.current_pose[2],
            self.sensor_angles, self.map_grid, self.resolution, self.max_sensor_range
        )
        self.obs_scan_fifo[:] = init_scan  # pre-warm all slots identically
        self.obs_vel_fifo[:]  = 0.0

        return self._get_obs(), {}

    # ─────────────────────────────────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        self.current_step += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # ── Vibration counter (request vs last-queued action) ────────────────
        delta = np.abs(action - self.action_fifo[-1])
        if delta[0] > 0.4 or delta[1] > 0.4:
            self.ep_vibration_count += 1

        # ── ACTUATOR DELAY FIFO ───────────────────────────────────────────────
        # Semantics:
        #   action_fifo[0] = command issued 200ms ago → EXECUTE THIS STEP
        #   action_fifo[1] = command issued 100ms ago → queued for next step
        # After update:
        #   action_fifo[0] ← old [1]  (100ms old, will execute next step)
        #   action_fifo[1] ← action   (new command, will execute in 200ms)
        delayed_action      = self.action_fifo[0].copy()
        self.action_fifo[0] = self.action_fifo[1]
        self.action_fifo[1] = action

        # ── Action → velocity target (forward-only) ───────────────────────────
        norm_lin = float(delayed_action[0])
        norm_ang = float(delayed_action[1])
        req_v    = norm_lin * self.max_lin_vel if norm_lin >= 0.0 else 0.0
        req_w    = norm_ang * self.max_ang_vel

        # ── Differential drive kinematic limits ───────────────────────────────
        req_wl = (req_v - req_w * self.wheel_base / 2.0) / self.wheel_radius
        req_wr = (req_v + req_w * self.wheel_base / 2.0) / self.wheel_radius
        max_req = max(abs(req_wl), abs(req_wr))
        if max_req > self.max_wheel_omega:
            s       = self.max_wheel_omega / max_req
            req_wl *= s
            req_wr *= s
        target_v = (self.wheel_radius / 2.0)             * (req_wl + req_wr)
        target_w = (self.wheel_radius / self.wheel_base) * (req_wr - req_wl)

        # ── Motor dynamics with per-episode randomised inertia ────────────────
        # Decoupled first-order EMA filters:
        #   α_v = dt / (τ_v + dt)   τ_v ~ U[0.50, 1.15]s
        #   α_w = dt / (τ_w + dt)   τ_w ~ U[0.03, 0.10]s
        alpha_v = self.dt / (self.tau_v + self.dt)
        alpha_w = self.dt / (self.tau_w + self.dt)
        self.current_lin_vel += alpha_v * (target_v - self.current_lin_vel)
        self.current_ang_vel += alpha_w * (target_w - self.current_ang_vel)

        # ── Observation-state FIFO update (1-step delayed velocities) ────────
        delayed_vel = self.obs_vel_fifo[0].copy()
        self.obs_vel_fifo[:-1] = self.obs_vel_fifo[1:]
        self.obs_vel_fifo[-1]  = np.array(
            [self.current_lin_vel, self.current_ang_vel], dtype=np.float32
        )

        # ── Kinematics ────────────────────────────────────────────────────────
        theta = self.current_pose[2]
        dx    = self.current_lin_vel * math.cos(theta) * self.dt
        dy    = self.current_lin_vel * math.sin(theta) * self.dt
        dth   = self.current_ang_vel * self.dt

        # ── Positional + heading slip (decoupled from ideal kinematics) ───────
        # Simulates wheel odometry drift / floor-surface variation / IMU noise.
        # The policy is NEVER given true ground-truth position — only from the
        # slipped pose — so it must learn to be robust to this accumulated error.
        v_abs  = abs(self.current_lin_vel)
        w_abs  = abs(self.current_ang_vel)
        slip_x = np.random.randn() * self.K_SLIP_POS * v_abs * self.dt
        slip_y = np.random.randn() * self.K_SLIP_POS * v_abs * self.dt
        slip_h = np.random.randn() * self.K_SLIP_HDG * w_abs * self.dt

        self.current_pose[0] += dx + slip_x
        self.current_pose[1] += dy + slip_y
        self.current_pose[2]  = self._angdiff(0.0, theta + dth + slip_h)

        # ── Traversed path tracking (ideal motion only, no slip component) ────
        self.traversed_path_length += math.hypot(dx, dy)

        # ── Physics states ────────────────────────────────────────────────────
        centrifugal = abs(self.current_lin_vel * self.current_ang_vel)
        is_tipped   = centrifugal > self.tipping_threshold
        is_collided = self._check_collision()

        # ── Goal logic ────────────────────────────────────────────────────────
        dist_to_goal   = float(np.linalg.norm(self.current_pose[:2] - self.current_goal))
        hit_checkpoint = False
        hit_final_goal = False

        if dist_to_goal < self.waypoint_radius:
            if self.current_goal_index < len(self.waypoints) - 1:
                self.current_goal_index += 1
                self.current_goal = np.array(
                    self.waypoints[self.current_goal_index], dtype=np.float32
                )
                hit_checkpoint         = True
                self.previous_distance = float(
                    np.linalg.norm(self.current_pose[:2] - self.current_goal)
                )
                dist_to_goal = self.previous_distance
            else:
                if dist_to_goal < self.goal_radius:
                    hit_final_goal = True

        # ── Raycasting (ground truth + sensor noise) ──────────────────────────
        fresh_scan = fast_raycast(
            self.current_pose[0], self.current_pose[1], self.current_pose[2],
            self.sensor_angles, self.map_grid, self.resolution, self.max_sensor_range
        )
        noise      = np.random.randn(self.num_rays).astype(np.float32) * 0.02
        fresh_scan = np.clip(fresh_scan + noise, 0.0, self.max_sensor_range)

        # ── Obs sensor FIFO update (1-step delay = 100ms) ─────────────────────
        # Pull the oldest entry for the observation BEFORE pushing the new one.
        delayed_scan           = self.obs_scan_fifo[0].copy()
        self.obs_scan_fifo[:-1] = self.obs_scan_fifo[1:]
        self.obs_scan_fifo[-1]  = fresh_scan

        # ── Navigation signals (computed from current slipped pose) ───────────
        desired_angle = math.atan2(
            self.current_goal[1] - self.current_pose[1],
            self.current_goal[0] - self.current_pose[0]
        )
        heading_error = self._angdiff(self.current_pose[2], desired_angle)

        # ── LEAKY INTEGRATOR — jerk update ────────────────────────────────────
        # δω*_t normalised by max angular velocity so units are consistent
        # regardless of the commanded ω range.
        delta_w_norm         = (
            abs(self.current_ang_vel - self.prev_ang_vel) / self.max_ang_vel
        )
        self.jerk_integrator = (
            self.JERK_LAMBDA * self.jerk_integrator + delta_w_norm
        )
        self.prev_ang_vel    = self.current_ang_vel

        # ── REWARD COMPUTATION ────────────────────────────────────────────────
        terminated = False
        truncated  = False
        reward     = self.EXIST_TAX

        # Clearance variables from delayed scan (matches what policy acted on)
        front_dist = float(np.min(delayed_scan[self.front_indices]))
        side_dist  = float(np.min(delayed_scan[self.side_indices]))

        # ─── 1. Progress ──────────────────────────────────────────────────────
        dist_improvement    = self.previous_distance - dist_to_goal
        reward             += self.PROGRESS_SCALE * dist_improvement
        self.previous_distance = dist_to_goal

        # ─── 2. Velocity budget (V1 aggressive + V3 quadratic penalty) ────────
        # budget_ratio: 0.0 = wall contact, 1.0 = clear open space
        front_ratio  = float(np.clip(front_dist / self.SPEED_SCALE,      0.0, 1.0))
        side_ratio   = float(np.clip(side_dist  / self.SPEED_SCALE_SIDE, 0.0, 1.0))
        budget_ratio = (front_ratio * (1.0 - self.SIDE_WEIGHT) +
                        side_ratio  * self.SIDE_WEIGHT)

        desired_max_v   = budget_ratio * self.max_lin_vel
        velocity_excess = max(0.0, self.current_lin_vel - desired_max_v)
        if velocity_excess > 0.01:
            reward -= self.VEL_EXCESS_SCALE * (velocity_excess ** 2)

        # ─── 3. TTC exponential backstop (V3) ─────────────────────────────────
        # Prevents late close-range dives that the velocity budget misses.
        if self.current_lin_vel > self.TTC_EPS:
            ttc = front_dist / self.current_lin_vel
            if ttc < self.TTC_SAFE:
                reward -= self.TTC_COEF * ((self.TTC_SAFE / ttc - 1.0) ** 2)

        # ─── 4. Heading penalty (gated — allow evasive turns near walls) ───────
        reward -= (self.HDG_PENALTY_SCALE *
                   (1.0 - math.cos(heading_error)) * budget_ratio)

        # ─── 5. Leaky integrator jerk penalty (V5 new) ────────────────────────
        # Quadratic on J_t: single evasive move costs ~−0.02, sustained
        # oscillation costs −3.55/step >> progress reward ~+0.4/step.
        reward -= self.JERK_K * (self.jerk_integrator ** 2)

        # ─── 6. Waypoint bonus ────────────────────────────────────────────────
        if hit_checkpoint:
            approach_quality  = max(0.0, math.cos(heading_error))
            reward           += self.WP_BASE_BONUS + self.WP_QUALITY_BONUS * approach_quality
            self.ep_checkpoints_hit += 1

        # ─── 7. Terminal events ───────────────────────────────────────────────
        if is_collided or is_tipped:
            reward     += self.CRASH_PENALTY
            terminated  = True

        elif hit_final_goal:
            # Path efficiency bonus: max +200 for perfect bee-line navigation.
            # r_eff = 200 × (L* / max(L_actual, L*))  ∈ (0, 200]
            eff_ratio   = self.ideal_path_length / max(
                self.traversed_path_length, self.ideal_path_length
            )
            reward     += self.SUCCESS_REWARD + self.EFF_SCALE * eff_ratio
            terminated  = True

        if self.current_step >= self.max_steps and not terminated:
            reward    += self.TIMEOUT_PENALTY
            truncated  = True

        # ── Telemetry update ──────────────────────────────────────────────────
        self.ep_velocity_history.append(self.current_lin_vel)
        self.ep_min_lidar_history.append(min(front_dist, side_dist))

        info = {}
        if terminated or truncated:
            eff_ratio_final = self.ideal_path_length / max(
                self.traversed_path_length, self.ideal_path_length
            )
            info["telemetry"] = {
                # Core metrics
                "rate_success":            1.0 if hit_final_goal else 0.0,
                "rate_crash":              1.0 if (is_collided or is_tipped) else 0.0,
                "rate_timeout":            1.0 if (truncated and not hit_final_goal) else 0.0,
                # Navigation quality
                "avg_velocity":            float(np.mean(self.ep_velocity_history))
                                           if self.ep_velocity_history else 0.0,
                "avg_wall_clearance":      float(np.mean(self.ep_min_lidar_history))
                                           if self.ep_min_lidar_history else 0.0,
                "checkpoint_capture_rate": float(self.ep_checkpoints_hit /
                                                  self.total_checkpoints),
                # V5 diagnostic metrics
                "path_efficiency":         float(eff_ratio_final),
                "traversed_vs_ideal":      float(self.traversed_path_length /
                                                  max(self.ideal_path_length, 0.1)),
                "vibration_events":        float(self.ep_vibration_count),
                "jerk_integrator_peak":    float(self.jerk_integrator),
                # Domain randomisation (for post-hoc analysis)
                "tau_v":                   float(self.tau_v),
                "tau_w":                   float(self.tau_w),
            }

        obs = self._build_obs(
            delayed_scan,
            dist_to_goal,
            heading_error,
            delayed_vel[0],
            delayed_vel[1],
        )
        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVATION BUILDER
    # ─────────────────────────────────────────────────────────────────────────

    def _build_obs(self, delayed_scan: np.ndarray,
                   dist_to_goal: float,
                   heading_error: float,
                   delayed_lin_vel: float,
                   delayed_ang_vel: float) -> np.ndarray:
        norm_scan  = delayed_scan / self.max_sensor_range          # [0:17]  ∈ [0, 1]
        norm_dist  = float(min(dist_to_goal / 10.0, 1.0))         # [17]    ∈ [0, 1]
        norm_head  = float(heading_error / math.pi)                # [18]    ∈ [-1, 1]
        norm_v     = float(self.current_lin_vel / self.max_lin_vel)# [19]    ∈ [0, 1]
        norm_w     = float(self.current_ang_vel / self.max_ang_vel)# [20]    ∈ [-1, 1]
        norm_jerk  = float(min(self.jerk_integrator / self.J_MAX, 1.0))  # [21] ∈ [0,1]
        norm_del_v = float(delayed_lin_vel / self.max_lin_vel)     # [22]    ∈ [0,1]
        norm_del_w = float(delayed_ang_vel / self.max_ang_vel)     # [23]    ∈ [-1,1]
        flat_fifo  = self.action_fifo.flatten()                    # [24:28] ∈ [-1, 1]

        obs = np.concatenate([
            norm_scan,
            [norm_dist, norm_head, norm_v, norm_w, norm_jerk, norm_del_v, norm_del_w],
            flat_fifo
        ]).astype(np.float32)

        assert obs.shape == (28,), f"V5 obs shape error: {obs.shape}"
        return obs

    def _get_obs(self) -> np.ndarray:
        """Called from reset() — FIFOs pre-warmed, dynamics zeroed."""
        delayed_scan  = self.obs_scan_fifo[0].copy()
        delayed_vel   = self.obs_vel_fifo[0].copy()
        dist_to_goal  = float(np.linalg.norm(self.current_pose[:2] - self.current_goal))
        desired_angle = math.atan2(
            self.current_goal[1] - self.current_pose[1],
            self.current_goal[0] - self.current_pose[0]
        )
        heading_error = self._angdiff(self.current_pose[2], desired_angle)
        return self._build_obs(
            delayed_scan,
            dist_to_goal,
            heading_error,
            delayed_vel[0],
            delayed_vel[1],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # COLLISION & UTILITY
    # ─────────────────────────────────────────────────────────────────────────

    def _check_collision(self) -> bool:
        cx, cy = self.current_pose[0], self.current_pose[1]
        if self._is_occupied(cx, cy):
            return True
        for ang in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            if self._is_occupied(cx + self.robot_radius * math.cos(ang),
                                  cy + self.robot_radius * math.sin(ang)):
                return True
        return False

    def _is_occupied(self, x: float, y: float) -> bool:
        ix = int(x * self.resolution)
        iy = int(y * self.resolution)
        if ix < 0 or ix >= self.map_grid.shape[1]:
            return True
        if iy < 0 or iy >= self.map_grid.shape[0]:
            return True
        return bool(self.map_grid[iy, ix] == 1)

    def _angdiff(self, th1: float, th2: float) -> float:
        return (th2 - th1 + math.pi) % (2 * math.pi) - math.pi