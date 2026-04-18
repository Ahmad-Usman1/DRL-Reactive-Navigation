"""
FineTunePPO.py
==============
Finetuning script that loads a pre-trained PPO model and continues training
with a targeted reward architecture and specialised maps.

Targeted failure modes being fixed:
  1. Corner crashes        — corner-aware velocity budget blends front+directional-side
  2. Waypoint heading spike — 15-step grace period after checkpoint transition
  3. Narrow passage avoidance — progress bonus that scales with corridor tightness
  4. Excessive speed        — velocity budget penalty coefficient increased

Architecture:
  FineTuneBotEnv subclasses PeopleBotEnv.
  Calling super().step() runs ALL physics (collision, goal detection, sensor
  raycasting, delay simulation) exactly as trained. The parent's reward is
  discarded and replaced by _compute_ft_reward().

  Checkpoint detection: compare current_goal_index before/after super().step().
  ep_checkpoints_hit is NOT incremented here — the parent already does it.

Hyperparameter changes from baseline:
  learning_rate : 3e-4 → 8e-5  (conservative — preserve base policy)
  n_epochs      : 10   → 5     (less aggressive updates per batch)
  ent_coef      : 0.005 → 0.015 (more exploration to learn new corner behavior)
  clip_range    : 0.2  → 0.1   (smaller policy step to avoid catastrophic forgetting)
  gamma         : 0.995 → 0.995 (unchanged — effective horizon ~200 steps is correct)

Outputs:
  Models → ./finetune_models/
  TensorBoard → ./finetune_tensorboard/
"""

import os
import math
import numpy as np
import gymnasium as gym
import torch.nn as nn
from typing import Callable
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from PeopleBotEnv import PeopleBotEnv
from FineTuneMaps import FineTuneMapBank

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
PRETRAINED_MODEL_PATH = "finetune_modelsv2\\BEANS_FineTuned_Final.zip"

FT_LOG_DIR   = "./finetune_tensorboard/"
FT_MODEL_DIR = "./finetune_modelsv3/"
FT_TIMESTEPS = 4_000_000
N_ENVS       = 16

os.makedirs(FT_LOG_DIR,   exist_ok=True)
os.makedirs(FT_MODEL_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FINETUNING ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class FineTuneBotEnv(PeopleBotEnv):
    """
    Subclass of PeopleBotEnv with:
      - FineTuneMapBank replacing the standard MapBank
      - Corner-aware velocity budget in reward
      - 15-step grace period after waypoint transitions
      - Narrow-passage progress bonus
      - Tighter waypoint/goal radii (1.2m / 0.9m) to force accurate navigation

    Physics, sensors, delay simulation and observation space are UNCHANGED.
    This means the pre-trained weights transfer cleanly.
    """

    def __init__(self):
        super().__init__()

        # Override map bank — all other parent state is unchanged
        self.ft_map_bank = FineTuneMapBank(
            standard_map_dir="training_maps",
            standard_difficulty=0.75
        )

        # Tighter acceptance radii: force the robot to actually reach each waypoint
        # rather than arcing through from the side
        self.waypoint_radius = 1.2   # was 2.5 in original, 1.5 after prior fix
        self.goal_radius     = 0.9   # was 1.5 in original

        # Finetuning-specific state (not present in parent)
        self.steps_since_checkpoint = 0
        self.ft_previous_distance   = 0.0

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Call parent reset (initialises all base state with a standard map),
        then override the map and re-derive pose/goal from the new waypoints.
        """
        # Parent handles: action_history flush, telemetry reset, current_step=0
        super().reset(seed=seed, options=options)

        # Load a finetuning map
        self.map_grid, wps, self.resolution = self.ft_map_bank.get_random_map()

        # wps must be a 2D numpy array: shape (N, 2)
        if not isinstance(wps, np.ndarray):
            wps = np.array(wps, dtype=np.float32)
        self.waypoints = wps

        self.total_checkpoints = max(1, len(self.waypoints) - 1)

        # Re-derive start pose and first goal from the new waypoints
        start_pt = self.waypoints[0]

        if len(self.waypoints) > 1:
            self.current_goal_index = 1
            self.current_goal = np.array(self.waypoints[1], dtype=np.float32)
            start_theta = math.atan2(
                self.current_goal[1] - start_pt[1],
                self.current_goal[0] - start_pt[0]
            )
        else:
            self.current_goal_index = 0
            self.current_goal = np.array(start_pt, dtype=np.float32)
            start_theta = 0.0

        self.current_pose       = np.array([start_pt[0], start_pt[1], start_theta])
        self.current_lin_vel    = 0.0
        self.current_ang_vel    = 0.0
        self.previous_distance  = float(np.linalg.norm(
            self.current_pose[:2] - self.current_goal))
        self.current_step       = 0

        # Finetuning-specific state
        self.steps_since_checkpoint = 0
        self.ft_previous_distance   = self.previous_distance

        return self._get_obs(), {}

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        """
        Run parent physics, discard parent reward, substitute ft reward.

        Pre-step state capture:
          prev_goal_index — to detect checkpoint transition
          ft_prev_dist    — our own distance tracker (not parent's)

        The parent's ep_checkpoints_hit is already incremented inside
        super().step() when a checkpoint fires. We MUST NOT increment it again.
        """
        prev_goal_index = self.current_goal_index
        ft_prev_dist    = self.ft_previous_distance

        # ── Run physics, sensor raycasting, collision, goal detection ──────
        obs, _parent_reward, terminated, truncated, info = super().step(action)

        # ── Detect events from post-step state ─────────────────────────────
        hit_checkpoint = self.current_goal_index > prev_goal_index

        tel            = info.get("telemetry", {})
        hit_final_goal = terminated and tel.get("rate_success",  0.0) == 1.0
        is_crashed     = terminated and tel.get("rate_crash",    0.0) == 1.0
        # Note: truncated is already set correctly by parent when step >= max_steps

        # ── Recover sensor values from obs (avoids re-running raycaster) ───
        # obs layout: [scan(17), dist(1), head(1), lin_v(1), ang_v(1), hist(6)]
        scan_data     = obs[:self.num_rays] * self.max_sensor_range
        heading_error = obs[self.num_rays + 1] * np.pi   # de-normalise

        # ── Update our own distance tracker ────────────────────────────────
        current_dist = float(np.linalg.norm(self.current_pose[:2] - self.current_goal))

        if hit_checkpoint:
            # Credit the full distance to the completed waypoint segment.
            # Do NOT use current_dist here — it's the distance to the NEW goal.
            dist_improvement = ft_prev_dist
        else:
            dist_improvement = ft_prev_dist - current_dist

        self.ft_previous_distance = current_dist

        # ── Update grace period counter ─────────────────────────────────────
        if hit_checkpoint:
            self.steps_since_checkpoint = 0
        else:
            self.steps_since_checkpoint += 1

        # ── Compute finetuned reward ────────────────────────────────────────
        reward = self._compute_ft_reward(
            scan_data      = scan_data,
            heading_error  = heading_error,
            dist_improvement = dist_improvement,
            hit_checkpoint = hit_checkpoint,
            hit_final_goal = hit_final_goal,
            is_crashed     = is_crashed,
            truncated      = truncated,
        )

        return obs, reward, terminated, truncated, info

    # ── reward ────────────────────────────────────────────────────────────────

    def _compute_ft_reward(self, scan_data, heading_error, dist_improvement,
                           hit_checkpoint, hit_final_goal, is_crashed, truncated):
        """
        Finetuned reward architecture.

        Changes from the original:

        1. Velocity budget uses corner-aware effective distance:
           When turning, the directional side clearance is blended in proportion
           to the turn rate. This catches the case where front_dist is large but
           the inside wall of a corner is closing fast.

        2. Heading penalty has a 15-step grace period after waypoint transitions.
           When current_goal changes, heading_error jumps by up to ±π in one step.
           The robot physically cannot instantaneously align. Without the grace
           period, it receives a huge misalignment penalty immediately after a
           checkpoint hit, which trains it to over-steer to recover heading
           (causing the wall collision the prior version exhibited).

        3. Narrow passage bonus: extra progress reward when min_dist < 0.8m.
           Without this, the bot has learned that tight spaces are high-penalty
           zones. The bonus makes "committed progress through narrow = positive"
           rather than "narrow = only negative (safety penalties)".

        4. Velocity excess coefficient increased 10→12: the original was not
           sufficient to prevent corner entry at speed.

        Terminal rewards are unchanged from the working 75%-success version to
        avoid disrupting the already-learned terminal behaviour.
        """

        front_dist = float(np.min(scan_data[self.front_indices]))
        side_dist  = float(np.min(scan_data[self.side_indices]))
        min_dist   = min(front_dist, side_dist)

        # ── 1. Existence tax ────────────────────────────────────────────────
        reward = -0.05

        # ── 2. Progress ─────────────────────────────────────────────────────
        reward += 10.0 * dist_improvement

        # --- 3. The Velocity Budget (Predictive Lateral Tightness Version) ---
        SPEED_SCALE_DIST = 3.0       # (from Solution 1, or keep at 1.5 if not using S1)
        SIDE_TIGHTNESS_SCALE = 2.0   # Distance below which side walls constrain speed
        SIDE_WEIGHT = 0.45           # How much lateral tightness bleeds into speed limit

        front_dist = float(np.min(scan_data[self.front_indices]))
        side_dist  = float(np.min(scan_data[self.side_indices]))

        # budget_ratio = 1.0 → full speed allowed; 0.0 → must stop
        front_ratio = np.clip(front_dist / SPEED_SCALE_DIST, 0.0, 1.0)
        side_ratio  = np.clip(side_dist / SIDE_TIGHTNESS_SCALE, 0.0, 1.0)

        # Geometric mean: both distances must be clear for full speed to be permitted
        budget_ratio = front_ratio * (1.0 - SIDE_WEIGHT) + side_ratio * SIDE_WEIGHT

        desired_max_v = budget_ratio * self.max_lin_vel
        velocity_excess = max(0.0, self.current_lin_vel - desired_max_v)
        if velocity_excess > 0.01:
            reward -= 20.0 * (velocity_excess ** 2)  # pair with Solution 1's quadratic

        # ── 4. Heading penalty with grace period ────────────────────────────
        # grace_factor: 0.0 immediately after checkpoint → 1.0 after 15 steps
        # This prevents the heading penalty from firing during the inevitable
        # misalignment period right after the goal pointer jumps to the next waypoint.
        grace_factor = min(1.0, self.steps_since_checkpoint / 15.0)

        reward -= (1.5
                   * (1.0 - math.cos(heading_error))
                   * budget_ratio
                   * grace_factor)

        # ── 5. Narrow passage progress bonus ────────────────────────────────
        # The bot has learned that tight spaces = safety penalties = avoid.
        # This bonus shifts the sign: progress THROUGH tight spaces is rewarded.
        # Only fires when actually making forward progress to avoid rewarding
        # scraping along a wall sideways.
        if min_dist < 0.8 and dist_improvement > 0.001:
            tightness = (0.8 - min_dist) / 0.8   # 0 at 0.8m → 1.0 at contact
            reward += 2.5 * tightness * dist_improvement

        # ── 6. Waypoint & terminal rewards (unchanged from 75%-success version)
        if hit_checkpoint:
            approach_quality = max(0.0, math.cos(heading_error))
            # ep_checkpoints_hit already incremented by parent — do NOT touch it here
            reward += 10.0 + (25.0 * approach_quality)

        if is_crashed:
            reward -= 500.0
        elif hit_final_goal:
            reward += 500.0

        if truncated:
            reward -= 150.0

        return float(reward)

    def set_difficulty(self, difficulty_level):
        """Forward difficulty to the ft_map_bank's standard sub-bank."""
        self.ft_map_bank.set_difficulty(difficulty_level)


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

class FineTuneCurriculumCallback(BaseCallback):
    """
    Lightweight curriculum for finetuning: the map bank difficulty is locked
    at 0.5 (standard maps) throughout. This callback logs finetuning-specific
    metrics and provides a clear signal for when to stop.

    Stopping criteria: when rolling success rate exceeds 90% AND avg corner
    crash rate is below 5% for 100 episodes.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.success_history = deque(maxlen=100)
        self.crash_history   = deque(maxlen=100)

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [])[i]
                if "telemetry" in info:
                    tel = info["telemetry"]
                    self.success_history.append(tel.get("rate_success", 0.0))
                    self.crash_history.append(tel.get("rate_crash",    0.0))

        if self.success_history:
            self.logger.record("finetune/rolling_success_rate",
                               float(np.mean(self.success_history)))
        if self.crash_history:
            self.logger.record("finetune/rolling_crash_rate",
                               float(np.mean(self.crash_history)))

        return True


class TelemetryLoggerCallback(BaseCallback):
    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [])[i]
                if "telemetry" in info:
                    for key, value in info["telemetry"].items():
                        self.logger.record(f"telemetry/{key}", value)
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════════

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT FACTORY (Named function — required for SubprocVecEnv pickling)
# ═══════════════════════════════════════════════════════════════════════════════

def make_finetune_env():
    return Monitor(FineTuneBotEnv())


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(" BEANS Finetuning Pipeline")
    print(f" Pretrained model : {PRETRAINED_MODEL_PATH}")
    print(f" Output models    : {FT_MODEL_DIR}")
    print(f" TensorBoard      : {FT_LOG_DIR}")
    print(f" Parallel envs    : {N_ENVS}")
    print("=" * 60)

    # ── Build vectorised environment ────────────────────────────────────────
    env = make_vec_env(make_finetune_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    # ── Load pre-trained weights ─────────────────────────────────────────────
    # We load into a temporary model to extract the policy state dict, then
    # construct a fresh PPO with the finetuning hyperparameters and transplant.
    # This avoids any internal SB3 state carrying over from the old training run
    # (e.g. optimizer momentum, rollout buffer contents).
    print("\nLoading pre-trained weights...")
    temp_model = PPO.load(PRETRAINED_MODEL_PATH, device="auto")
    pretrained_weights = temp_model.policy.state_dict()
    print(f"  Loaded {len(pretrained_weights)} parameter tensors.")
    del temp_model   # free memory

    # ── Policy architecture MUST match original exactly ─────────────────────
    # Changing this breaks the weight transplant. 128×128×128 + Tanh is correct.
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128])
    )

    # ── Create new PPO with finetuning hyperparameters ───────────────────────
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs   = policy_kwargs,
        verbose         = 1,
        tensorboard_log = FT_LOG_DIR,

        # --- Core finetuning hyperparameter changes ---
        learning_rate   = linear_schedule(4e-5),  # was 3e-4 — conservative
        n_epochs        = 5,                       # was 10  — less aggressive
        ent_coef        = 0.015,                   # was 0.005 — more exploration
        clip_range      = 0.1,                     # was 0.2  — prevent forgetting

        # --- Unchanged from original ---
        n_steps         = 2048,
        batch_size      = 64,
        gamma           = 0.995,
        gae_lambda      = 0.95,
        device          = "auto",
    )

    # ── Transplant weights ───────────────────────────────────────────────────
    # The observation space (27 inputs) and action space (2 outputs) are
    # identical between PeopleBotEnv and FineTuneBotEnv, so this is safe.
    model.policy.load_state_dict(pretrained_weights)
    print("  Weights transplanted successfully.\n")

    # Verify the transplant by checking one parameter tensor name
    first_key = list(pretrained_weights.keys())[0]
    print(f"  Sanity check — first param key: '{first_key}'  "
          f"shape: {pretrained_weights[first_key].shape}\n")

    # ── Callbacks ────────────────────────────────────────────────────────────
    curriculum_cb  = FineTuneCurriculumCallback()
    telemetry_cb   = TelemetryLoggerCallback()
    checkpoint_cb  = CheckpointCallback(
        save_freq   = max(1, 50_000 // N_ENVS),   # every ~50k env steps
        save_path   = FT_MODEL_DIR,
        name_prefix = "BEANS_FineTuned",
    )
    callback_list = CallbackList([checkpoint_cb, telemetry_cb, curriculum_cb])

    # ── Train ─────────────────────────────────────────────────────────────────
    try:
        model.learn(
            total_timesteps      = FT_TIMESTEPS,
            callback             = callback_list,
            tb_log_name          = "PPO_FineTune_Run",
            reset_num_timesteps  = True,   # fresh step counter for this run
            progress_bar         = True,
        )
        save_path = os.path.join(FT_MODEL_DIR, "BEANS_FineTuned_Final")
        model.save(save_path)
        print(f"\nFinal model saved → {save_path}")

    except KeyboardInterrupt:
        save_path = os.path.join(FT_MODEL_DIR, "BEANS_FineTuned_Interrupted")
        model.save(save_path)
        print(f"\nInterrupted — model saved → {save_path}")

    env.close()


if __name__ == "__main__":
    main()
