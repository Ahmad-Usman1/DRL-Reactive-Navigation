"""



"""

import os
import math
import numpy as np
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

from PeopleBotEnv_V3 import PeopleBotEnv
from FineTuneMaps_Phase2 import FineTuneMapBank

# ─── Load from the Phase 2 final checkpoint ──────────────────────────────────
PRETRAINED_MODEL_PATH = "finetune_models_v3_phase2\\BEANS_V3_Phase2_Final"

FT_LOG_DIR   = "./finetune_tensorboard_v3/"
FT_MODEL_DIR = "./finetune_models_v3_phase2b/"
FT_TIMESTEPS = 6_000_000   
N_ENVS       = 16

os.makedirs(FT_LOG_DIR,   exist_ok=True)
os.makedirs(FT_MODEL_DIR, exist_ok=True)


class FineTuneBotEnv_V3(PeopleBotEnv):

    def __init__(self):
        super().__init__()

        self.ft_map_bank = FineTuneMapBank(
            standard_map_dir="training_maps",
            standard_difficulty=0.75
        )
        self.waypoint_radius = 1.5
        self.goal_radius     = 0.9

        self.steps_since_checkpoint   = 0
        self.ft_previous_distance     = 0.0
        self._last_requested_action   = np.zeros(2, dtype=np.float32)

        # Diagonal rays: 15°–30° band, both sides (sensor order per PeopleBotEnv_V3)
        # Indices 2,3,4,5 = 30°,25°,20°,15° (left)
        # Indices 11,12,13,14 = -15°,-20°,-25°,-30° (right)
        self._diag_idx = np.array([2, 3, 4, 5, 11, 12, 13, 14], dtype=np.int64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.map_grid, wps, self.resolution = self.ft_map_bank.get_random_map()
        if not isinstance(wps, np.ndarray):
            wps = np.array(wps, dtype=np.float32)
        self.waypoints         = wps
        self.total_checkpoints = max(1, len(self.waypoints) - 1)

        start_pt = self.waypoints[0]
        if len(self.waypoints) > 1:
            self.current_goal_index = 1
            self.current_goal       = np.array(self.waypoints[1], dtype=np.float32)
            start_theta = math.atan2(self.current_goal[1] - start_pt[1],
                                     self.current_goal[0] - start_pt[0])
        else:
            self.current_goal_index = 0
            self.current_goal       = np.array(start_pt, dtype=np.float32)
            start_theta             = 0.0

        self.current_pose      = np.array([start_pt[0], start_pt[1], start_theta])
        self.current_lin_vel   = 0.0
        self.current_ang_vel   = 0.0
        self.previous_distance = float(np.linalg.norm(
            self.current_pose[:2] - self.current_goal))
        self.current_step             = 0
        self.steps_since_checkpoint   = 0
        self.ft_previous_distance     = self.previous_distance

        return self._get_obs(), {}

    def step(self, action):


        prev_goal_index = self.current_goal_index
        ft_prev_dist    = self.ft_previous_distance

        obs, _parent_reward, terminated, truncated, info = super().step(action)

        hit_checkpoint = self.current_goal_index > prev_goal_index
        tel            = info.get("telemetry", {})
        hit_final_goal = terminated and tel.get("rate_success", 0.0) == 1.0
        is_crashed     = terminated and tel.get("rate_crash",   0.0) == 1.0

        scan_data     = obs[:self.num_rays] * self.max_sensor_range
        heading_error = obs[self.num_rays + 1] * np.pi

        current_dist = float(np.linalg.norm(self.current_pose[:2] - self.current_goal))
        dist_improvement = ft_prev_dist if hit_checkpoint else ft_prev_dist - current_dist
        self.ft_previous_distance = current_dist

        reward = self._compute_ft_reward(
            scan_data=scan_data,
            heading_error=heading_error,
            dist_improvement=dist_improvement,
            hit_checkpoint=hit_checkpoint,
            hit_final_goal=hit_final_goal,
            is_crashed=is_crashed,
            truncated=truncated,
        )
        return obs, reward, terminated, truncated, info

    def _compute_ft_reward(self, scan_data, heading_error, dist_improvement,
                           hit_checkpoint, hit_final_goal, is_crashed, truncated):
        """
        V1 weighted-blend reward + V3 diagonal corner fix + angular jerk penalty.

        All terms except the jerk penalty are IDENTICAL to the successful
        91%-run. Do not modify them during Phase 1 — the single-variable change
        isolates the jerk penalty's effect on vibration_events.
        """

        front_dist = float(np.min(scan_data[self.front_indices]))
        side_dist  = float(np.min(scan_data[self.side_indices]))
        diag_dist  = float(np.min(scan_data[self._diag_idx]))
        min_dist   = min(front_dist, side_dist)

        effective_front = min(front_dist, diag_dist * 0.85)

        # 1. Existence tax
        reward = -0.05

        # 2. Progress
        reward += 10.0 * dist_improvement

        # 3. Velocity budget — V1 weighted blend
        SPEED_SCALE_DIST     = 3.0
        SIDE_TIGHTNESS_SCALE = 2.0
        SIDE_WEIGHT          = 0.45

        front_ratio  = float(np.clip(effective_front / SPEED_SCALE_DIST,     0.0, 1.0))
        side_ratio   = float(np.clip(side_dist       / SIDE_TIGHTNESS_SCALE, 0.0, 1.0))
        budget_ratio = front_ratio * (1.0 - SIDE_WEIGHT) + side_ratio * SIDE_WEIGHT

        velocity_excess = max(0.0, self.current_lin_vel - budget_ratio * self.max_lin_vel)
        if velocity_excess > 0.01:
            reward -= 20.0 * (velocity_excess ** 2)

        # 4. TTC backstop
        if self.current_lin_vel > 0.05:
            ttc = effective_front / self.current_lin_vel
            if ttc < 1.5:
                reward -= 4.0 * ((1.5 / ttc - 1.0) ** 2)

        # 5. Heading penalty with grace period
        grace_factor = min(1.0, self.steps_since_checkpoint / 15.0)
        reward -= 1.5 * (1.0 - math.cos(heading_error)) * budget_ratio * grace_factor

        # 6. Narrow passage progress bonus
        if min_dist < 0.8 and dist_improvement > 0.001:
            tightness = (0.8 - min_dist) / 0.8
            reward += 2.5 * tightness * dist_improvement

        # 8. Terminal rewards
        if hit_checkpoint:
            reward += 10.0 + 25.0 * max(0.0, math.cos(heading_error))

        if is_crashed:
            reward -= 500.0
        elif hit_final_goal:
            reward += 500.0

        if truncated:
            reward -= 150.0

        return float(reward)

    def set_difficulty(self, difficulty_level):
        self.ft_map_bank.set_difficulty(difficulty_level)

    def set_curriculum_stage(self, stage: int):
        self.ft_map_bank.set_curriculum_stage(stage)


# ─── Callbacks ──────────────────────────────────────────────────────────────

class FineTuneCurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.success_history   = deque(maxlen=100)
        self.crash_history     = deque(maxlen=100)
        self.timeout_history   = deque(maxlen=100)
        self.vibration_history = deque(maxlen=100)

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [])[i]
                if "telemetry" in info:
                    tel = info["telemetry"]
                    self.success_history.append(tel.get("rate_success", 0.0))
                    self.crash_history.append(tel.get("rate_crash",     0.0))
                    self.timeout_history.append(tel.get("rate_timeout", 0.0))
                    self.vibration_history.append(tel.get("vibration_events", 0.0))

        if self.success_history:
            self.logger.record("finetune/rolling_success_rate",
                               float(np.mean(self.success_history)))
        if self.crash_history:
            self.logger.record("finetune/rolling_crash_rate",
                               float(np.mean(self.crash_history)))
        if self.timeout_history:
            self.logger.record("finetune/rolling_timeout_rate",
                               float(np.mean(self.timeout_history)))
        if self.vibration_history:
            self.logger.record("finetune/rolling_vibration_events",
                               float(np.mean(self.vibration_history)))
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


class CurriculumStageCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose=1):
        super().__init__(verbose)
        self.stage2_start = int(0.50 * total_timesteps)
        self.current_stage = 1

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.stage2_start and self.current_stage < 2:
            self.current_stage = 2
            self.training_env.env_method("set_curriculum_stage", 2)
            if self.verbose: print(f"[Curriculum] → Stage 2 at {self.num_timesteps}")
        self.logger.record("curriculum/stage", self.current_stage)
        return True


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def make_finetune_env():
    return Monitor(FineTuneBotEnv_V3())


def main():
    print("=" * 60)
    print(" BEANS V3 Finetuning — PHASE 2b")
    print(f" Pretrained : {PRETRAINED_MODEL_PATH}")
    print(f" Steps      : {FT_TIMESTEPS:,}")
    print(f" Envs       : {N_ENVS}")
    print("=" * 60)

    env = make_vec_env(make_finetune_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    print("\nLoading pretrained weights...")
    temp_model = PPO.load(PRETRAINED_MODEL_PATH, device="auto")
    pretrained_weights = temp_model.policy.state_dict()
    print(f"  {len(pretrained_weights)} tensors loaded.")
    del temp_model

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128])
    )

    model = PPO(
        "MlpPolicy", env,
        policy_kwargs   = policy_kwargs,
        verbose         = 1,
        tensorboard_log = FT_LOG_DIR,

        learning_rate   = linear_schedule(8e-5),
        n_epochs        = 8,
        ent_coef        = 0.01,
        clip_range      = 0.1,

        n_steps         = 4096,
        batch_size      = 64,
        gamma           = 0.995,
        gae_lambda      = 0.95,
        device          = "auto",
    )

    model.policy.load_state_dict(pretrained_weights)
    print("  Weights transplanted.\n")

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(1, 50_000 // N_ENVS),
        save_path   = FT_MODEL_DIR,
        name_prefix = "BEANS_V3_Phase2b",
    )

    callback_list = CallbackList([
        checkpoint_cb, 
        TelemetryLoggerCallback(), 
        FineTuneCurriculumCallback(), 
        CurriculumStageCallback(FT_TIMESTEPS, verbose=1)
    ])

    try:
        model.learn(
            total_timesteps     = FT_TIMESTEPS,
            callback            = callback_list,
            tb_log_name         = "PPO_FineTune_V3_Phase2b",
            reset_num_timesteps = True,
            progress_bar        = True,
        )
        save_path = os.path.join(FT_MODEL_DIR, "BEANS_V3_Phase2b_Final")
        model.save(save_path)
        print(f"\nFinal model saved → {save_path}")
    except KeyboardInterrupt:
        save_path = os.path.join(FT_MODEL_DIR, "BEANS_V3_Phase2b_Interrupted")
        model.save(save_path)
        print(f"\nInterrupted → {save_path}")

    env.close()


if __name__ == "__main__":
    main()
