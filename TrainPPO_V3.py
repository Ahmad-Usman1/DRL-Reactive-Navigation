"""
TrainPPO_V3.py
==============
Full retraining script for the BEANS V3 navigation stack.

WHY A FULL RETRAIN (not fine-tune)
────────────────────────────────────
V3 introduces two breaking changes that make V1/V2 weights incompatible:

  1. Observation space: 27-dim → 25-dim
     The first policy layer changes from (27,128) → (25,128).
     There is no mathematically valid way to trim a trained weight matrix
     without destroying the learned feature representations.

  2. Physics semantics: every obs element points to a different timestep.
     V1/V2 action history covered 300ms (3 steps). V3 covers 100ms (1 step).
     V1/V2 scan was instantaneous. V3 scan is a merged delayed reading.
     Fine-tuning across this change would invert the network's existing causal
     model of time — gradients would point in contradictory directions.

HYPERPARAMETER RATIONALE
──────────────────────────
  n_steps   4096 (was 2048)
    With simpler delay physics, each rollout contains denser gradient signal.
    Longer rollouts give the critic a better value baseline before each update,
    which is important because anticipatory braking decisions that pay off
    5-10 steps later need a long enough return window to be credited correctly.

  gamma     0.995 (unchanged)
    Effective horizon = 1/(1-0.995) = 200 steps = 20 real seconds at 10Hz.
    This is correct for a 40m×40m map at 0.4 m/s. Do not change.

  ent_coef  0.015 (was 0.005)
    Higher entropy early prevents premature collapse to the high-speed policy
    inherited from the old physics. Annealed back by the linear LR schedule.

  learning_rate  linear_schedule(3e-4) (unchanged)
    Clean scratch start — no reason to be conservative with learning rate.
    The schedule decays naturally as training progresses.

  batch_size 64 (unchanged)
    Validated empirically across prior runs.

  n_epochs   10 (unchanged)

OUTPUT LOCATIONS
─────────────────
  Models     → ./models_beans_v3/
  TensorBoard → ./tensorboard_beans_v3/

Run TensorBoard with:
  tensorboard --logdir ./tensorboard_beans_v3
"""

import os
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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Distinct naming — cannot be confused with V1/V2 checkpoints
MODEL_DIR       = "./models_beans_v3/"
LOG_DIR         = "./tensorboard_beans_v3/"
TOTAL_TIMESTEPS = 8_000_000   # V3 starts from scratch; extra budget vs V2 finetune
N_ENVS          = 16

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING RATE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Decays linearly from initial_value → 0 over the full training run."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class CompetenceCurriculumCallback(BaseCallback):
    """
    Promotes map difficulty when rolling 100-episode success rate hits 90%.

    Tiers: 0.0 → 0.10 → 0.25 → 0.50 → 0.75 → 1.0
    The 0.10 tier (sparse open rooms with pillars) is critical for teaching
    the new velocity derivative signals before corridors are introduced.

    TensorBoard tags logged every step:
      curriculum/map_difficulty        — current tier value
      curriculum/rolling_success_rate  — 100-ep rolling mean
      curriculum/episodes_at_tier      — episode count at current tier
      curriculum/tier_index            — integer tier index
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.difficulty_tiers       = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
        self.current_tier_idx       = 0
        self.success_history        = deque(maxlen=100)
        self.episodes_at_tier       = 0

    def _on_step(self) -> bool:
        current_diff = self.difficulty_tiers[self.current_tier_idx]
        self.training_env.env_method("set_difficulty", current_diff)

        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [])[i]
                if "telemetry" in info:
                    self.success_history.append(
                        info["telemetry"]["rate_success"]
                    )
                    self.episodes_at_tier += 1

                    if (len(self.success_history) == 100
                            and self.episodes_at_tier >= 100):
                        avg_success = float(np.mean(self.success_history))

                        if (avg_success >= 0.90
                                and self.current_tier_idx < len(self.difficulty_tiers) - 1):
                            prev = self.difficulty_tiers[self.current_tier_idx]
                            self.current_tier_idx  += 1
                            self.success_history.clear()
                            self.episodes_at_tier   = 0
                            next_diff = self.difficulty_tiers[self.current_tier_idx]
                            print(
                                f"\n[V3 CURRICULUM] Mastered diff={prev:.2f} "
                                f"→ advancing to diff={next_diff:.2f}\n"
                            )

        # TensorBoard — always log current state
        self.logger.record("curriculum/map_difficulty",
                           float(self.difficulty_tiers[self.current_tier_idx]))
        self.logger.record("curriculum/tier_index",
                           float(self.current_tier_idx))
        self.logger.record("curriculum/episodes_at_tier",
                           float(self.episodes_at_tier))
        if self.success_history:
            self.logger.record("curriculum/rolling_success_rate",
                               float(np.mean(self.success_history)))

        return True


# ─────────────────────────────────────────────────────────────────────────────
# TELEMETRY CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryLoggerCallback(BaseCallback):
    """
    Logs all episode telemetry from PeopleBotEnv_V3 to TensorBoard.

    Tags:
      telemetry/avg_velocity            — mean lin_vel over episode
      telemetry/avg_wall_clearance      — mean min(front, side) lidar dist
      telemetry/vibration_events        — rapid action reversals per episode
      telemetry/checkpoint_capture_rate — fraction of waypoints reached
      telemetry/rate_success            — 1.0 = reached final goal
      telemetry/rate_crash              — 1.0 = collision or tip
      telemetry/rate_timeout            — 1.0 = step budget exhausted
    """

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [])[i]
                if "telemetry" in info:
                    for key, value in info["telemetry"].items():
                        self.logger.record(f"telemetry/{key}", float(value))
        return True


# ─────────────────────────────────────────────────────────────────────────────
# DELAY SANITY CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class DelayPhysicsSanityCallback(BaseCallback):
    """
    Logs the simulated pipeline latencies once at training start so that the
    TensorBoard run record permanently documents what delay model was used.

    This makes it impossible to confuse V3 checkpoints with V1/V2 ones when
    reviewing runs months later.

    Tags (logged once at step 1, then every 100k steps):
      delay_model/sonar_delay_ms
      delay_model/camera_delay_ms
      delay_model/actuator_delay_ms
      delay_model/tau_motor_ms
      delay_model/obs_dims
    """

    def __init__(self, env_instance: PeopleBotEnv, log_every: int = 100_000,
                 verbose=0):
        super().__init__(verbose)
        self.env_ref   = env_instance
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.n_calls == 1 or self.n_calls % self.log_every == 0:
            dt = self.env_ref.dt
            self.logger.record(
                "delay_model/sonar_delay_ms",
                self.env_ref.SONAR_DELAY_STEPS * dt * 1000
            )
            self.logger.record(
                "delay_model/camera_delay_ms",
                self.env_ref.CAMERA_DELAY_STEPS * dt * 1000
            )
            self.logger.record(
                "delay_model/actuator_delay_ms",
                self.env_ref.ACTUATOR_LAG_STEPS * dt * 1000
            )
            self.logger.record(
                "delay_model/tau_motor_ms",
                self.env_ref.tau_motor * 1000
            )
            self.logger.record(
                "delay_model/obs_dims",
                float(self.env_ref.observation_space.shape[0])
            )
        return True


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_env():
    """Named factory required for SubprocVecEnv pickling."""
    return Monitor(PeopleBotEnv())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(" BEANS V3 Training Pipeline")
    print(" Dual-delay physics | 25-dim obs | Full scratch retrain")
    print(f" Output models    : {MODEL_DIR}")
    print(f" TensorBoard logs : {LOG_DIR}")
    print(f" Parallel envs    : {N_ENVS}")
    print(f" Total timesteps  : {TOTAL_TIMESTEPS:,}")
    print("=" * 65)

    # Delay model summary printed at startup for the training log
    _ref = PeopleBotEnv()
    print("\n── Delay Model ──────────────────────────────────────────────")
    print(f"  Sonar delay   : {_ref.SONAR_DELAY_STEPS} step  "
          f"= {_ref.SONAR_DELAY_STEPS * _ref.dt * 1000:.0f}ms  "
          f"(serial read + robot Tx)")
    print(f"  Camera delay  : {_ref.CAMERA_DELAY_STEPS} steps "
          f"= {_ref.CAMERA_DELAY_STEPS * _ref.dt * 1000:.0f}ms  "
          f"(ONNX depth pipeline)")
    print(f"  Actuator delay: {_ref.ACTUATOR_LAG_STEPS} step  "
          f"= {_ref.ACTUATOR_LAG_STEPS * _ref.dt * 1000:.0f}ms  "
          f"(command Tx + robot receive)")
    print(f"  tau_motor     : {_ref.tau_motor * 1000:.0f}ms  "
          f"(electromechanical inertia only)")
    print(f"  Obs dims      : {_ref.observation_space.shape[0]}")
    print(f"  Max steps/ep  : {_ref.max_steps}")
    print("─" * 65 + "\n")

    # Build vectorised environment
    env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    # Policy architecture — 128×128×128 + Tanh, identical to V1/V2
    # Input layer auto-sized by SB3 from observation_space (25 → 128)
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128])
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs   = policy_kwargs,
        verbose         = 1,
        tensorboard_log = LOG_DIR,

        # ── Core hyperparameters ──────────────────────────────────────────
        learning_rate   = linear_schedule(3e-4),
        n_steps         = 4096,     # was 2048 — denser signal per rollout
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.995,    # 200-step horizon = 20s real time @ 10Hz
        gae_lambda      = 0.95,
        ent_coef        = 0.015,    # was 0.005 — more exploration for new physics
        clip_range      = 0.2,

        device          = "auto",
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    curriculum_cb = CompetenceCurriculumCallback(verbose=1)
    telemetry_cb  = TelemetryLoggerCallback()
    delay_cb      = DelayPhysicsSanityCallback(env_instance=_ref)

    checkpoint_cb = CheckpointCallback(
        # Save every ~100k env steps across all parallel envs
        save_freq   = max(1, 100_000 // N_ENVS),
        save_path   = MODEL_DIR,
        name_prefix = "BEANS_V3",          # distinct from V1 "BEANS_PPO_Adaptive"
                                            # and V2 "BEANS_FineTuned"
    )

    callback_list = CallbackList([
        checkpoint_cb,
        telemetry_cb,
        curriculum_cb,
        delay_cb,
    ])

    # ── Training ──────────────────────────────────────────────────────────────
    try:
        model.learn(
            total_timesteps     = TOTAL_TIMESTEPS,
            callback            = callback_list,
            tb_log_name         = "BEANS_V3_Run",
            reset_num_timesteps = True,
            progress_bar        = True,
        )
        save_path = os.path.join(MODEL_DIR, "BEANS_V3_Final")
        model.save(save_path)
        print(f"\n[V3] Final model saved → {save_path}.zip")

    except KeyboardInterrupt:
        save_path = os.path.join(MODEL_DIR, "BEANS_V3_Interrupted")
        model.save(save_path)
        print(f"\n[V3] Interrupted — checkpoint saved → {save_path}.zip")

    finally:
        env.close()
        print("[V3] Environment closed cleanly.")


if __name__ == "__main__":
    main()
