"""
TrainV5.py
==========
BEANS V5 Full Training Pipeline — Single Massive Run with Automatic Curriculum

WHY A FULL RETRAIN (not fine-tune from V3)
──────────────────────────────────────────
1. Observation space: 25-dim (V3) → 26-dim (V5)
   Added jerk_integrator state at [21], action FIFO expanded to 4 elements.
   First policy layer changes from (25,256) → (26,256). No valid weight reuse.

2. Physics semantics shift:
   V3 had dual sonar/camera FIFOs with different delays per ray index.
   V5 has a unified 1-step obs FIFO — every scan ray is delayed identically.
   Fine-tuning V3 weights would map wrong ray delays to wrong hidden features.

3. Domain randomisation added (τ_v, τ_w, positional slip):
   V3 policy learned a fixed-τ world. The gradient signal for variable inertia
   is fundamentally different — the value function must now account for
   unknown inertia, requiring a different internal representation.

HYPERPARAMETER RATIONALE
──────────────────────────
  n_steps   = 4096
    Long rollouts critical for delayed credit assignment. Waypoint bonuses
    (+10 to +35) for actions taken 5–10 steps before reaching the waypoint
    need a sufficient return window to backpropagate correctly.
    At 16 envs: 65,536 samples per PPO update — dense gradient signal.

  batch_size = 256
    65,536 / 256 = 256 minibatches per update × 10 epochs = 2,560 gradient
    steps per PPO iteration. Appropriate for a 26→256→256→128→2 network.
    V1/V2 used batch_size=64 — too small for the expanded architecture.

  gamma = 0.995
    Effective horizon = 1/(1−0.995) = 200 steps = 20 real seconds at 10Hz.
    Correct for a 40m×40m map at 0.4m/s. Do not change.

  gae_lambda = 0.95
    Standard value. Balances variance (lambda→1) and bias (lambda→0).
    Higher lambda preferred for sparse reward environments (waypoint bonuses
    are sparse) to reduce bias in advantage estimates.

  ent_coef = 0.010
    Midpoint between V1 (0.005) and V3 (0.015). Domain randomisation
    (varying τ_v, τ_w) provides natural exploration diversity — we don't
    need as high entropy as V3 which had less internal variation.
    Annealed to ~0 by end of training via linear LR schedule.

  learning_rate = linear_schedule(3e-4)
    Proven starting point. Decays to 0 over full TOTAL_TIMESTEPS.
    If curriculum stalls at a hard tier, the LR has already decayed —
    consider adding a LR warmup on tier transition if needed (not implemented
    here to keep the script simple).

  clip_range = 0.2
    Standard PPO clip. No reason to change — validated across all prior runs.

NETWORK ARCHITECTURE
──────────────────────
  [256, 256, 128] + Tanh for both policy and value networks.

  WHY TANH (not ELU/ReLU):
    Tanh saturates at ±1, providing implicit output bounding. For this specific
    task, inputs are pre-normalised to [−1, 1], and Tanh preserves this range
    through hidden layers. This prevents gradient explosion that ReLU would
    allow on extreme LiDAR readings near walls. V1's success with Tanh on hard
    maps confirms this is the right choice for this observation distribution.

  WHY [256, 256, 128] (not V1's [128, 128, 128]):
    26 inputs vs 27 in V1. The extra width at layers 1–2 accommodates:
      - Jerk integrator state (requires separate feature dimension)
      - Larger action FIFO (4 values vs 6 in V1, but semantically denser)
      - Domain-randomised dynamics (τ_v, τ_w uncertainty = more features needed)
    Layer 3 narrows to 128 to compress before the 2-action output head.
    This is NOT a gratuitous architecture increase — it's justified by the
    additional input semantics.

CURRICULUM DESIGN
───────────────────
  Tiers: [0.0, 0.1, 0.25, FINETUNE, 0.5, 0.75, 1.0]
  Gate:  90% rolling success over 100 episodes

  Efficiency of env_method calls:
    V1/V3 called env_method("set_difficulty", val) EVERY step across all envs.
    With SubprocVecEnv and N_ENVS=16, that is 16 inter-process pipe messages
    per inference step, ~160,000 IPC calls per second at 10K env-steps/sec.
    V5 calls env_method ONLY on tier transitions (once per tier advancement).
    This eliminates essentially all curriculum-related IPC overhead.

TOTAL TIMESTEPS ESTIMATE
─────────────────────────
  Tier 0.0  (open + pillars)    :   500K  steps  — nearly trivial
  Tier 0.1  (dense pillars)     : 1,500K  steps  — first real obstacles
  Tier 0.25 (corridors)         : 2,500K  steps  — first narrow passages
  Tier FINE (curated maps)      : 2,000K  steps  — targeted skill acquisition
  Tier 0.5  (medium density)    : 3,000K  steps  — robust obstacle avoidance
  Tier 0.75 (high density)      : 2,500K  steps  — near-expert navigation
  Tier 1.0  (maximum density)   : 3,000K  steps  — final robustness
                                  ───────────────
  Total                         : 15,000K steps  (15M)

  At ~10K–15K env-steps/sec with 16 SubprocVecEnv workers:
    Lower bound: 15M / 15K = 1,000s ≈ 17 minutes
    Upper bound: 15M /  8K = 1,875s ≈ 31 minutes
  This is a conservative budget — most runs will master tiers 0.0–0.25 faster.

OUTPUT LOCATIONS
─────────────────
  Models      → ./models_beans_v5/
  TensorBoard → ./tensorboard_beans_v5/

Run TensorBoard:
  tensorboard --logdir ./tensorboard_beans_v5

Key TensorBoard tags to monitor:
  curriculum/tier_index            — which tier we are on (0–6)
  curriculum/rolling_success_rate  — must hit 0.90 to advance
  curriculum/map_difficulty        — current difficulty value (or -1 for finetune)
  telemetry/path_efficiency        — target > 0.85 (bot taking near-ideal paths)
  telemetry/traversed_vs_ideal     — target < 1.3 (no more than 30% detour)
  telemetry/vibration_events       — target < 5 per episode (jerk penalty working)
  telemetry/avg_wall_clearance     — should NOT drop below 0.4m on hard maps
"""

import os
import time
import numpy as np
import torch.nn as nn
from typing import Callable, List
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from PeopleBotEnvV5 import PeopleBotEnv, MapBankV5

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these, touch nothing else
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR       = "./models_beans_v5/"
LOG_DIR         = "./tensorboard_beans_v5/"
TOTAL_TIMESTEPS = 15_000_000
N_ENVS          = 16   # tuned for i3-1215U (2P + 4E cores + hyperthreading)
DATASET_DIR     = "training_maps"
FINETUNE_DIR    = "finetune_maps"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING RATE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear decay: initial_value → 0 over the full training run.
    At progress_remaining=1.0 (start): LR = initial_value
    At progress_remaining=0.0 (end):   LR = 0
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class CompetenceCurriculumCallback(BaseCallback):
    """
    Advances map difficulty when the rolling 100-episode success rate hits 90%.

    TIER STRUCTURE:
      Index  Type       Value    Description
      ─────  ─────────  ───────  ──────────────────────────────────────────
        0    density    0.00     Open room, 2–4 pillars
        1    density    0.10     Open room, 8–12 pillars — first obstacle field
        2    density    0.25     Corridors, medium obstacle density
        3    finetune  -1.00     Curated maps (from finetune_maps/) — targeted skills
        4    density    0.50     Medium density — robust avoidance required
        5    density    0.75     High density — expert navigation
        6    density    1.00     Maximum density — final robustness

    FINETUNE SENTINEL:
      Difficulty=-1.0 is MapBankV5.FINETUNE_SENTINEL. When set, the env
      draws from the curated finetune_maps/ pool instead of procedural maps.

    IPC EFFICIENCY:
      env_method("set_difficulty", val) is called ONLY when the tier index
      changes. Not every step (as in V1/V3). This eliminates ~99.9% of
      curriculum-related inter-process communication overhead in SubprocVecEnv.

    PROMOTION WINDOW:
      After a tier promotion, success_history is cleared and episodes_at_tier
      is reset to 0. The callback requires a minimum of 100 NEW episodes at
      the new tier before evaluating promotion again. This prevents the agent
      from coasting on a lucky streak in a few episodes.
    """

    TIERS: List[dict] = [
        {"label": "Open Room",         "type": "density",  "value": 0.00},
        {"label": "Pillar Field",      "type": "density",  "value": 0.10},
        {"label": "Corridors",         "type": "density",  "value": 0.25},
        {"label": "Finetune Maps",     "type": "finetune", "value": -1.00},
        {"label": "Med Density",       "type": "density",  "value": 0.50},
        {"label": "High Density",      "type": "density",  "value": 0.75},
        {"label": "Max Density",       "type": "density",  "value": 1.00},
    ]
    PROMOTION_THRESHOLD = 0.90
    WINDOW_SIZE         = 100    # rolling episode window for success rate

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_tier_idx   = 0
        self.success_history    = deque(maxlen=self.WINDOW_SIZE)
        self.episodes_at_tier   = 0
        self._last_set_tier_idx = -1   # forces initial env_method call

        # Timing metrics
        self.tier_start_wall    = time.time()
        self.tier_start_steps   = 0

    @property
    def _current_tier(self) -> dict:
        return self.TIERS[self.current_tier_idx]

    def _on_step(self) -> bool:
        # ── Broadcast difficulty ONLY on tier change (eliminates IPC overhead) ─
        if self.current_tier_idx != self._last_set_tier_idx:
            self.training_env.env_method(
                "set_difficulty", self._current_tier["value"]
            )
            self._last_set_tier_idx = self.current_tier_idx

        # ── Collect episode outcomes from all parallel envs ───────────────────
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [])[i]
                if "telemetry" in info:
                    self.success_history.append(info["telemetry"]["rate_success"])
                    self.episodes_at_tier += 1

        # ── Promotion check ───────────────────────────────────────────────────
        if (len(self.success_history) >= self.WINDOW_SIZE and
                self.episodes_at_tier  >= self.WINDOW_SIZE):
            avg_success = float(np.mean(self.success_history))

            if (avg_success >= self.PROMOTION_THRESHOLD and
                    self.current_tier_idx < len(self.TIERS) - 1):

                # Compute time spent at this tier for the log
                elapsed_s   = time.time() - self.tier_start_wall
                elapsed_ep  = self.episodes_at_tier
                elapsed_ts  = self.num_timesteps - self.tier_start_steps

                prev_tier   = self.TIERS[self.current_tier_idx]
                self.current_tier_idx  += 1
                next_tier   = self.TIERS[self.current_tier_idx]

                # Reset window for fresh evaluation at new tier
                self.success_history.clear()
                self.episodes_at_tier  = 0
                self.tier_start_wall   = time.time()
                self.tier_start_steps  = self.num_timesteps

                print(
                    f"\n{'='*65}\n"
                    f"[V5 CURRICULUM] TIER PROMOTED\n"
                    f"  From : [{prev_tier['type']:>8}] {prev_tier['label']}"
                    f"  (diff={prev_tier['value']:.2f})\n"
                    f"  To   : [{next_tier['type']:>8}] {next_tier['label']}"
                    f"  (diff={next_tier['value']:.2f})\n"
                    f"  Stats: {elapsed_ep} episodes | {elapsed_ts:,} steps"
                    f" | {elapsed_s:.0f}s elapsed\n"
                    f"{'='*65}\n"
                )

        # ── TensorBoard logging (every step) ─────────────────────────────────
        tier = self._current_tier
        self.logger.record("curriculum/tier_index",
                           float(self.current_tier_idx))
        self.logger.record("curriculum/map_difficulty",
                           float(tier["value"]))
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
    Streams all PeopleBotEnv_V5 episode telemetry to TensorBoard.

    TensorBoard tags logged (per episode, averaged across parallel envs):
      telemetry/rate_success            — primary success metric
      telemetry/rate_crash              — collision or tipping events
      telemetry/rate_timeout            — step budget exhausted
      telemetry/avg_velocity            — mean lin_vel over episode
      telemetry/avg_wall_clearance      — mean min(front, side) lidar
      telemetry/checkpoint_capture_rate — fraction of waypoints reached
      telemetry/path_efficiency         — L* / max(L_actual, L*) ∈ (0, 1]
      telemetry/traversed_vs_ideal      — L_actual / L*  (1.0 = perfect)
      telemetry/vibration_events        — rapid action reversals per episode
      telemetry/jerk_integrator_peak    — peak J_t value per episode
      telemetry/tau_v                   — linear inertia this episode
      telemetry/tau_w                   — angular inertia this episode
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
# PHYSICS SANITY CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class PhysicsSanityCallback(BaseCallback):
    """
    Permanently stamps the V5 physical configuration into TensorBoard at
    training start and every 100K steps. Makes it impossible to confuse
    V5 checkpoints with earlier versions when reviewing runs later.

    Tags:
      physics/obs_delay_ms          — 100ms
      physics/actuator_delay_ms     — 200ms
      physics/jerk_lambda           — 0.85
      physics/jerk_k                — 0.02
      physics/j_max                 — 13.33
      physics/speed_scale_m         — 2.0
      physics/tau_v_min_s           — 0.50
      physics/tau_v_max_s           — 1.15
      physics/tau_w_min_s           — 0.03
      physics/tau_w_max_s           — 0.10
      physics/k_slip_pos            — 0.05
      physics/obs_dims              — 26
    """

    def __init__(self, env_ref: PeopleBotEnv, log_every: int = 100_000, verbose=0):
        super().__init__(verbose)
        self.env_ref   = env_ref
        self.log_every = log_every

    def _on_step(self) -> bool:
        if self.n_calls == 1 or self.n_calls % self.log_every == 0:
            e = self.env_ref
            self.logger.record("physics/obs_delay_ms",
                               float(e.OBS_DELAY_STEPS    * e.dt * 1000))
            self.logger.record("physics/actuator_delay_ms",
                               float(e.ACTUATOR_LAG_STEPS * e.dt * 1000))
            self.logger.record("physics/jerk_lambda",  float(e.JERK_LAMBDA))
            self.logger.record("physics/jerk_k",       float(e.JERK_K))
            self.logger.record("physics/j_max",        float(e.J_MAX))
            self.logger.record("physics/speed_scale_m",float(e.SPEED_SCALE))
            self.logger.record("physics/tau_v_min_s",  0.50)
            self.logger.record("physics/tau_v_max_s",  1.15)
            self.logger.record("physics/tau_w_min_s",  0.03)
            self.logger.record("physics/tau_w_max_s",  0.10)
            self.logger.record("physics/k_slip_pos",   float(e.K_SLIP_POS))
            self.logger.record("physics/obs_dims",
                               float(e.observation_space.shape[0]))
        return True


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_env():
    """
    Named factory function required for SubprocVecEnv pickling.
    Lambda functions cannot be pickled across process boundaries.
    """
    return Monitor(PeopleBotEnv(dataset_dir=DATASET_DIR, finetune_dir=FINETUNE_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Startup banner ────────────────────────────────────────────────────────
    print("=" * 65)
    print(" BEANS V5 Training Pipeline")
    print(" Domain Randomisation | Leaky Jerk | Path Efficiency | 26-dim obs")
    print(f" Output models    : {MODEL_DIR}")
    print(f" TensorBoard logs : {LOG_DIR}")
    print(f" Parallel envs    : {N_ENVS}")
    print(f" Total timesteps  : {TOTAL_TIMESTEPS:,}")
    print("=" * 65)

    # Reference env for config logging (not used for training)
    _ref = PeopleBotEnv(dataset_dir=DATASET_DIR, finetune_dir=FINETUNE_DIR)

    print("\n── Delay Model ──────────────────────────────────────────────")
    print(f"  Obs delay      : {_ref.OBS_DELAY_STEPS} step  "
          f"= {_ref.OBS_DELAY_STEPS * _ref.dt * 1000:.0f}ms  (unified sensor FIFO)")
    print(f"  Actuator delay : {_ref.ACTUATOR_LAG_STEPS} steps "
          f"= {_ref.ACTUATOR_LAG_STEPS * _ref.dt * 1000:.0f}ms  (command → motor)")
    print(f"\n── Domain Randomisation ─────────────────────────────────────")
    print(f"  τ_v ~ U[0.50, 1.15]s  (linear  inertia per episode)")
    print(f"  τ_w ~ U[0.03, 0.10]s  (angular inertia per episode)")
    print(f"  K_slip_pos = {_ref.K_SLIP_POS}  (positional drift coefficient)")
    print(f"\n── Leaky Integrator ─────────────────────────────────────────")
    print(f"  λ = {_ref.JERK_LAMBDA}  k_j = {_ref.JERK_K}  J_max = {_ref.J_MAX:.2f}")
    print(f"  Sustained oscillation cost: {_ref.JERK_K * _ref.J_MAX**2:.2f}/step")
    print(f"\n── Reward Constants ─────────────────────────────────────────")
    print(f"  Progress scale   : {_ref.PROGRESS_SCALE}")
    print(f"  Speed scale      : {_ref.SPEED_SCALE}m")
    print(f"  Crash / Success  : {_ref.CRASH_PENALTY} / +{_ref.SUCCESS_REWARD}")
    print(f"  Eff bonus (max)  : +{_ref.EFF_SCALE}")
    print(f"  Obs dims         : {_ref.observation_space.shape[0]}")
    print(f"  Max steps/ep     : {_ref.max_steps}")
    print("─" * 65 + "\n")

    # ── Curriculum tier summary ───────────────────────────────────────────────
    print("── Curriculum Tiers ─────────────────────────────────────────")
    for i, tier in enumerate(CompetenceCurriculumCallback.TIERS):
        print(f"  [{i}] {tier['label']:<20} type={tier['type']:<8} "
              f"diff={tier['value']:>5.2f}")
    print(f"  Gate: {CompetenceCurriculumCallback.PROMOTION_THRESHOLD*100:.0f}% "
          f"rolling success over {CompetenceCurriculumCallback.WINDOW_SIZE} episodes")
    print("─" * 65 + "\n")

    # ── Build vectorised environment ──────────────────────────────────────────
    env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    # ── Policy architecture ───────────────────────────────────────────────────
    # [256, 256, 128] + Tanh — see module docstring for full rationale.
    # Input: 26-dim obs → 256 → 256 → 128 → [policy head: 2, value head: 1]
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(
            pi=[256, 256, 128],
            vf=[256, 256, 128]
        )
    )

    # ── PPO model ────────────────────────────────────────────────────────────
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs   = policy_kwargs,
        verbose         = 1,
        tensorboard_log = LOG_DIR,

        # ── Core hyperparameters (see module docstring for rationale) ──────
        learning_rate   = linear_schedule(3e-4),
        n_steps         = 4096,      # steps per env per update → 65,536 total
        batch_size      = 256,       # 65,536 / 256 = 256 minibatches × 10 epochs
        n_epochs        = 10,
        gamma           = 0.995,     # 200-step effective horizon = 20s at 10Hz
        gae_lambda      = 0.95,
        ent_coef        = 0.010,     # entropy bonus — annealed by LR schedule
        clip_range      = 0.2,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,

        device          = "auto",    # uses CUDA if available, else CPU
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    curriculum_cb = CompetenceCurriculumCallback(verbose=1)
    telemetry_cb  = TelemetryLoggerCallback()
    physics_cb    = PhysicsSanityCallback(env_ref=_ref, log_every=100_000)

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(1, 100_000 // N_ENVS),   # ~100K env steps between saves
        save_path   = MODEL_DIR,
        name_prefix = "BEANS_V5",                  # distinct from all prior runs
    )

    callback_list = CallbackList([
        checkpoint_cb,
        telemetry_cb,
        curriculum_cb,
        physics_cb,
    ])

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"[V5] Training start — {TOTAL_TIMESTEPS:,} total timesteps\n")
    train_start = time.time()

    try:
        model.learn(
            total_timesteps     = TOTAL_TIMESTEPS,
            callback            = callback_list,
            tb_log_name         = "BEANS_V5_Run",
            reset_num_timesteps = True,
            progress_bar        = True,
        )
        elapsed = time.time() - train_start
        save_path = os.path.join(MODEL_DIR, "BEANS_V5_Final")
        model.save(save_path)
        print(f"\n[V5] Training complete in {elapsed/60:.1f} minutes.")
        print(f"[V5] Final model saved → {save_path}.zip")

    except KeyboardInterrupt:
        elapsed = time.time() - train_start
        save_path = os.path.join(MODEL_DIR, "BEANS_V5_Interrupted")
        model.save(save_path)
        print(f"\n[V5] Interrupted after {elapsed/60:.1f} minutes.")
        print(f"[V5] Checkpoint saved → {save_path}.zip")

    finally:
        env.close()
        print("[V5] Environments closed cleanly.")


if __name__ == "__main__":
    main()