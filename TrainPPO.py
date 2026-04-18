import os
import torch.nn as nn
from typing import Callable
from collections import deque
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from PeopleBotEnv import PeopleBotEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

# --- CONFIGURATION ---
LOG_DIR = "./ppo_tensorboard/"
MODEL_DIR = "./saved_models/"
TOTAL_TIMESTEPS = 6_000_000
N_ENVS = 16  # Optimized for i3-1215U (2P + 4E Cores)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. THE COMPETENCE ARCHITECT (The Fixed Version) ---
class CompetenceCurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.difficulty_tiers = [0.0, 0.1, 0.25, 0.50, 0.75, 1.0] 
        self.current_tier_idx = 0
        self.success_history = deque(maxlen=100) # Rolling window of 100 episodes
        self.episodes_at_current_tier = 0

    def _on_step(self) -> bool:
        # Pushing difficulty down to the environments
        current_diff = self.difficulty_tiers[self.current_tier_idx]
        
        # We use env_method to broadcast the change to all parallel workers
        self.training_env.env_method("set_difficulty", current_diff) 

        # Track success from all parallel environments
        for i, done in enumerate(self.locals.get("dones")):
            if done:
                info = self.locals.get("infos")[i]
                if "telemetry" in info:
                    self.success_history.append(info["telemetry"]["rate_success"])
                    self.episodes_at_current_tier += 1

                    # PROMOTION CHECK: Every 100 episodes
                    if len(self.success_history) == 100 and self.episodes_at_current_tier >= 100:
                        avg_success = np.mean(self.success_history)
                        
                        # Threshold for promotion: 90% Success
                        if avg_success >= 0.90 and self.current_tier_idx < len(self.difficulty_tiers) - 1:
                            self.current_tier_idx += 1
                            self.success_history.clear()
                            self.episodes_at_current_tier = 0
                            print(f"\n[UPGRADE] Mastered Tier {self.difficulty_tiers[self.current_tier_idx-1]}!")
                            print(f"--> Advancing to Difficulty: {self.difficulty_tiers[self.current_tier_idx]}\n")

        # Log metrics to TensorBoard
        self.logger.record("curriculum/map_difficulty", current_diff)
        if len(self.success_history) > 0:
            self.logger.record("curriculum/rolling_success_rate", np.mean(self.success_history))
            
        return True

# --- 2. LEARNING RATE SCHEDULE ---
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# --- 3. TELEMETRY LOGGER ---
class TelemetryLoggerCallback(BaseCallback):
    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones")):
            if done:
                info = self.locals.get("infos")[i]
                if "telemetry" in info:
                    for key, value in info["telemetry"].items():
                        self.logger.record(f"telemetry/{key}", value)
        return True

def main():
    print(f"--- Booting BEANS Pipeline | Parallelism: {N_ENVS} Envs ---")
    
    # Vectorized Environment (This is where your multi-core speedup happens)
    env = make_vec_env(lambda: Monitor(PeopleBotEnv()), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128])
    )
    
    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=linear_schedule(3e-4),
        n_steps=2048,           
        batch_size=64,
        n_epochs=10,            
        gamma=0.995, # Increased for better long-term goal foresight
        gae_lambda=0.95,
        ent_coef=0.005,          
        device="auto"           
    )
    
    # Callbacks
    curriculum_cb = CompetenceCurriculumCallback()
    telemetry_cb = TelemetryLoggerCallback()
    checkpoint_cb = CheckpointCallback(
        save_freq=100000 // N_ENVS, # Adjusted frequency for parallel envs
        save_path=MODEL_DIR,
        name_prefix="BEANS_PPO_Adaptive"
    )
    
    callback_list = CallbackList([checkpoint_cb, telemetry_cb, curriculum_cb])
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=callback_list, 
            tb_log_name="PPO_Nav_Adaptive_Run", 
            progress_bar=True
        )
        model.save(os.path.join(MODEL_DIR, "BEANS_PPO_Final"))
    except KeyboardInterrupt:
        print("\nSaving progress before shutdown...")
        model.save(os.path.join(MODEL_DIR, "BEANS_PPO_Interrupted"))

if __name__ == "__main__":
    main()