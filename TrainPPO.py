import gymnasium as gym
import numpy as np
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# Import your Environment
from PeopleBotEnv import PeopleBotEnv

# --- CONFIGURATION ---
TOTAL_TIMESTEPS = 2_000_000  
LOG_DIR = "logs/"
MODEL_DIR = "models/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env(rank, seed=0):
    """
    Utility function for environment.
    """
    def _init():
        print(f"Initializing Env #{rank}...") # Debug Print
        env = PeopleBotEnv()
        env.reset(seed=seed + rank) 
        log_file = os.path.join(LOG_DIR, str(rank))
        env = Monitor(env, log_file)
        return env
    return _init

def main():
    print("---------------------------------------")
    print("   PPO TRAINING - DEBUG MODE           ")
    print("---------------------------------------")
    
    # 1. USE SINGLE PROCESS FIRST (Safe Mode)
    # If this works, we can change num_cpu back to 4 later.
    num_cpu = 1 
    print(f"Launching {num_cpu} environment(s)...")
    
    # DummyVecEnv runs in the same process (No freezing)
    env = DummyVecEnv([make_env(i) for i in range(num_cpu)])

    # 2. NETWORK ARCHITECTURE
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh
    )
    
    # 3. SETUP MODEL
    print("Creating PPO Model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR, # <--- We will use this instead of the plot window
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        device="auto"
    )
    
    # Save checkpoint every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=MODEL_DIR,
        name_prefix="ppo_peoplebot_debug"
    )

    print(f"Starting Training for {TOTAL_TIMESTEPS} steps...")
    print("To view graphs, open terminal and run: tensorboard --logdir logs/")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback],
            progress_bar=True # Adds a nice loading bar in terminal
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving...")

    model.save(os.path.join(MODEL_DIR, "ppo_peoplebot_final"))
    print("Done.")
    env.close()

if __name__ == "__main__":
    main()