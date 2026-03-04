import os
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from PeopleBotEnv import PeopleBotEnv

# --- DIRECTORIES ---
LOG_DIR = "./ppo_tensorboard/"
MODEL_DIR = "./saved_models/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- CUSTOM TELEMETRY LOGGER ---
class TelemetryLoggerCallback(BaseCallback):
    """
    Custom callback to extract physical telemetry from the environment
    and plot it in a dedicated TensorBoard category.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Loop through all parallel environments (even if n_envs=1)
        for i, done in enumerate(self.locals.get("dones")):
            if done:
                # Extract the info dictionary for the environment that just finished
                info = self.locals.get("infos")[i]
                
                # If our custom telemetry payload exists, log it to TensorBoard
                if "telemetry" in info:
                    for key, value in info["telemetry"].items():
                        # This creates a new "telemetry/" folder in TensorBoard
                        self.logger.record(f"telemetry/{key}", value)
        return True

def main():
    print("--- Booting RL Training Pipeline (Deep Architecture + Telemetry) ---")
    
    # 1. Environment Wrapper
    env = make_vec_env(lambda: Monitor(PeopleBotEnv()), n_envs=1)
    
    # 2. Hyperparameter & Architecture Configuration
    # --- NEW: DEEP NETWORK ARCHITECTURE ---
    # Upgraded from default [64, 64] to give the AI the capacity to understand physics
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256])
    )
    
    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs, # Injecting the larger brain
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=2048,           
        batch_size=64,
        n_epochs=10,            
        gamma=0.99,             
        gae_lambda=0.95,
        clip_range=0.2,         
        ent_coef=0.01,          
        device="auto"           
    )
    
    # 3. Callbacks Setup
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=MODEL_DIR,
        name_prefix="BEANS_PPO_Deep" # Renamed to prevent overwriting old weights
    )
    
    telemetry_callback = TelemetryLoggerCallback()
    
    # Combine the callbacks into a single list for the learn() function
    callback_list = CallbackList([checkpoint_callback, telemetry_callback])
    
    # 4. Execute Training
    print("Beginning PPO Optimization...")
    # --- NEW: EXTENDED TRAINING HORIZON ---
    TOTAL_TIMESTEPS = 5_000_000  # 5 Million steps for physics mastery
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=callback_list, # Injected the combined callbacks here
            tb_log_name="PPO_Nav_Deep_5M", # Renamed for TensorBoard tracking
            progress_bar=True
        )
        
        # 5. Save the final finalized model
        model.save(os.path.join(MODEL_DIR, "BEANS_PPO_Final_Deep"))
        print("\nTraining Complete. Final model saved.")
        
    except KeyboardInterrupt:
        print("\nTraining manually interrupted by user. Saving current brain state...")
        model.save(os.path.join(MODEL_DIR, "BEANS_PPO_Interrupted_Deep"))
        print("Model saved successfully. Safe to exit.")

if __name__ == "__main__":
    main()