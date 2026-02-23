import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from PeopleBotEnv import PeopleBotEnv

# --- DIRECTORIES ---
LOG_DIR = "./ppo_tensorboard/"
MODEL_DIR = "./saved_models/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("--- Booting RL Training Pipeline ---")
    
    # 1. Environment Wrapper
    # We wrap it in a Monitor to automatically log episodic rewards and lengths
    # make_vec_env automatically handles environment vectorization if you want to scale up later
    env = make_vec_env(lambda: Monitor(PeopleBotEnv()), n_envs=1)
    
    # 2. Hyperparameter Configuration
    # These are tuned for a dense-reward continuous navigation task
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        n_steps=2048,           # Steps collected before updating the network
        batch_size=64,
        n_epochs=10,            # Optimization passes per update
        gamma=0.99,             # Discount factor (long-term vision)
        gae_lambda=0.95,
        clip_range=0.2,         # Trust Region clipping parameter
        ent_coef=0.01,          # Encourages exploration (keeps the agent from getting stuck)
        device="auto"           # Automatically uses GPU if available, otherwise highly-optimized CPU
    )
    
    # 3. Checkpoint Callback
    # Saves a backup of the AI brain every 100,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=MODEL_DIR,
        name_prefix="BEANS_PPO"
    )
    
    # 4. Execute Training
    print("Beginning PPO Optimization...")
    TOTAL_TIMESTEPS = 3_000_000  # 3 Million steps is a solid baseline to see convergence
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=checkpoint_callback,
            tb_log_name="PPO_Nav_Run1",
            progress_bar=True
        )
        
        # 5. Save the final finalized model
        model.save(os.path.join(MODEL_DIR, "BEANS_PPO_Final"))
        print("\nTraining Complete. Final model saved.")
        
    except KeyboardInterrupt:
        print("\nTraining manually interrupted by user. Saving current brain state...")
        model.save(os.path.join(MODEL_DIR, "BEANS_PPO_Interrupted"))
        print("Model saved successfully. Safe to exit.")

if __name__ == "__main__":
    main()