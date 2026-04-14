import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from PeopleBotEnv import PeopleBotEnv
from TrainPPO import CompetenceCurriculumCallback, TelemetryLoggerCallback, CheckpointCallback, CallbackList

# --- CONFIGURATION ---
LOG_DIR = "./ppo_tensorboard/"
MODEL_DIR = "./saved_models/"

# UPDATE THIS: Point this directly to your newly trained 4m/0.4v model file
BASE_MODEL_PATH = os.path.join(MODEL_DIR, "YOUR_MODEL_NAME_HERE") 

ADDITIONAL_TIMESTEPS = 5_000_000 
N_ENVS = 4 

def main():
    print("--- Booting BEANS Continuation Pipeline ---")
    
    env = make_vec_env(lambda: Monitor(PeopleBotEnv()), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    
    print(f"Loading weights and optimizer state from: {BASE_MODEL_PATH}...")
    try:
        # NOTICE: No custom_objects. We are loading the brain exactly as it was when you stopped it.
        model = PPO.load(BASE_MODEL_PATH, env=env)
    except Exception as e:
        print(f"FAILED TO LOAD MODEL. Ensure the file exists in {MODEL_DIR}.")
        print(f"Error: {e}")
        return

    # --- CALLBACK SETUP ---
    curriculum_cb = CompetenceCurriculumCallback()
    
    # CRITICAL MANUAL OVERRIDE:
    # You must set this index to match wherever your previous training run left off.
    # 0 = Tier 0.05, 1 = Tier 0.10, 2 = Tier 0.25, 3 = Tier 0.50, 4 = Tier 0.75, 5 = Tier 1.0
    curriculum_cb.current_tier_idx = 3 
    
    telemetry_cb = TelemetryLoggerCallback()
    
    run_id = int(time.time())
    checkpoint_cb = CheckpointCallback(
        save_freq=50000 // N_ENVS, 
        save_path=MODEL_DIR,
        name_prefix=f"BEANS_Continued_{run_id}"
    )
    callback_list = CallbackList([checkpoint_cb, telemetry_cb, curriculum_cb])
    
    # --- EXECUTION ---
    print(f"Resuming training for {ADDITIONAL_TIMESTEPS} timesteps...")
    try:
        model.learn(
            total_timesteps=ADDITIONAL_TIMESTEPS, 
            callback=callback_list, 
            tb_log_name=f"PPO_Continued_Run_{run_id}", 
            progress_bar=True,
            # reset_num_timesteps=False ensures TensorBoard continues the x-axis 
            # from where it left off, rather than resetting to Step 0.
            reset_num_timesteps=False 
        )
        model.save(os.path.join(MODEL_DIR, f"BEANS_Continued_Final_{run_id}"))
        print("Training expansion complete. Model saved.")
    except KeyboardInterrupt:
        print("\nSaving progress before shutdown...")
        model.save(os.path.join(MODEL_DIR, f"BEANS_Continued_Interrupted_{run_id}"))

if __name__ == "__main__":
    main()