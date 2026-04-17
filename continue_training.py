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
BASE_MODEL_PATH = os.path.join(MODEL_DIR, "BEANS_PPO_Adaptive_1800000_steps") 

ADDITIONAL_TIMESTEPS = 5_000_000 
N_ENVS = 10 

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

    # --- STATIC RUN CONFIGURATION ---
    # You MUST change this string manually if you start a completely new continuation run,
    # otherwise you will overwrite your previous checkpoints.
    RUN_NAME = "BEANS_Continued_v2" 
    
    # --- CALLBACK SETUP ---
    curriculum_cb = CompetenceCurriculumCallback()
    curriculum_cb.current_tier_idx = 4 
    telemetry_cb = TelemetryLoggerCallback()
    
    # The callback will automatically generate files like: "BEANS_Continued_v2_50000_steps.zip"
    checkpoint_cb = CheckpointCallback(
        save_freq=50000 // N_ENVS, 
        save_path=MODEL_DIR,
        name_prefix=RUN_NAME
    )
    callback_list = CallbackList([checkpoint_cb, telemetry_cb, curriculum_cb])
    
    # --- EXECUTION ---
    print(f"Resuming training for {ADDITIONAL_TIMESTEPS} timesteps under name: {RUN_NAME}...")
    try:
        model.learn(
            total_timesteps=ADDITIONAL_TIMESTEPS, 
            callback=callback_list, 
            tb_log_name=RUN_NAME, 
            progress_bar=True,
            reset_num_timesteps=False # CRITICAL: Keeps your global step count accurate
        )
        
        # Grab the absolute total step count directly from the model's brain for the final save
        final_steps = model.num_timesteps
        final_save_path = os.path.join(MODEL_DIR, f"{RUN_NAME}_Final_{final_steps}_steps")
        
        model.save(final_save_path)
        print(f"Training expansion complete. Final model saved to: {final_save_path}")
        
    except KeyboardInterrupt:
        print("\nInterrupt detected. Saving emergency checkpoint...")
        current_steps = model.num_timesteps
        model.save(os.path.join(MODEL_DIR, f"{RUN_NAME}_Interrupted_{current_steps}_steps"))

if __name__ == "__main__":
    main()