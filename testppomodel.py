import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from PeopleBotEnv import PeopleBotEnv

def test_ppo_performance(model_path, num_episodes=20, test_difficulty=1):
    # 1. Create Output Directory
    output_dir = "ppo_test_results"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Setup Environment
    env = PeopleBotEnv()
    env.set_difficulty(test_difficulty)
    
    # 3. Load the Model
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return
    
    print(f"--- Loading PPO Model: {model_path} ---")
    model = PPO.load(model_path)
    
    success_count = 0
    print(f"--- Starting AI Evaluation (Difficulty: {test_difficulty}) ---")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        
        path_history = []
        v_history = []
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Use deterministic=True to remove the "exploration noise"
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record robot position
            path_history.append(env.current_pose[:2].copy())
            v_history.append(env.current_lin_vel)

        # Calculate Success
        is_success = info['telemetry']['rate_success'] > 0
        if is_success: success_count += 1

        # --- GENERATE PLOT ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot 1: The Navigation Map
        ax1.imshow(env.map_grid == 0, cmap='gray', origin='lower')
        path_pts = np.array(path_history) * env.resolution
        ax1.plot(path_pts[:, 0], path_pts[:, 1], color='magenta', label='AI Path', linewidth=2)
        
        wp_px = np.array(env.waypoints) * env.resolution
        ax1.scatter(wp_px[:, 0], wp_px[:, 1], c='red', s=30, label='Waypoints')
        ax1.set_title(f"Episode {ep+1} | Success: {is_success}")
        ax1.legend()

        # Plot 2: Velocity Profile
        ax2.plot(v_history, color='magenta', label='Lin Vel (m/s)')
        ax2.set_title("AI Velocity Control")
        ax2.set_ylabel("m/s")
        ax2.set_xlabel("Steps")
        ax2.axhline(y=0.8, color='r', linestyle='--', label='Max Speed')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/ai_run_{ep+1:02d}.png")
        plt.close()
        
        print(f"Run {ep+1:02d}: {'SUCCESS' if is_success else 'CRASHED'}")

    final_rate = (success_count / num_episodes) * 100
    print(f"\n--- Final AI Performance: {final_rate}% Success Rate ---")
    print(f"Check the '{output_dir}' folder for the visualization maps.")

if __name__ == "__main__":
    # Update this path to point to your 1M step model file
    MODEL_FILE = "saved_models/BEANS_PPO_Adaptive_1000000_steps.zip" 
    test_ppo_performance(MODEL_FILE)