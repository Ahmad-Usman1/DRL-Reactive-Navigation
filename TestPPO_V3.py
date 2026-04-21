import gymnasium as gym
import numpy as np
import os
import math
import time
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Arrow
from stable_baselines3 import PPO

# --- V3 IMPORTS ---
from PeopleBotEnv_V3 import PeopleBotEnv
from MapGenerator import MapGenerator
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

# --- CONFIGURATION ---
MODEL_DIR = "finetune_models_v3_phase2b"
MODEL_PREFIX = "BEANS_V3"
RENDER_FPS = 30 
RENDER_SKIP = 2 

# --- DYNAMIC TEST ENVIRONMENT ---
class DynamicTestEnv_V3(PeopleBotEnv):
    """Overrides the reset function to generate fresh maps dynamically instead of using the RAM Bank."""
    def reset(self, seed=None, options=None):
        # Explicitly call the base Gym reset to handle random seeds
        gym.Env.reset(self, seed=seed)
        
        # --- V3 SPECIFIC: FLUSH DELAY FIFOS ---
        self.action_history = np.zeros((self.lag_steps, 2), dtype=np.float32)
        self._init_sensor_fifos()
        
        # --- TELEMETRY TRACKERS ---
        self.ep_velocity_history = []
        self.ep_min_lidar_history = []
        self.ep_vibration_count = 0
        self.ep_checkpoints_hit = 0

        # 1. Bypass MapBank, generate a completely new 40x40 map
        self.map_grid, self.waypoints, self.resolution = MapGenerator.generate(40, 40)
        self.total_checkpoints = max(1, len(self.waypoints) - 1)
            
        start_pt = self.waypoints[0]
        if len(self.waypoints) > 1:
            self.current_goal_index = 1
            self.current_goal = np.array(self.waypoints[1], dtype=np.float32)
            start_theta = math.atan2(self.current_goal[1] - start_pt[1], 
                                     self.current_goal[0] - start_pt[0])
        else:
            self.current_goal_index = 0
            self.current_goal = np.array(start_pt, dtype=np.float32)
            start_theta = 0.0
            
        self.current_pose = np.array([start_pt[0], start_pt[1], start_theta])
        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0
        self.previous_distance = float(np.linalg.norm(self.current_pose[:2] - self.current_goal))
        self.current_step = 0
        
        # Pull safe 25-dim obs via V3 helper
        return self._get_obs(), {}

def main():
    print("--- DIAGNOSTIC PIPELINE ACTIVATED (V3) ---")
    
    # 1. Load Model
    if not os.path.exists(MODEL_DIR):
        print(f"FATAL ERROR: Directory '{MODEL_DIR}' not found. Have you trained a V3 model yet?")
        return
        
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".zip")]
    if not models:
        print(f"FATAL ERROR: No trained models found in {MODEL_DIR}.")
        return
        
    # Attempt to grab the final model, otherwise grab the latest checkpoint
    latest_model = "BEANS_V3_Phase2b_Final.zip"
    if latest_model not in models:
        # Sort to get highest step count
        latest_model = sorted(models)[-1]

    load_path = os.path.join(MODEL_DIR, latest_model)
    print(f"Loading Brain: {load_path}")
    model = PPO.load(load_path)

    # 2. Load Dynamic Environment
    env = DynamicTestEnv_V3()

    # ==========================================
    # PHASE 2: DIAGNOSTIC DASHBOARD (20 MAPS)
    # ==========================================
    print("\n--- PHASE 2: DIAGNOSTIC DASHBOARD GENERATION (20 MAPS) ---")
    num_headless = 20
    
    success_count = 0
    crash_count = 0
    timeout_count = 0
    
    # Create directory for reports
    report_dir = "test_results_v3"
    os.makedirs(report_dir, exist_ok=True)
    
    for ep in range(num_headless):
        obs, _ = env.reset()
        done = False
        truncated = False
        step_count = 0
        total_reward = 0.0
        outcome = "UNKNOWN"
        
        # Telemetry Arrays
        hist_x, hist_y = [], []
        hist_v, hist_w = [], []
        
        while not done and not truncated:
            # Log exact position and velocity before the step
            hist_x.append(env.current_pose[0])
            hist_y.append(env.current_pose[1])
            hist_v.append(env.current_lin_vel)
            hist_w.append(env.current_ang_vel)
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            step_count += 1
            total_reward += reward
            
            # Pull outcome strictly from V3 Telemetry dict
            if done or truncated:
                tel = info.get("telemetry", {})
                if tel.get("rate_crash", 0.0) == 1.0:
                    outcome = "CRASHED"
                    crash_count += 1
                elif tel.get("rate_success", 0.0) == 1.0:
                    outcome = "SUCCESS"
                    success_count += 1
                else:
                    outcome = "TIMEOUT"
                    timeout_count += 1
            
        progress_pct = (env.current_goal_index / len(env.waypoints)) * 100
        print(f"Map {ep+1:02d} | Result: {outcome:<7} | Steps: {step_count:04d} | Reward: {total_reward:7.1f} | Path Cleared: {progress_pct:5.1f}%")

        # --- GENERATE DIAGNOSTIC DASHBOARD ---
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1])
        
        # Panel 1: The Spatial Map
        ax_map = fig.add_subplot(gs[:, 0])
        ax_map.set_title(f"Run {ep+1:02d}: {outcome} | Cleared: {progress_pct:.1f}%")
        ax_map.imshow(1 - env.map_grid, cmap='Greys', origin='lower', extent=[0, 40, 0, 40], vmin=0, vmax=1)
        
        # Plot Global Path (Dashed Red)
        wp_array = np.array(env.waypoints)
        ax_map.plot(wp_array[:, 0], wp_array[:, 1], 'r--', alpha=0.5, linewidth=2, label='Global Path')
        
        # Plot Driven Path with Velocity Gradient
        points = np.array([hist_x, hist_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # 'plasma' colormap: dark blue/purple = slow, bright yellow = fast
        norm = plt.Normalize(0, env.max_lin_vel)
        lc = LineCollection(segments, cmap='plasma', norm=norm)
        lc.set_array(np.array(hist_v))
        lc.set_linewidth(3)
        line = ax_map.add_collection(lc)
        fig.colorbar(line, ax=ax_map, label='Linear Velocity (m/s)', fraction=0.046, pad=0.04)
        
        # Mark Start (Blue) and End (Green or Red depending on crash)
        ax_map.plot(hist_x[0], hist_y[0], 'bo', markersize=8, label='Start')
        end_color = 'go' if outcome == "SUCCESS" else 'ro'
        ax_map.plot(hist_x[-1], hist_y[-1], end_color, markersize=8, label='End')
        ax_map.legend()
        
        # Panel 2: Linear Velocity Graph
        ax_lin = fig.add_subplot(gs[0, 1])
        ax_lin.set_title("Throttle Control (Linear Velocity)")
        ax_lin.plot(hist_v, color='blue', linewidth=2)
        ax_lin.axhline(y=env.max_lin_vel, color='r', linestyle='--', alpha=0.5, label='Hardware Max')
        ax_lin.set_ylabel("Velocity (m/s)")
        ax_lin.grid(True, alpha=0.3)
        ax_lin.legend()
        
        # Panel 3: Angular Velocity Graph
        ax_ang = fig.add_subplot(gs[1, 1])
        ax_ang.set_title("Steering Control (Angular Velocity)")
        ax_ang.plot(hist_w, color='purple', linewidth=2)
        ax_ang.axhline(y=env.max_ang_vel, color='r', linestyle='--', alpha=0.5, label='Hardware Max')
        ax_ang.axhline(y=-env.max_ang_vel, color='r', linestyle='--', alpha=0.5)
        ax_ang.set_ylabel("Angular Vel (rad/s)")
        ax_ang.set_xlabel("Time Steps")
        ax_ang.grid(True, alpha=0.3)
        
        # Save and close to prevent memory leaks
        plt.tight_layout()
        safe_outcome = outcome.lower()
        filepath = os.path.join(report_dir, f"run_{ep+1:02d}_{safe_outcome}.png")
        plt.savefig(filepath, dpi=150)
        plt.close(fig)

    # --- FINAL REPORT ---
    print("\n==================================")
    print("      FINAL ROBUSTNESS REPORT     ")
    print("==================================")
    print(f"Total Maps Evaluated: {num_headless}")
    print(f"Success Rate:         {(success_count/num_headless)*100:.1f}% ({success_count})")
    print(f"Crash Rate:           {(crash_count/num_headless)*100:.1f}% ({crash_count})")
    print(f"Timeout Rate:         {(timeout_count/num_headless)*100:.1f}% ({timeout_count})")
    print("==================================\n")
    print(f"Dashboards saved to: ./{report_dir}/")

if __name__ == "__main__":
    main()