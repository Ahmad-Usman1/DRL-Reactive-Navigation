import gymnasium as gym
import numpy as np
import os
import math
import time
import onnxruntime as ort
from stable_baselines3 import PPO
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

from PeopleBotEnv import PeopleBotEnv
from MapGenerator import MapGenerator

# --- CONFIGURATION ---
MODEL_DIR = "Performing_Models"
PYTORCH_MODEL = "BEANS_FineTuned_3600000_steps.zip" # Make sure this matches your best model
ONNX_FILENAME = "BEANS_FineTuned_3600000_steps.onnx" # Make sure this matches the exported ONNX file

class CompareEnv(PeopleBotEnv):
    def inject_frozen_map(self, grid, waypoints, resolution):
        """Forces the environment to use a pre-generated map for A/B testing."""
        self.reset() # Flush history and telemetry
        
        self.map_grid = grid.copy()
        self.waypoints = [wp.copy() for wp in waypoints]
        self.resolution = resolution
        self.total_checkpoints = max(1, len(self.waypoints) - 1)
        
        start_pt = self.waypoints[0]
        if len(self.waypoints) > 1:
            self.current_goal_index = 1
            self.current_goal = np.array(self.waypoints[1])
            start_theta = math.atan2(self.current_goal[1] - start_pt[1], 
                                     self.current_goal[0] - start_pt[0])
        else:
            self.current_goal_index = 0
            self.current_goal = np.array(start_pt)
            start_theta = 0.0
            
        self.current_pose = np.array([start_pt[0], start_pt[1], start_theta])
        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0
        self.previous_distance = np.linalg.norm(self.current_pose[:2] - self.current_goal)
        self.current_step = 0
        
        # Flush the action pipe perfectly
        self.action_history = np.zeros((self.lag_steps, 2), dtype=np.float32)
        
        return self._get_obs()

def run_agent(env, brain_type, model_or_session):
    """Runs a single episode and extracts telemetry."""
    obs = env._get_obs()
    done, truncated = False, False
    step_count = 0
    total_reward = 0.0
    
    hist_x, hist_y, hist_v, hist_w = [], [], [], []
    
    while not done and not truncated:
        hist_x.append(env.current_pose[0])
        hist_y.append(env.current_pose[1])
        hist_v.append(env.current_lin_vel)
        hist_w.append(env.current_ang_vel)
        
        if brain_type == "PYTORCH":
            action, _ = model_or_session.predict(obs, deterministic=True)
        elif brain_type == "ONNX":
            obs_tensor = np.array(obs, dtype=np.float32).reshape(1, -1)
            action = model_or_session.run(None, {"observation": obs_tensor})[0][0]
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # # Temporary: prints whenever clipping actually fires
            # if not np.allclose(raw_action, action, atol=1e-4):
            #     print(f"  [CLIP] raw={raw_action} → clipped={action}")
            
        obs, reward, done, truncated, _ = env.step(action)
        step_count += 1
        total_reward += reward
        
    if done and reward <= -10.0: outcome = "CRASHED"
    elif done: outcome = "SUCCESS"
    else: outcome = "TIMEOUT"
        
    pct = (env.current_goal_index / len(env.waypoints)) * 100
    
    return outcome, step_count, total_reward, pct, hist_x, hist_y, hist_v, hist_w

def plot_agent_row(fig, gs, row_idx, title_prefix, env, outcome, pct, hist_x, hist_y, hist_v, hist_w):
    """Plots a single agent's performance in a specific row of the grid."""
    # Map
    ax_map = fig.add_subplot(gs[row_idx, 0])
    ax_map.set_title(f"{title_prefix} | {outcome} | Cleared: {pct:.1f}%")
    ax_map.imshow(1 - env.map_grid, cmap='Greys', origin='lower', extent=[0, 40, 0, 40], vmin=0, vmax=1)
    
    wp_array = np.array(env.waypoints)
    ax_map.plot(wp_array[:, 0], wp_array[:, 1], 'r--', alpha=0.5, linewidth=2)
    
    points = np.array([hist_x, hist_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, env.max_lin_vel)
    lc = LineCollection(segments, cmap='plasma', norm=norm)
    lc.set_array(np.array(hist_v))
    lc.set_linewidth(3)
    ax_map.add_collection(lc)
    
    ax_map.plot(hist_x[0], hist_y[0], 'bo', markersize=8)
    ax_map.plot(hist_x[-1], hist_y[-1], 'go' if outcome == "SUCCESS" else 'ro', markersize=8)
    
    # Throttle
    ax_lin = fig.add_subplot(gs[row_idx, 1])
    ax_lin.set_title(f"{title_prefix} Throttle")
    ax_lin.plot(hist_v, color='blue', linewidth=2)
    ax_lin.axhline(y=env.max_lin_vel, color='r', linestyle='--', alpha=0.5)
    ax_lin.set_ylim([-0.1, env.max_lin_vel + 0.1])
    ax_lin.grid(True, alpha=0.3)
    
    # Steering
    ax_ang = fig.add_subplot(gs[row_idx, 2])
    ax_ang.set_title(f"{title_prefix} Steering")
    ax_ang.plot(hist_w, color='purple', linewidth=2)
    ax_ang.axhline(y=env.max_ang_vel, color='r', linestyle='--', alpha=0.5)
    ax_ang.axhline(y=-env.max_ang_vel, color='r', linestyle='--', alpha=0.5)
    ax_ang.set_ylim([-env.max_ang_vel - 0.5, env.max_ang_vel + 0.5])
    ax_ang.grid(True, alpha=0.3)

def main():
    print("--- BEANS A/B DIAGNOSTIC PROTOCOL ---")
    
    # Load Models
    pt_path = os.path.join(MODEL_DIR, PYTORCH_MODEL)
    onnx_path = os.path.join(MODEL_DIR, ONNX_FILENAME)
    
    print("Loading PyTorch Master...")
    pt_model = PPO.load(pt_path)
    
    print("Loading ONNX Clone...")
    onnx_session = ort.InferenceSession(onnx_path)
    
    env = CompareEnv()
    report_dir = "ab_test_results"
    os.makedirs(report_dir, exist_ok=True)
    
    num_tests = 10
    
    for ep in range(num_tests):
        print(f"\n--- Generating Map {ep+1}/{num_tests} ---")
        grid, wps, res = MapGenerator.generate(40, 40)
        
        # --- RUN PYTORCH ---
        env.inject_frozen_map(grid, wps, res)
        pt_out, pt_steps, pt_rew, pt_pct, pt_x, pt_y, pt_v, pt_w = run_agent(env, "PYTORCH", pt_model)
        print(f"PYTORCH | {pt_out:<7} | Steps: {pt_steps:04d} | Rew: {pt_rew:7.1f} | Cleared: {pt_pct:5.1f}%")
        
        # --- RUN ONNX ---
        env.inject_frozen_map(grid, wps, res)
        ox_out, ox_steps, ox_rew, ox_pct, ox_x, ox_y, ox_v, ox_w = run_agent(env, "ONNX", onnx_session)
        print(f"ONNX    | {ox_out:<7} | Steps: {ox_steps:04d} | Rew: {ox_rew:7.1f} | Cleared: {ox_pct:5.1f}%")
        
        # --- PLOT COMPARISON ---
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1.5, 1, 1], hspace=0.3)
        
        plot_agent_row(fig, gs, 0, "PyTorch", env, pt_out, pt_pct, pt_x, pt_y, pt_v, pt_w)
        plot_agent_row(fig, gs, 1, "ONNX", env, ox_out, ox_pct, ox_x, ox_y, ox_v, ox_w)
        
        plt.tight_layout()
        filepath = os.path.join(report_dir, f"compare_map_{ep+1:02d}.png")
        plt.savefig(filepath, dpi=150)
        plt.close(fig)
        
    print(f"\nA/B Tests complete. Check the '{report_dir}' folder.")

if __name__ == "__main__":
    main()