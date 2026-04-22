"""
TestPPO_V5.py
================
Diagnostic dashboard and testing pipeline specifically built for PeopleBotEnv_V5.
Extracts V5-specific telemetry (Path Efficiency, Jerk Peak, Domain Randomization).
"""

import gymnasium as gym
import numpy as np
import os
import math
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
from stable_baselines3 import PPO

# --- V5 IMPORTS ---
from PeopleBotEnvV5 import PeopleBotEnv, fast_raycast
from MapGenerator import MapGenerator

# --- CONFIGURATION ---
MODEL_DIR = "models_beans_v5"  # Ensure this points to where your V5 training script saves models
REPORT_DIR = "test_results_v5"

class DynamicTestEnv_V5(PeopleBotEnv):
    """
    Overrides the reset function to generate fresh maps dynamically using MapGenerator,
    bypassing the MapBank while maintaining strict V5 physics and FIFOs.
    """
    def reset(self, seed=None, options=None):
        # 1. Call base Gym reset for seeds
        gym.Env.reset(self, seed=seed)

        # 2. Flush V5 Asymmetric Delay Pipelines
        self.obs_scan_fifo[:] = self.max_sensor_range
        self.obs_vel_fifo[:]  = 0.0
        self.action_fifo[:]   = 0.0

        # 3. Domain Randomization (Crucial for V5)
        rng = np.random.default_rng(seed)
        self.tau_v = float(rng.uniform(0.50, 1.15))
        self.tau_w = float(rng.uniform(0.03, 0.10))

        # 4. Reset Dynamics & Telemetry
        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0
        self.prev_ang_vel    = 0.0
        self.jerk_integrator = 0.0
        
        self.ep_velocity_history  = []
        self.ep_min_lidar_history = []
        self.ep_vibration_count   = 0
        self.ep_checkpoints_hit   = 0

        # 5. Generate Dynamic Map (Bypass MapBank)
        self.map_grid, raw_wps, self.resolution = MapGenerator.generate(size_x=40, size_y=40)
        self.waypoints = [list(wp) for wp in raw_wps]
        self.total_checkpoints = max(1, len(self.waypoints) - 1)

        # Pre-compute ideal A* path length for V5 Efficiency Metric
        self.ideal_path_length = max(0.1, sum(
            math.hypot(self.waypoints[i + 1][0] - self.waypoints[i][0],
                       self.waypoints[i + 1][1] - self.waypoints[i][1])
            for i in range(len(self.waypoints) - 1)
        ))
        self.traversed_path_length = 0.0

        # 6. Set Starting Pose
        start_pt = self.waypoints[0]
        if len(self.waypoints) > 1:
            self.current_goal_index = 1
            self.current_goal       = np.array(self.waypoints[1], dtype=np.float32)
            start_theta = math.atan2(self.current_goal[1] - start_pt[1],
                                     self.current_goal[0] - start_pt[0])
        else:
            self.current_goal_index = 0
            self.current_goal       = np.array(start_pt, dtype=np.float32)
            start_theta = 0.0

        self.current_pose = np.array([start_pt[0], start_pt[1], start_theta], dtype=np.float64)
        self.previous_distance = float(np.linalg.norm(self.current_pose[:2] - self.current_goal))
        self.current_step = 0

        # 7. Warm the V5 Observation FIFO with the true starting scan
        init_scan = fast_raycast(
            self.current_pose[0], self.current_pose[1], self.current_pose[2],
            self.sensor_angles, self.map_grid, self.resolution, self.max_sensor_range
        )
        self.obs_scan_fifo[:] = init_scan
        self.obs_vel_fifo[:]  = 0.0

        return self._get_obs(), {}


def main():
    print("--- BEANS DIAGNOSTIC PIPELINE ACTIVATED (V5) ---")
    
    if not os.path.exists(MODEL_DIR):
        print(f"FATAL ERROR: Directory '{MODEL_DIR}' not found. Verify your V5 model path.")
        return
        
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".zip")]
    if not models:
        print(f"FATAL ERROR: No trained models found in {MODEL_DIR}.")
        return
        
    # Grab the most recent model based on string sorting (assuming step-count naming)
    latest_model = sorted(models)[-1]
    load_path = os.path.join(MODEL_DIR, latest_model)
    print(f"Loading Brain: {load_path}")
    
    model = PPO.load(load_path)
    env = DynamicTestEnv_V5()

    print("\n--- PHASE 2: DIAGNOSTIC DASHBOARD GENERATION (20 MAPS) ---")
    num_headless = 20
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    success_count, crash_count, timeout_count = 0, 0, 0
    avg_efficiency = []
    
    for ep in range(num_headless):
        obs, _ = env.reset()
        done = False
        truncated = False
        step_count = 0
        total_reward = 0.0
        outcome = "UNKNOWN"
        telemetry_data = {}
        
        hist_x, hist_y = [], []
        hist_v, hist_w = [], []
        hist_jerk = []
        
        while not done and not truncated:
            hist_x.append(env.current_pose[0])
            hist_y.append(env.current_pose[1])
            hist_v.append(env.current_lin_vel)
            hist_w.append(env.current_ang_vel)
            hist_jerk.append(env.jerk_integrator)
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            step_count += 1
            total_reward += reward
            
            if done or truncated:
                telemetry_data = info.get("telemetry", {})
                if telemetry_data.get("rate_crash", 0.0) == 1.0:
                    outcome = "CRASHED"
                    crash_count += 1
                elif telemetry_data.get("rate_success", 0.0) == 1.0:
                    outcome = "SUCCESS"
                    success_count += 1
                    avg_efficiency.append(telemetry_data.get("path_efficiency", 0.0))
                else:
                    outcome = "TIMEOUT"
                    timeout_count += 1
            
        progress_pct = (env.current_goal_index / len(env.waypoints)) * 100
        eff_str = f"{telemetry_data.get('path_efficiency', 0.0)*100:5.1f}%" if outcome == "SUCCESS" else "N/A"
        
        print(f"Map {ep+1:02d} | {outcome:<7} | Steps: {step_count:04d} | Rew: {total_reward:7.1f} | Path: {progress_pct:5.1f}% | Eff: {eff_str}")

        # --- GENERATE V5 DIAGNOSTIC DASHBOARD ---
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(3, 2, width_ratios=[1.2, 1])
        
        # Panel 1: Map & Path
        ax_map = fig.add_subplot(gs[:, 0])
        title_str = f"Run {ep+1:02d}: {outcome} | $\\tau_v$={env.tau_v:.2f}s | $\\tau_w$={env.tau_w:.2f}s"
        ax_map.set_title(title_str, fontsize=14, fontweight='bold')
        ax_map.imshow(1 - env.map_grid, cmap='Greys', origin='lower', extent=[0, 40, 0, 40], vmin=0, vmax=1)
        
        wp_array = np.array(env.waypoints)
        ax_map.plot(wp_array[:, 0], wp_array[:, 1], 'r--', alpha=0.5, linewidth=2, label='A* Ideal Path')
        
        points = np.array([hist_x, hist_y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, env.max_lin_vel)
        lc = LineCollection(segments, cmap='plasma', norm=norm)
        lc.set_array(np.array(hist_v))
        lc.set_linewidth(3)
        line = ax_map.add_collection(lc)
        fig.colorbar(line, ax=ax_map, label='Linear Velocity (m/s)', fraction=0.046, pad=0.04)
        
        ax_map.plot(hist_x[0], hist_y[0], 'bo', markersize=8, label='Start')
        ax_map.plot(hist_x[-1], hist_y[-1], 'go' if outcome == "SUCCESS" else 'ro', markersize=8, label='End')
        ax_map.legend()
        
        # Panel 2: Linear Velocity
        ax_lin = fig.add_subplot(gs[0, 1])
        ax_lin.set_title("Throttle Control & Velocity")
        ax_lin.plot(hist_v, color='blue', linewidth=2)
        ax_lin.axhline(y=env.max_lin_vel, color='r', linestyle='--', alpha=0.5, label='Hardware Max')
        ax_lin.set_ylabel("Velocity (m/s)")
        ax_lin.grid(True, alpha=0.3)
        ax_lin.legend(loc="upper right")
        
        # Panel 3: Angular Velocity
        ax_ang = fig.add_subplot(gs[1, 1])
        ax_ang.set_title("Steering Control")
        ax_ang.plot(hist_w, color='purple', linewidth=2)
        ax_ang.axhline(y=env.max_ang_vel, color='r', linestyle='--', alpha=0.5)
        ax_ang.axhline(y=-env.max_ang_vel, color='r', linestyle='--', alpha=0.5)
        ax_ang.set_ylabel("Angular Vel (rad/s)")
        ax_ang.grid(True, alpha=0.3)

        # Panel 4: V5 Jerk Integrator State
        ax_jerk = fig.add_subplot(gs[2, 1])
        ax_jerk.set_title("Leaky Jerk Integrator (Oscillation Debt)")
        ax_jerk.plot(hist_jerk, color='red', linewidth=2)
        ax_jerk.axhline(y=env.J_MAX, color='black', linestyle='--', alpha=0.5, label='J_MAX Normalizer')
        ax_jerk.set_ylabel("Jerk Debt")
        ax_jerk.set_xlabel("Time Steps")
        ax_jerk.grid(True, alpha=0.3)
        ax_jerk.legend(loc="upper right")
        
        plt.tight_layout()
        safe_outcome = outcome.lower()
        filepath = os.path.join(REPORT_DIR, f"run_{ep+1:02d}_{safe_outcome}.png")
        plt.savefig(filepath, dpi=150)
        plt.close(fig)

    # --- FINAL REPORT ---
    print("\n========================================")
    print("      V5 FINAL ROBUSTNESS REPORT      ")
    print("========================================")
    print(f"Total Maps Evaluated: {num_headless}")
    print(f"Success Rate:         {(success_count/num_headless)*100:.1f}% ({success_count})")
    print(f"Crash Rate:           {(crash_count/num_headless)*100:.1f}% ({crash_count})")
    print(f"Timeout Rate:         {(timeout_count/num_headless)*100:.1f}% ({timeout_count})")
    if success_count > 0:
        print(f"Avg Path Efficiency:  {np.mean(avg_efficiency)*100:.1f}% (Successes Only)")
    print("========================================\n")
    print(f"Dashboards saved to: ./{REPORT_DIR}/")

if __name__ == "__main__":
    main()