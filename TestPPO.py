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

from PeopleBotEnv import PeopleBotEnv
from MapGenerator import MapGenerator
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

# --- CONFIGURATION ---
MODEL_DIR = "saved_models"
MODEL_PREFIX = "BEANS_PPO"
RENDER_FPS = 30 
RENDER_SKIP = 2 

# --- DYNAMIC TEST ENVIRONMENT ---
class DynamicTestEnv(PeopleBotEnv):
    """Overrides the reset function to generate fresh maps dynamically instead of using the RAM Bank."""
    def reset(self, seed=None, options=None):
        # Explicitly call the base Gym reset to handle random seeds, skipping PeopleBotEnv's reset
        gym.Env.reset(self, seed=seed)
        
        # 1. Bypass MapBank, generate a completely new 40x40 map
        self.map_grid, self.waypoints, self.resolution = MapGenerator.generate(40, 40)
            
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
        
        return self._get_obs(), {}

def main():
    print("--- DIAGNOSTIC PIPELINE ACTIVATED ---")
    
    # 1. Load Model
    if not os.path.exists(MODEL_DIR):
        print(f"FATAL ERROR: Directory '{MODEL_DIR}' not found.")
        return
        
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".zip")]
    if not models:
        print("FATAL ERROR: No trained models found in directory.")
        return
        
    # Grab the most recently saved model
    print(sorted(models))
    latest_model = sorted(models)[-1]
    load_path = os.path.join(MODEL_DIR, latest_model).replace(".zip", "")
    
    print(f"Loading Brain: {load_path}")
    model = PPO.load(load_path)

    # 2. Load Dynamic Environment
    env = DynamicTestEnv()

    # # ==========================================
    # # PHASE 1: VISUALIZATION (3 MAPS)
    # # ==========================================
    # print("\n--- PHASE 1: VISUAL EVALUATION (3 MAPS) ---")
    # plt.ion()
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.set_aspect('equal')
    
    # for ep in range(3):
    #     obs, _ = env.reset()
    #     done = False
    #     truncated = False
    #     total_reward = 0
    #     step_count = 0
        
    #     ax.clear()
    #     ax.set_title(f"Visual Run {ep+1}/3")
        
    #     # Plot Map (Fixed for 40x40m bounds)
    #     ax.imshow(1 - env.map_grid, cmap='Greys', origin='lower', 
    #               extent=[0, 40, 0, 40], vmin=0, vmax=1)
        
    #     # Plot Global Waypoint Path
    #     wp_array = np.array(env.waypoints)
    #     ax.plot(wp_array[:, 0], wp_array[:, 1], 'r--', alpha=0.5, zorder=4, label='Global Path')
        
    #     robot_patch = Circle((env.current_pose[0], env.current_pose[1]), env.robot_radius, color='blue', zorder=10, alpha=0.8, label='Robot')
    #     ax.add_patch(robot_patch)
        
    #     arrow_patch = Arrow(0, 0, 0.5, 0, width=0.3, color='yellow', zorder=11)
    #     ax.add_patch(arrow_patch)
        
    #     goal_patch, = ax.plot([env.current_goal[0]], [env.current_goal[1]], 'g*', markersize=18, zorder=9, label='Current Goal')
    #     goal_zone = Circle((env.current_goal[0], env.current_goal[1]), env.waypoint_radius, color='green', fill=False, linestyle='--', linewidth=2, zorder=8)
    #     ax.add_patch(goal_zone)
        
    #     lidar_lines = [Line2D([], [], color='red', linewidth=1, alpha=0.5, zorder=5) for _ in range(16)]
    #     for l in lidar_lines: ax.add_line(l)
        
    #     stats_text = ax.text(0.5, 39.5, 'Initializing...', color='black', fontsize=10, 
    #                          fontfamily='monospace', fontweight='bold', verticalalignment='top',
    #                          bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))
        
    #     ax.legend(loc='upper right')
        
    #     while not done and not truncated:
    #         # Deterministic=True tests the pure policy without exploration noise
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, done, truncated, info = env.step(action)
    #         total_reward += reward
    #         step_count += 1
            
    #         if step_count % RENDER_SKIP == 0:
    #             rx, ry, theta = env.current_pose
    #             gx, gy = env.current_goal 
                
    #             robot_patch.center = (rx, ry)
    #             arrow_patch.remove()
    #             arrow_patch = Arrow(rx, ry, 0.6*math.cos(theta), 0.6*math.sin(theta), width=0.3, color='yellow', zorder=11)
    #             ax.add_patch(arrow_patch)
                
    #             goal_patch.set_data([gx], [gy])
    #             goal_zone.center = (gx, gy)
                
    #             scan = obs[:16]
    #             angles = theta + env.sensor_angles
    #             for i, line in enumerate(lidar_lines):
    #                 dist = scan[i]
    #                 if dist < env.max_sensor_range * 0.95:
    #                     ex = rx + dist * math.cos(angles[i])
    #                     ey = ry + dist * math.sin(angles[i])
    #                     line.set_data([rx, ex], [ry, ey])
    #                 else:
    #                     line.set_data([], [])
                
    #             hud_txt = (f"Vel: {env.current_lin_vel:.2f} m/s\n"
    #                        f"Ang: {env.current_ang_vel:.2f} rad/s\n"
    #                        f"Rew: {total_reward:.1f}\n"
    #                        f"Step: {step_count}/{env.max_steps}")
    #             stats_text.set_text(hud_txt)
                
    #             plt.draw()
    #             plt.pause(1 / RENDER_FPS)
                
    #     time.sleep(1.0) # Pause to see final result

    # plt.ioff()
    # plt.close()

    # ==========================================
    # PHASE 2: DIAGNOSTIC DASHBOARD (20 MAPS)
    # ==========================================
    print("\n--- PHASE 2: DIAGNOSTIC DASHBOARD GENERATION (20 MAPS) ---")
    num_headless = 20
    
    success_count = 0
    crash_count = 0
    timeout_count = 0
    
    # Create directory for reports
    report_dir = "test_results"
    os.makedirs(report_dir, exist_ok=True)
    
    for ep in range(num_headless):
        obs, _ = env.reset()
        done = False
        truncated = False
        step_count = 0
        total_reward = 0.0
        
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
            
        # Analyze outcome
        if done and reward <= -10.0: 
            outcome = "CRASHED"
            crash_count += 1
        elif done:
            outcome = "SUCCESS"
            success_count += 1
        elif truncated:
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