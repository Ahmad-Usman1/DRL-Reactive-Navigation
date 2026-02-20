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

# Import your Environment
from PeopleBotEnv import PeopleBotEnv

# --- CONFIGURATION ---
MODEL_PATH = "models/ppo_peoplebot_final"
RENDER_FPS = 30 
RENDER_SKIP = 2 

def main():
    print("--- DIAGNOSTIC MODE (FIXED) ACTIVATED ---")
    
    # 1. Load Environment
    env = PeopleBotEnv()

    # 2. Load Model
    load_path = MODEL_PATH
    if not os.path.exists(load_path + ".zip"):
        if os.path.exists("models"):
            models = [f for f in os.listdir("models") if f.endswith(".zip")]
            if models:
                load_path = os.path.join("models", models[-1]).replace(".zip", "")
    
    print(f"Loading Brain: {load_path}")
    model = PPO.load(load_path)

    # 3. Setup Visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    # Map & Objects
    obs, _ = env.reset() 
    # Invert map for display (0=White, 1=Black)
    map_display = ax.imshow(1 - env.map_grid, cmap='Greys', origin='lower', 
                            extent=[0, 20, 0, 20], vmin=0, vmax=1)

    robot_patch = Circle((0, 0), env.robot_radius, color='blue', zorder=10, alpha=0.8, label='Robot')
    ax.add_patch(robot_patch)
    
    arrow_patch = Arrow(0, 0, 0.5, 0, width=0.3, color='yellow', zorder=11)
    ax.add_patch(arrow_patch)
    
    goal_patch, = ax.plot([], [], 'g*', markersize=18, zorder=9, label='Goal')
    
    goal_zone = Circle((0,0), env.waypoint_radius, color='green', fill=False, linestyle='--', linewidth=2, zorder=8)
    ax.add_patch(goal_zone)
    
    lidar_lines = [Line2D([], [], color='red', linewidth=1, alpha=0.5, zorder=5) for _ in range(16)]
    for l in lidar_lines: ax.add_line(l)
    
    # HUD
    stats_text = ax.text(0.5, 19.5, 'Initializing...', color='black', fontsize=10, 
                         fontfamily='monospace', fontweight='bold', verticalalignment='top',
                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round'))

    ax.legend(loc='upper right')
    
    # --- SIMULATION LOOP ---
    num_episodes = 5
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        # Reset Map Display
        map_display.set_data(1 - env.map_grid)
        ax.set_title(f"Episode {ep+1}")
        
        print(f"\n--- Episode {ep+1} Start ---")
        
        # DIAGNOSTIC: Check Spawn Safety
        spawn_safety = np.min(obs[:16])
        print(f"Spawn Lidar Min Dist: {spawn_safety:.2f}m")
        
        while not done and not truncated:
            # RESTORE WIGGLES (Stochastic Policy) to fix wall clipping
            action, _ = model.predict(obs, deterministic=False)
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Visuals
            if step_count % RENDER_SKIP == 0:
                rx, ry, theta = env.current_pose
                gx, gy = env.current_goal  # Unpacks into two floats
                
                # Robot
                robot_patch.center = (rx, ry)
                
                # Arrow (Recreate to rotate)
                arrow_patch.remove()
                arrow_patch = Arrow(rx, ry, 0.6*math.cos(theta), 0.6*math.sin(theta), 
                                    width=0.3, color='yellow', zorder=11)
                ax.add_patch(arrow_patch)
                
                # Goal & Zone (FIXED: No indexing needed here)
                goal_patch.set_data([gx], [gy])
                goal_zone.center = (gx, gy)
                
                # Lidar
                scan = obs[:16]
                angles = theta + env.sensor_angles
                for i, line in enumerate(lidar_lines):
                    dist = scan[i]
                    if dist < env.max_sensor_range * 0.95:
                        ex = rx + dist * math.cos(angles[i])
                        ey = ry + dist * math.sin(angles[i])
                        line.set_data([rx, ex], [ry, ey])
                    else:
                        line.set_data([], [])
                
                hud_txt = (f"Vel: {env.current_lin_vel:.2f} m/s\n"
                           f"Ang: {env.current_ang_vel:.2f} rad/s\n"
                           f"Rew: {total_reward:.1f}")
                stats_text.set_text(hud_txt)
                
                plt.draw()
                plt.pause(1 / RENDER_FPS)
        
        if total_reward > 0:
            ax.set_title(f"Ep {ep+1}: SUCCESS ({total_reward:.1f})", color='green', fontweight='bold')
            print(f"-> Episode Result: SUCCESS (+{total_reward:.1f})")
        else:
            ax.set_title(f"Ep {ep+1}: FAILED ({total_reward:.1f})", color='red', fontweight='bold')
            print(f"-> Episode Result: CRASHED ({total_reward:.1f})")
            
        time.sleep(1.0)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()