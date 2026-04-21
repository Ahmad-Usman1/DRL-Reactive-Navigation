import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# --- V3 IMPORTS ---
from Finetune_V3 import FineTuneBotEnv_V3
from FineTuneMaps import (
    CornerGauntletMap, 
    PinchPointMap, 
    ClutterCorridorMap, 
    ForkTrapMap, 
    BlindCornerMap, 
)

# Define the model path
MODEL_PATH = "finetune_models_v3/BEANS_V3_FineTuned_1700000_steps.zip"

def evaluate_map(env, model, map_class):
    """Generates a specific map, injects it into the env, and runs the policy."""
    # 1. Generate specialized map
    grid, waypoints, res = map_class.generate(size_x=40, size_y=40)
    
    # 2. Reset env (Crucial in V3 to flush the action and sensor FIFOs)
    env.reset()
    
    # Forcefully inject the custom map geometry
    env.map_grid = grid.copy()
    env.waypoints = waypoints.copy()
    env.resolution = res
    env.total_checkpoints = max(1, len(waypoints) - 1)
    
    env.current_goal_index = 1 if len(waypoints) > 1 else 0
    env.current_goal = np.array(waypoints[env.current_goal_index], dtype=np.float32)
    
    start_theta = 0.0
    if len(waypoints) > 1:
        start_theta = np.arctan2(waypoints[1][1] - waypoints[0][1], 
                                 waypoints[1][0] - waypoints[0][0])
        
    env.current_pose = np.array([waypoints[0][0], waypoints[0][1], start_theta])
    env.current_lin_vel = 0.0
    env.current_ang_vel = 0.0
    
    # CRITICAL V3 FIX: Update the distance trackers after teleporting the robot
    env.previous_distance = float(np.linalg.norm(env.current_pose[:2] - env.current_goal))
    env.ft_previous_distance = env.previous_distance
    
    # 3. Execute the policy
    obs = env._get_obs()
    trajectory = [env.current_pose[:2].copy()]
    outcome = "Trap (Timeout)"

    for _ in range(3000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append(env.current_pose[:2].copy())
        
        if terminated or truncated:
            tel = info.get("telemetry", {})
            if tel.get("rate_success", 0.0) == 1.0:
                outcome = "Success"
            elif tel.get("rate_crash", 0.0) == 1.0:
                outcome = "Crash"
            break

    return grid, waypoints, res, np.array(trajectory), outcome

def main():
    # Allow for loading without the .zip extension explicitly written
    actual_path = MODEL_PATH if MODEL_PATH.endswith(".zip") else MODEL_PATH + ".zip"
    if not os.path.exists(actual_path):
        print(f"[ERROR] Model not found at: {actual_path}")
        return

    print("Loading FineTuned V3 Model...")
    env = FineTuneBotEnv_V3()
    model = PPO.load(actual_path, env=env)

    maps_to_test = [
        ("Corner Gauntlet (Braking)", CornerGauntletMap),
        ("Pinch Point (Commitment)", PinchPointMap),
        ("Clutter Corridor (Weaving)", ClutterCorridorMap),
        ("Fork Trap (Branch Choice)", ForkTrapMap),
        ("Blind Corner (Ambush)", BlindCornerMap),
    ]

    # Create a 2x3 grid to fit all 5 maps
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        # Hide the 6th empty subplot
        if i >= len(maps_to_test):
            ax.axis('off')
            continue
            
        title, map_class = maps_to_test[i]
        print(f"Executing: {title}...")
        grid, waypoints, res, trajectory, outcome = evaluate_map(env, model, map_class)
        
        # Convert dimensions to physical meters for plotting
        h_px, w_px = grid.shape
        w_m, h_m = w_px / res, h_px / res

        # Draw occupancy grid
        ax.imshow(grid == 0, cmap='gray', origin='lower', extent=[0, w_m, 0, h_m])

        # Draw planned waypoints
        if len(waypoints) > 0:
            wx, wy = waypoints[:, 0], waypoints[:, 1]
            ax.plot(wx, wy, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax.scatter(wx, wy, c='yellow', s=60, zorder=5)
            ax.scatter(wx[0], wy[0], c='lime', s=120, edgecolors='black', zorder=6, label="Start")
            ax.scatter(wx[-1], wy[-1], c='blue', s=120, marker='X', edgecolors='black', zorder=6, label="Goal")

        # Draw executed trajectory
        if len(trajectory) > 0:
            tx, ty = trajectory[:, 0], trajectory[:, 1]
            # Color code the trajectory line based on the outcome
            if outcome == "Success":
                color = 'cyan'
            elif outcome == "Crash":
                color = 'red'
            else:
                color = 'orange'
                
            ax.plot(tx, ty, color=color, linewidth=2, label=f"Path ({outcome})")

        ax.set_title(f"{title} | Outcome: {outcome}", fontsize=14, fontweight='bold')
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.legend(loc="upper right")
        ax.set_facecolor('#222222')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()