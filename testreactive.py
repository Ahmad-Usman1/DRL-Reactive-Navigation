import os
import numpy as np
import matplotlib.pyplot as plt
import math
from PeopleBotEnv import PeopleBotEnv

class ReactiveController:
    def __init__(self, env):
        self.env = env
        self.k_att = 1.4  # Slightly stronger attraction to goal
        self.k_rep = 0.8  # Stronger repulsion to stay away from the new bigger clutter
        self.safe_dist_norm = 0.35 # (0.35 * 5m = 1.75m detection zone)

    def predict(self, obs):
        num_rays = self.env.num_rays
        scan = obs[0:num_rays]
        goal_head = obs[num_rays + 1]
        
        # 1. Attractive Force
        f_att_x = math.cos(goal_head * np.pi)
        f_att_y = math.sin(goal_head * np.pi)

        # 2. Repulsive Force
        f_rep_x, f_rep_y = 0.0, 0.0
        angles = self.env.sensor_angles

        for i in range(num_rays):
            dist = scan[i]
            if dist < self.safe_dist_norm:
                strength = self.k_rep * (1.0 / (dist + 0.01) - 1.0 / self.safe_dist_norm)
                f_rep_x -= strength * np.cos(angles[i])
                f_rep_y -= strength * np.sin(angles[i])

        # 3. Decision
        total_x = self.k_att * f_att_x + f_rep_x
        total_y = self.k_att * f_att_y + f_rep_y
        desired_heading = np.arctan2(total_y, total_x)
        
        # 4. Action Mapping
        lin_action = np.cos(desired_heading) * 0.9
        ang_action = (desired_heading / np.pi) * 1.5 # Boost turn response
        
        # 5. Hard Emergency Brake
        if np.min(scan[self.env.front_indices]) < 0.12: # ~0.6m
            lin_action = -0.3
            ang_action = 1.0 if goal_head > 0 else -1.0

        return [np.clip(lin_action, -1, 1), np.clip(ang_action, -1, 1)]

def run_validation_suite(num_episodes=20):
    output_dir = "reactive_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    env = PeopleBotEnv()
    env.set_difficulty(1) # Testing the "Knowledge Chasm"
    controller = ReactiveController(env)
    
    print(f"--- Booting Reactive Validation Suite | Targets: {num_episodes} Maps ---")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        
        # Telemetry storage
        path_history = []
        v_history = []
        w_history = []
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = controller.predict(obs)
            obs, reward, terminated, truncated, info = env.step(np.array(action))
            
            # Record data
            path_history.append(env.current_pose[:2].copy())
            v_history.append(env.current_lin_vel)
            w_history.append(env.current_ang_vel)

        # --- DATA VISUALIZATION ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 1. Map & Path Plot
        ax1.imshow(env.map_grid == 0, cmap='gray', origin='lower')
        path_pts = np.array(path_history) * env.resolution
        ax1.plot(path_pts[:, 0], path_pts[:, 1], color='cyan', label='Robot Path', linewidth=2)
        
        # Draw Waypoints
        wp_px = np.array(env.waypoints) * env.resolution
        ax1.scatter(wp_px[:, 0], wp_px[:, 1], c='red', s=20, label='Waypoints')
        ax1.set_title(f"Map {ep+1} | Success: {info['telemetry']['rate_success']}")
        ax1.legend()

        # 2. Velocity Graph
        steps = np.arange(len(v_history))
        ax2.plot(steps, v_history, label='Lin Vel (m/s)', color='blue')
        ax2.plot(steps, w_history, label='Ang Vel (rad/s)', color='orange', alpha=0.6)
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Velocity")
        ax2.set_title("Robot Dynamics Profile")
        ax2.legend()

        # Save result
        plt.tight_layout()
        plt.savefig(f"{output_dir}/test_run_{ep+1:02d}.png")
        plt.close()
        
        print(f"Map {ep+1:02d} Saved. Outcome: {'Success' if info['telemetry']['rate_success'] else 'Failure'}")

    print(f"\n--- Done! Check the '{output_dir}' folder for the report. ---")

if __name__ == "__main__":
    run_validation_suite()