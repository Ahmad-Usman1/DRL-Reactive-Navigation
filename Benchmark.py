import numpy as np
import math
from PeopleBotEnv import PeopleBotEnv
from MapGenerator import MapGenerator
from stable_baselines3 import PPO

# --- Baseline 1: DWA (simplified but representative) ---
def dwa_action(scan_data, heading_error, sensor_angles, lin_vel, max_lin, max_ang):
    """Simplified DWA: scores (v, w) pairs on heading + clearance."""
    best_score = -np.inf
    best_v, best_w = 0.0, 0.0
    for v in np.linspace(0.05, max_lin, 5):
        for w in np.linspace(-max_ang, max_ang, 7):
            # Simulate 0.5s forward: check if heading improves
            pred_heading = heading_error - w * 0.5
            heading_score = math.cos(pred_heading)
            # Clearance: penalize low lidar in the predicted forward direction
            front_mask = np.abs(sensor_angles) < np.deg2rad(45)
            clearance = np.min(scan_data[front_mask]) if front_mask.any() else 5.0
            clearance_score = min(clearance / 2.0, 1.0)
            score = heading_score * v + 0.5 * clearance_score - 0.3 * abs(w)
            if score > best_score:
                best_score = score
                best_v, best_w = v, w
    # Normalize to [-1, 1] action space
    return np.array([best_v / max_lin, best_w / max_ang])

# --- Baseline 2: APF (Attractive + Repulsive) ---
def apf_action(pose, goal, scan_data, sensor_angles, max_lin, max_ang):
    """Attractive force to goal, repulsive from obstacles."""
    # Attractive
    dx, dy = goal[0] - pose[0], goal[1] - pose[1]
    dist = math.hypot(dx, dy)
    att_x = dx / (dist + 1e-6)
    att_y = dy / (dist + 1e-6)
    # Repulsive
    rep_x, rep_y = 0.0, 0.0
    influence_dist = 1.5
    for i, d in enumerate(scan_data):
        if d < influence_dist:
            obs_angle = pose[2] + sensor_angles[i]
            strength = (1.0/max(d, 0.1) - 1.0/influence_dist) / (d**2 + 1e-6)
            rep_x -= math.cos(obs_angle) * strength * 0.5
            rep_y -= math.sin(obs_angle) * strength * 0.5
    # Combine
    total_x = att_x + rep_x
    total_y = att_y + rep_y
    desired_angle = math.atan2(total_y, total_x)
    heading_error = (desired_angle - pose[2] + math.pi) % (2*math.pi) - math.pi
    v = max_lin * max(0, math.cos(heading_error))
    w = np.clip(2.0 * heading_error, -max_ang, max_ang)
    return np.array([v / max_lin, w / max_ang])

# --- Evaluation Runner ---
def run_episode(env, policy_fn, max_steps=3000):
    obs, _ = env.reset()
    path_length = 0.0
    prev_pos = env.current_pose[:2].copy()
    total_jerk = 0.0
    prev_action = np.zeros(2)
    min_lidar_log = []
    
    for step in range(max_steps):
        action = policy_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Track metrics
        curr_pos = env.current_pose[:2].copy()
        path_length += np.linalg.norm(curr_pos - prev_pos)
        prev_pos = curr_pos.copy()
        
        jerk = np.linalg.norm(action - prev_action)
        total_jerk += jerk
        prev_action = action.copy()
        
        scan = obs[:env.num_rays] * env.max_sensor_range
        min_lidar_log.append(np.min(scan))
        
        # At the top of the for loop in run_episode:
        if step % 500 == 0:
            print(f"  Step {step}/{max_steps}, pos={env.current_pose[:2].round(2)}", end='\r')

        if terminated or truncated:
            success = "telemetry" in info and info["telemetry"]["rate_success"] == 1.0
            return {
                "success": success,
                "path_length": path_length,
                "smoothness": total_jerk / max(step, 1),
                "avg_clearance": np.mean(min_lidar_log),
                "steps": step
            }
    return {"success": False, "path_length": path_length, 
            "smoothness": total_jerk / max_steps, "avg_clearance": np.mean(min_lidar_log), "steps": max_steps}

def compute_spl(successes, optimal_lengths, actual_lengths):
    spl_per_ep = []
    for s, l_star, p in zip(successes, optimal_lengths, actual_lengths):
        if s:
            spl_per_ep.append(l_star / max(p, l_star))
        else:
            spl_per_ep.append(0.0)
    return np.mean(spl_per_ep)

def benchmark(model_path, n_episodes=50, difficulties=[0.1, 0.5, 1.0]):
    env = PeopleBotEnv()
    ppo_model = PPO.load(model_path)
    
    results = {"PPO": [], "DWA": [], "APF": []}
    
    for diff in difficulties:
        env.set_difficulty(diff)
        for _ in range(n_episodes // len(difficulties)):
            # Same map for all three — fair comparison
            env.reset()
            saved_map = env.map_grid.copy()
            saved_wp = env.waypoints.copy()
            
            def reset_to_same(e):
                e.reset()
                e.map_grid = saved_map
                e.waypoints = saved_wp
                e.current_goal = np.array(saved_wp[1]) if len(saved_wp) > 1 else np.array(saved_wp[0])
                e.current_goal_index = 1
                return e._get_obs()
            
            # PPO
            def ppo_policy(obs, e): 
                a, _ = ppo_model.predict(obs, deterministic=True)
                return a
            r_ppo = run_episode(env, ppo_policy)
            results["PPO"].append(r_ppo)
            
            # DWA on same map
            reset_to_same(env)
            def dwa_policy(obs, e):
                scan = obs[:e.num_rays] * e.max_sensor_range
                he = obs[e.num_rays + 1] * np.pi
                return dwa_action(scan, he, e.sensor_angles, e.current_lin_vel, e.max_lin_vel, e.max_ang_vel)
            r_dwa = run_episode(env, dwa_policy)
            results["DWA"].append(r_dwa)
            
            # APF on same map
            reset_to_same(env)
            def apf_policy(obs, e):
                return apf_action(e.current_pose, e.current_goal, 
                                  obs[:e.num_rays] * e.max_sensor_range,
                                  e.sensor_angles, e.max_lin_vel, e.max_ang_vel)
            r_apf = run_episode(env, apf_policy)
            results["APF"].append(r_apf)
    
    # Compute SPL and print table
    print(f"\n{'Algorithm':<10} {'SPL':>8} {'Success%':>10} {'Smoothness':>12} {'Clearance(m)':>14}")
    print("-" * 56)
    for algo, ep_results in results.items():
        successes = [r["success"] for r in ep_results]
        # Optimal length: straight-line distance between first and last waypoint
        opt_lengths = [5.0] * len(ep_results)  # Replace with actual straight-line dist per episode
        act_lengths = [r["path_length"] for r in ep_results]
        
        spl = compute_spl(successes, opt_lengths, act_lengths)
        success_pct = 100 * np.mean(successes)
        smoothness = np.mean([r["smoothness"] for r in ep_results])
        clearance = np.mean([r["avg_clearance"] for r in ep_results if r["success"]])
        
        print(f"{algo:<10} {spl:>8.3f} {success_pct:>9.1f}% {smoothness:>12.4f} {clearance:>13.3f}m")

if __name__ == "__main__":
    benchmark("./Performing_Models/BEANS_Continued_v2_Final_6817600_steps.zip")