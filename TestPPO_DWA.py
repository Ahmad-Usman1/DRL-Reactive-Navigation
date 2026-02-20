import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Arrow
import math
import time

# Import your Env
from PeopleBotEnv import PeopleBotEnv

try:
    from numba import njit
    print("Numba detected! Activating High-Performance Mode.")
except ImportError:
    print("Numba not found. Running in Slow Mode.")
    def njit(f): return f

# --- JIT KERNELS ---

@njit(fastmath=True)
def predict_trajectory(x, y, th, v, w, dt, steps):
    traj = np.zeros((steps, 2), dtype=np.float32)
    for i in range(steps):
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        th += w * dt
        traj[i, 0] = x
        traj[i, 1] = y
    return traj

@njit(fastmath=True)
def calculate_score(traj, obstacles, v, w, 
                    dist_weight, head_weight, vel_weight, 
                    max_v, max_accel_v,
                    gx, gy, rx, ry, rth, predict_time):
    
    # 1. MINIMUM DISTANCE
    EFFECTIVE_RANGE = 2.0 
    min_dist = EFFECTIVE_RANGE 
    
    num_pts = traj.shape[0]
    num_obs = obstacles.shape[0]
    
    if num_obs > 0:
        for i in range(num_pts):
            tx = traj[i, 0]
            ty = traj[i, 1]
            for j in range(num_obs):
                ox = obstacles[j, 0]
                oy = obstacles[j, 1]
                d = math.sqrt((tx-ox)**2 + (ty-oy)**2)
                if d < min_dist:
                    min_dist = d
    
    # 2. BRAKING CHECK (RELAXED)
    # Only enforce strict braking if we are moving fast
    if v > 0.3:
        braking_distance = (v**2) / (2.0 * max_accel_v) + 0.10 # Reduced buffer
        if min_dist < braking_distance:
            return -1e9 # Hard Safety Stop
    elif min_dist < 0.25:
        # If moving slow, only stop if we are literally about to hit
        return -1e9

    # 3. SCORING
    score_dist = min_dist / EFFECTIVE_RANGE

    # Heading
    final_heading = rth + (w * predict_time)
    ex, ey = traj[-1, 0], traj[-1, 1]
    angle_to_goal = math.atan2(gy - ey, gx - ex)
    error = angle_to_goal - final_heading
    while error <= -math.pi: error += 2*math.pi
    while error > math.pi: error -= 2*math.pi
    score_head = 1.0 - (abs(error) / math.pi)

    # Velocity
    score_vel = v / max_v

    return (dist_weight * score_dist) + (head_weight * score_head) + (vel_weight * score_vel)

# --- CONTROLLER ---

class DWAController:
    def __init__(self, env):
        self.env = env
        
        # --- BALANCED WEIGHTS ---
        self.dist_weight = 2.0   # Reduced from 4.0 (Was too scared)
        self.heading_weight = 1.8 
        self.vel_weight = 1.5    # Increased from 1.5 (Needs more drive)
        
        self.predict_time = 2.0      
        self.dt_sim = 0.2
        self.steps = int(self.predict_time / self.dt_sim)
        
        self.max_v = env.max_lin_vel    
        self.min_v = env.min_lin_vel 
        self.max_w = env.max_ang_vel    
        
        self.max_accel_v = 1.0       
        self.max_accel_w = 4.0 

    def get_action(self, obs, current_v, current_w):
        scan = obs[:16]
        
        # Dynamic Window
        dw_v_min = max(self.min_v, current_v - self.max_accel_v * self.env.dt)
        dw_v_max = min(self.max_v, current_v + self.max_accel_v * self.env.dt)
        dw_w_min = max(-self.max_w, current_w - self.max_accel_w * self.env.dt)
        dw_w_max = min(self.max_w, current_w + self.max_accel_w * self.env.dt)
        
        obstacles = self.get_obstacles_np(scan)
        
        v_samples = np.linspace(dw_v_min, dw_v_max, 15)
        w_samples = np.linspace(dw_w_min, dw_w_max, 25) 
        
        if dw_v_max > 0.05: v_samples[-1] = dw_v_max
        if abs(current_v) > 0.1: v_samples[0] = 0.0

        best_u = [0.0, 0.0]
        max_score = -float('inf')
        
        rx, ry, rth = self.env.current_pose
        gx, gy = self.env.current_goal

        for v in v_samples:
            for w in w_samples:
                traj = predict_trajectory(rx, ry, rth, v, w, self.dt_sim, self.steps)
                score = calculate_score(traj, obstacles, v, w, 
                                        self.dist_weight, self.heading_weight, self.vel_weight,
                                        self.env.max_lin_vel, self.max_accel_v,
                                        gx, gy, rx, ry, rth, self.predict_time)
                
                if score > max_score:
                    max_score = score
                    best_u = [v, w]
                    
        # Force Unstuck if score is invalid
        if max_score == -1e9 or max_score == -float('inf'):
            return np.array([0.0, 1.0]), -999.0

        # Normalize
        real_v, real_w = best_u
        if real_v >= 0:
            norm_v = real_v / self.env.max_lin_vel
        else:
            norm_v = -1.0 * (abs(real_v) / abs(self.env.min_lin_vel))
        norm_w = real_w / self.env.max_ang_vel
        
        return np.array([norm_v, norm_w]), max_score

    def get_obstacles_np(self, scan):
        valid_mask = scan < (self.env.max_sensor_range - 0.1)
        if not np.any(valid_mask):
            return np.zeros((0, 2), dtype=np.float32)
        dists = scan[valid_mask]
        angles = self.env.sensor_angles[valid_mask]
        ox = dists * np.cos(angles)
        oy = dists * np.sin(angles)
        rx, ry, rth = self.env.current_pose
        gx = rx + (ox * math.cos(rth) - oy * math.sin(rth))
        gy = ry + (ox * math.sin(rth) + oy * math.cos(rth))
        return np.column_stack((gx, gy)).astype(np.float32)

def main():
    env = PeopleBotEnv()
    obs, info = env.reset()
    dwa = DWAController(env)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.imshow(1 - env.map_grid, cmap='Greys', origin='lower', extent=[0, 20, 0, 20])
    
    robot_patch = Circle((0, 0), env.robot_radius, color='blue', zorder=5)
    ax.add_patch(robot_patch)
    arrow_patch = Arrow(0, 0, 0.5, 0, width=0.3, color='yellow', zorder=6)
    ax.add_patch(arrow_patch)
    goal_patch, = ax.plot([], [], 'g*', markersize=18, zorder=4)
    detection_circle = Circle((0,0), env.waypoint_radius, color='green', fill=False, linestyle='--', alpha=0.8)
    ax.add_patch(detection_circle)
    
    lidar_lines = [Line2D([], [], color='red', linewidth=0.5, alpha=0.6) for _ in range(16)]
    for l in lidar_lines: ax.add_line(l)
    
    stats_text = ax.text(0.5, 19.5, '', color='black', fontsize=10, 
                         fontfamily='monospace', fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))

    print("Starting Physics-Aware DWA (Unfrozen)...")
    
    RENDER_SKIP = 3
    
    try:
        for step in range(5000):
            cur_v = env.current_lin_vel
            cur_w = env.current_ang_vel
            action, score = dwa.get_action(obs, cur_v, cur_w)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % RENDER_SKIP == 0:
                rx, ry, theta = env.current_pose
                robot_patch.center = (rx, ry)
                arrow_patch.remove()
                arrow_patch = Arrow(rx, ry, 0.6*math.cos(theta), 0.6*math.sin(theta), width=0.3, color='yellow')
                ax.add_patch(arrow_patch)
                gx, gy = env.current_goal
                goal_patch.set_data([gx], [gy])
                detection_circle.center = (gx, gy)
                scan = obs[:16]
                angles = theta + env.sensor_angles 
                ex_all = rx + scan * np.cos(angles)
                ey_all = ry + scan * np.sin(angles)
                for i, line in enumerate(lidar_lines):
                    line.set_data([rx, ex_all[i]], [ry, ey_all[i]])
                stats_text.set_text(f"V: {cur_v:.2f} m/s\nScore: {score:.2f}")
                fig.canvas.draw()
                fig.canvas.flush_events()
            
            if terminated:
                if reward > 0: print("SUCCESS!"); ax.set_title("SUCCESS", color='green')
                else: print("CRASH!"); ax.set_title("CRASH", color='red')
                time.sleep(2)
                break
                
    except KeyboardInterrupt: pass
    plt.ioff(); plt.show()

if __name__ == "__main__":
    main()