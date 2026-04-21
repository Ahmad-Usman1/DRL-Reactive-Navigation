import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import math

# Project Imports
from MapBank import MapBank
from dwa_model import dwa_control, DWAConfig
from PeopleBotEnv import fast_raycast # Reusing your C-compiled raycaster

# ─── CONFIGURATION ────────────────────────────────────────────────────────
UPDATE_RATE_MS = 100
ACTUATOR_LAG_STEPS = 3 # Simulating the old 300ms delay

config = DWAConfig()
bank = MapBank(dataset_dir="training_maps")

# Global State
map_grid = None
waypoints = []
res = 50
current_goal_idx = 1
pose = np.zeros(5) # [x, y, theta, v, w]
action_history = [[0.0, 0.0] for _ in range(ACTUATOR_LAG_STEPS)]
running = False

# 17-Ray Sensor Specs
sensor_angles = np.deg2rad([90, 50, 30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30, -50, -90]).astype(np.float32)

# ─── MATPLOTLIB GUI SETUP ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.35)

ax.set_title("DWA Live Tuner (Actuator Lag: 300ms)")
ax.set_aspect('equal')

# Map and plots
map_img = ax.imshow(np.zeros((10,10)), cmap='gray', origin='lower')
path_line, = ax.plot([], [], 'y--', alpha=0.5)

# FIXED: Removed the comma so it doesn't try to unpack a PathCollection
goal_marker = ax.scatter([], [], c='lime', s=100, marker='X', zorder=5)
robot_marker, = ax.plot([], [], 'bo', markersize=10, zorder=4)
traj_line, = ax.plot([], [], 'c-', linewidth=2, zorder=3)
scan_scatter = ax.scatter([], [], c='red', s=10, zorder=3)

# Sliders
ax_alpha = plt.axes([0.15, 0.20, 0.65, 0.03])
ax_beta  = plt.axes([0.15, 0.15, 0.65, 0.03])
ax_gamma = plt.axes([0.15, 0.10, 0.65, 0.03])
ax_diff  = plt.axes([0.15, 0.05, 0.65, 0.03])

s_alpha = Slider(ax_alpha, 'Alpha (Heading)', 0.0, 5.0, valinit=1.0)
s_beta  = Slider(ax_beta,  'Beta (Clearance)', 0.0, 5.0, valinit=1.0)
s_gamma = Slider(ax_gamma, 'Gamma (Velocity)', 0.0, 5.0, valinit=1.0)
s_diff  = Slider(ax_diff,  'Map Difficulty', 0.0, 1.0, valinit=0.0)

# Buttons
ax_new_map = plt.axes([0.85, 0.05, 0.1, 0.04])
btn_new_map = Button(ax_new_map, 'New Map')
ax_play = plt.axes([0.85, 0.12, 0.1, 0.04])
btn_play = Button(ax_play, 'Play/Pause')

# ─── CORE LOGIC ───────────────────────────────────────────────────────────

def get_17_ray_obstacles(x, y, theta):
    """Fires 17 rays and converts hits back to (X,Y) coordinates for DWA."""
    scan = fast_raycast(x, y, theta, sensor_angles, map_grid, res, 3.0)
    obstacles = []
    for i, dist in enumerate(scan):
        if dist < 2.99: # Ignore rays that hit nothing
            glob_angle = theta + sensor_angles[i]
            ox = x + dist * math.cos(glob_angle)
            oy = y + dist * math.sin(glob_angle)
            obstacles.append([ox, oy])
    return obstacles

def generate_new_map(event=None):
    global map_grid, waypoints, res, pose, current_goal_idx, action_history, running
    
    running = False
    bank.set_difficulty(s_diff.val)
    map_grid, wps, res = bank.get_random_map()
    waypoints = np.array(wps)
    current_goal_idx = 1 if len(waypoints) > 1 else 0
    
    # Reset Pose
    start_x, start_y = waypoints[0]
    start_th = math.atan2(waypoints[1][1] - start_y, waypoints[1][0] - start_x) if len(waypoints) > 1 else 0.0
    pose = np.array([start_x, start_y, start_th, 0.0, 0.0])
    action_history = [[0.0, 0.0] for _ in range(ACTUATOR_LAG_STEPS)]
    
    # Update Graphics
    h_m, w_m = map_grid.shape[0]/res, map_grid.shape[1]/res
    map_img.set_data(map_grid == 0)
    map_img.set_extent([0, w_m, 0, h_m])
    
    path_line.set_data(waypoints[:,0], waypoints[:,1])
    
    # FIXED: Wrapped in double brackets to satisfy newer Matplotlib 2D array requirement
    goal_marker.set_offsets([[waypoints[current_goal_idx][0], waypoints[current_goal_idx][1]]])
    
    ax.set_xlim(0, w_m)
    ax.set_ylim(0, h_m)
    fig.canvas.draw_idle()

def toggle_play(event):
    global running
    running = not running

btn_new_map.on_clicked(generate_new_map)
btn_play.on_clicked(toggle_play)

def update(frame):
    global pose, current_goal_idx, action_history
    
    if not running or map_grid is None:
        return robot_marker, traj_line, scan_scatter, goal_marker
        
    # 1. Goal Tracking
    goal_x, goal_y = waypoints[current_goal_idx]
    dist_to_goal = math.hypot(goal_x - pose[0], goal_y - pose[1])
    if dist_to_goal < 1.0 and current_goal_idx < len(waypoints) - 1:
        current_goal_idx += 1
        goal_x, goal_y = waypoints[current_goal_idx]
        
        # FIXED: Wrapped in double brackets
        goal_marker.set_offsets([[goal_x, goal_y]])
        
    goal_pt = [goal_x, goal_y]

    # 2. Extract 17-Ray Obstacles
    ob_points = get_17_ray_obstacles(pose[0], pose[1], pose[2])
    
    # 3. Time-Travel Projection (Fixing the 300ms Actuator Lag)
    sim_pose = pose.copy()
    for past_u in action_history:
        sim_pose[2] += past_u[1] * config.dt
        sim_pose[0] += past_u[0] * math.cos(sim_pose[2]) * config.dt
        sim_pose[1] += past_u[0] * math.sin(sim_pose[2]) * config.dt
        sim_pose[3] = past_u[0]
        sim_pose[4] = past_u[1]

    # 4. Run DWA
    u, best_traj = dwa_control(
        sim_pose, config, goal_pt, ob_points,
        s_alpha.val, s_beta.val, s_gamma.val
    )
    
    # 5. Execute Hardware Physics (Executing the 300ms OLD command)
    delayed_u = action_history.pop(0)
    action_history.append(u)
    
    # Diff drive / tau_motor simplified approximation for GUI
    pose[2] += delayed_u[1] * config.dt
    pose[0] += delayed_u[0] * math.cos(pose[2]) * config.dt
    pose[1] += delayed_u[0] * math.sin(pose[2]) * config.dt
    pose[3] = delayed_u[0]
    pose[4] = delayed_u[1]
    
    # 6. Update Visuals
    # FIXED: Wrapped pose variables in brackets to satisfy sequence requirement
    robot_marker.set_data([pose[0]], [pose[1]])
    traj_line.set_data(best_traj[:, 0], best_traj[:, 1])
    
    if ob_points:
        ob_arr = np.array(ob_points)
        scan_scatter.set_offsets(ob_arr)
    else:
        scan_scatter.set_offsets(np.empty((0,2)))
        
    return robot_marker, traj_line, scan_scatter, goal_marker

# Initialize first map
generate_new_map()

# Start loop
ani = FuncAnimation(fig, update, interval=UPDATE_RATE_MS, blit=True, cache_frame_data=False)
plt.show()